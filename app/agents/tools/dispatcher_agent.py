import os
import time
import json
import asyncio
import logging
# [수정] 동기 방식의 OpenAI 대신 비동기 방식의 AsyncOpenAI 임포트
from openai import AsyncOpenAI 
from dotenv import load_dotenv

# [모듈 로드] 동일 폴더 내의 IntelAgent 임포트
try:
    from intel_agent import IntelAgent
except ImportError:
    from agents.tools.intel_agent import IntelAgent

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. 환경 변수 및 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '..', '..', '.env')
load_dotenv(dotenv_path)

class AutoGuardAgent:
    """
    AutoGuard 디스패처 에이전트.
    사용자의 요청을 분석하여 URL, 위협 인텔리전스, 파일 해시 분석 도구를 적절히 호출합니다.
    """
    def __init__(self, intel_agent):
        ''' [수정] OpenAI 클라이언트를 AsyncOpenAI로 변경하여 비동기 지원 '''
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.assistant_id = None
        
        # 프롬프트 로드
        self.instruction = self._load_instruction()
        # 주입받은 인텔 에이전트 인스턴스 저장
        self.intel_agent = intel_agent

    def _load_instruction(self):
        ''' prompts/dispatcher.txt 파일을 읽어오는 내부 함수 '''
        prompt_path = os.path.join(current_dir, '..', 'prompts', 'dispatcher.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    # [수정] 어시스턴트 생성 시에도 await가 필요하므로 async def로 변경
    async def create_inspector(self):
        ''' OpenAI 서버에 AutoGuard 에이전트를 실제로 생성합니다. '''
        assistant = await self.client.beta.assistants.create(
            name='AutoGuard Dispatcher',
            instructions=self.instruction,
            model='gpt-4o',
            tools=[
                {'type': 'function', 'function': self._get_url_tool_schema()},
                {'type': 'function', 'function': self._get_intel_tool_schema()},
                {'type': 'function', 'function': self._get_file_tool_schema()}
            ]
        )
        self.assistant_id = assistant.id
        print(f'[*] 에이전트 생성 완료! ID: {self.assistant_id}')
        return assistant

    # ------------------------------------------------------
    # [도구 스키마 정의] GPT가 어떤 도구를 쓸지 결정하는 명세서
    # ------------------------------------------------------
    def _get_url_tool_schema(self):
        return {
            'name': 'predict_url_malicious',
            'description': 'URL의 악성 여부를 VirusTotal 및 SafeBrowsing으로 판별합니다.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'url': {'type': 'string', 'description': '분석할 URL 주소'}
                },
                'required': ['url']
            }
        }

    def _get_intel_tool_schema(self):
        return {
            'name': 'search_threat_intel',
            'description': '웹 검색을 통해 최신 보안 위협 정보 및 도메인 평판을 확인합니다.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': '검색할 키워드 또는 위협 요소'}
                },
                'required': ['query']
            }
        }

    def _get_file_tool_schema(self):
        return {
            'name': 'predict_file_malicious',
            'description': '파일의 SHA-256 해시값을 기반으로 악성 파일 여부를 정밀 분석합니다.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'file_hash': {'type': 'string', 'description': '분석할 파일의 SHA-256 해시 문자열'}
                },
                'required': ['file_hash']
            }
        }

    # ------------------------------------------------------
    # [핵심 로직] 에이전트 실행 및 비동기 도구 호출 루프
    # ------------------------------------------------------
    async def run_agent(self, user_message):
        ''' 사용자 메시지를 처리하고 필요한 도구를 호출하여 최종 답변 생성 '''
        
        # [수정] 모든 self.client 호출 앞에 await 추가
        # 1. 대화 스레드 생성
        thread = await self.client.beta.threads.create()
        
        # 2. 메시지 추가
        await self.client.beta.threads.messages.create(
            thread_id=thread.id, 
            role='user',
            content=user_message
        )

        # 3. 에이전트 실행
        run = await self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant_id
        )
        print(f'[*] 분석 시작... (Run ID: {run.id})')

        # 4. 루프 시작: 상태 감시 및 도구 실행
        while True:
            # [수정] 상태 확인(retrieve) 시 await를 걸어야 서버가 멈추지 않고 다른 요청을 처리합니다.
            run = await self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            if run.status == 'completed':
                # 최종 답변 추출 시에도 await
                messages = await self.client.beta.threads.messages.list(thread_id=thread.id)
                return messages.data[0].content[0].text.value

            elif run.status == 'requires_action':
                print('[!] 에이전트가 도구 호출을 요청했습니다.')
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if fn_name == 'predict_file_malicious':
                        result = await self.intel_agent.predict_file_malicious(args.get('file_hash'))
                    
                    elif fn_name == 'predict_url_malicious':
                        result = await self.intel_agent.search_web(args.get('url'))
                    
                    else:
                        result = await self.intel_agent.search_web(args.get('query'))

                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(result)
                    })

                # [수정] 도구 결과 제출 시 await 추가
                await self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, 
                    run_id=run.id, 
                    tool_outputs=tool_outputs
                )

            elif run.status in ['failed', 'expired', 'cancelled']:
                return f'[-] 에이전트 실행 실패: {run.status}'
            
            # [중요] 1초 대기하며 이벤트 루프에 제어권을 넘깁니다. 
            await asyncio.sleep(1) 

# ==========================================================
# [테스트 실행부]
# ==========================================================
if __name__ == '__main__':
    async def main():
        intel = IntelAgent()
        agent = AutoGuardAgent(intel_agent=intel)
        # [수정] 테스트 실행 시에도 await 추가
        await agent.create_inspector()

        test_query = "이 파일 해시가 위험한지 분석해줘: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        response = await agent.run_agent(test_query)
        print(f"\n[최종 분석 결과]\n{response}")

    asyncio.run(main())