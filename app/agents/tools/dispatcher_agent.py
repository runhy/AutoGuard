import os
import sys
import json
import asyncio

# ==========================================================
# [중요] 경로 에러 완벽 해결
# 실행 파일 위치에 관계없이 'app' 폴더를 패키지 루트로 강제 인식시킵니다.
# ==========================================================
current_file = os.path.abspath(__file__)
# agents/tools/dispatcher_agent.py -> 상위로 3번 이동하면 'app' 폴더
app_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

if app_root not in sys.path:
    sys.path.insert(0, app_root)

# [수정 포인트] 경로 확보 후 IntelAgent 임포트
try:
    from agents.intel_agent import IntelAgent
    print(f"[+] 모듈 로드 성공: agents.intel_agent")
except ImportError:
    # 패키지 구조 오인식 대비 보조 임포트 로직
    sys.path.append(os.path.dirname(os.path.dirname(current_file)))
    from intel_agent import IntelAgent
    print(f"[+] 모듈 로드 성공 (보조 경로 활용)")

# 나머지 라이브러리 임포트
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(app_root, '.env'))

class AutoGuardAgent:
    """
    AutoGuard 디스패처 에이전트.
    IntelAgent 인스턴스를 활용하여 위협을 분석하고 최종 리포트를 생성합니다.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.intel_agent = IntelAgent() # 설계 엔진 주입
        self.assistant_id = None
        self.instruction = self._load_instruction()

    def _load_instruction(self):
        # dispatcher.txt 프롬프트 파일 동적 로드
        prompt_path = os.path.join(app_root, 'agents', 'prompts', 'dispatcher.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def create_inspector(self):
        """ Assistants API 설정 및 분석 도구(Function) 등록 """
        assistant = self.client.beta.assistants.create(
            name='AutoGuard Dispatcher',
            instructions=self.instruction,
            model='gpt-4o',
            tools=[
                {'type': 'function', 'function': self._get_url_tool_schema()},
                {'type': 'function', 'function': self._get_intel_tool_schema()}
            ]
        )
        self.assistant_id = assistant.id
        print(f'[*] 에이전트 생성 완료 (ID: {self.assistant_id})')
        return assistant

    def _get_url_tool_schema(self):
        return {'name': 'predict_url_malicious', 'description': 'URL 악성 여부 분석', 
                'parameters': {'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']}}

    def _get_intel_tool_schema(self):
        return {'name': 'search_threat_intel', 'description': '최신 위협 인텔리전스 정보 조회', 
                'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}

    async def run_agent(self, user_message):
        """ 사용자 요청 분석 및 도구 호출 루프 """
        thread = self.client.beta.threads.create()
        self.client.beta.threads.messages.create(thread_id=thread.id, role='user', content=user_message)
        run = self.client.beta.threads.runs.create(thread_id=thread.id, assistant_id=self.assistant_id)

        while True:
            run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            
            if run.status == 'completed':
                messages = self.client.beta.threads.messages.list(thread_id=thread.id)
                return messages.data[0].content[0].text.value
            
            elif run.status == 'requires_action':
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    print(f'[*] 도구 호출 실행: {fn_name}')

                    # 모든 외부 위협 분석을 IntelAgent 인터페이스로 통합
                    target = args.get('url') or args.get('query')
                    result = await self.intel_agent.search_web(target)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id, 
                        "output": json.dumps(result)
                    })
                
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, 
                    run_id=run.id, 
                    tool_outputs=tool_outputs
                )
            
            elif run.status in ['failed', 'expired', 'cancelled']:
                return f"[-] 에이전트 실행 실패: {run.status}"
            
            await asyncio.sleep(1)

# --- 메인 테스트 실행부 ---
if __name__ == '__main__':
    async def main():
        agent = AutoGuardAgent()
        agent.create_inspector()
        
        # 기존에 url은 화이트 리스트에 있어서 진짜 악성 url로 테스트 -> 악성 판별 완료
        test_url = "http://testsafebrowsing.appspot.com/s/malware.html"
        print(f"\n[*] 분석 요청 테스트: {test_url}")
        
        res = await agent.run_agent(f"{test_url} 이 URL의 안전성을 검사해줘.")
        print(f"\n[최종 분석 리포트]\n{res}")

    asyncio.run(main())