# tools/dispatcher_agent.py
# Dispatcher Agent - 사용자 요청을 분석하여 적절한 도구를 호출하고 최종 리포트를 생성합니다.
import os
import sys
import json
import asyncio
import logging
# [수정] 비동기 방식의 AsyncOpenAI 임포트
from openai import AsyncOpenAI 
from dotenv import load_dotenv


# ----------------------------------------------------------
# [수정] 경로 설정 및 모듈 임포트 최적화
# ----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) # tools 폴더 위치
# 최상위 .env 파일 경로 자동 탐색 (상위로 3단계 이동)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
# [추가] 경로 설정 및 임포트
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# .env 로드
load_dotenv(os.path.join(project_root, '.env'))

# # [수정] 모듈 임포트: 실행 환경에 구애받지 않도록 시도
# try:
#     from app.agents.tools.intel_agent import IntelAgent
# except ImportError:
#     try:
#         from intel_agent import IntelAgent
#     except ImportError:
#         from agents.tools.intel_agent import IntelAgent

# 에이전트 임포트
from app.agents.tools.intel_agent import IntelAgent
from app.agents.tools.analyzer_agent import AnalyzerAgent
from app.agents.tools.advisor_agent import AdvisorAgent # Advisor 추가


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoGuardAgent:
    """
    AutoGuard 디스패처 에이전트.
    사용자의 요청을 분석하여 URL, 위협 인텔리전스, 파일 해시 분석 도구를 적절히 호출합니다.
    """
    def __init__(self, intel_agent):
        ''' [수정] OpenAI 클라이언트를 AsyncOpenAI로 변경하여 비동기 지원 '''
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY를 찾을 수 없습니다. .env 파일을 확인하세요.")
            
        self.client = AsyncOpenAI(api_key=api_key)
        self.assistant_id = None
        
        # [수정] 함수명 언더바 1개로 통일 (_load_instruction)
        self.instruction = self._load_instruction()
        self.intel_agent = intel_agent
        self.analyzer = AnalyzerAgent()
        self.advisor = AdvisorAgent() # Advisor 초기화

    def _load_instruction(self):
        ''' prompts/dispatcher.txt 파일을 읽어오는 내부 함수 '''
        # 경로를 현재 파일 기준으로 유연하게 설정
        prompt_path = os.path.join(current_dir, '..', 'prompts', 'dispatcher.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    async def create_inspector(self):
        ''' OpenAI 서버에 AutoGuard 에이전트를 실제로 생성합니다. '''
        assistant = await self.client.beta.assistants.create(
            name='AutoGuard Dispatcher',
            instructions=self.instruction,
            model='gpt-4o',
            tools=[
                {'type': 'function', 'function': self._get_url_tool_schema()},
                {'type': 'function', 'function': self._get_intel_tool_schema()},
                {'type': 'function', 'function': self._get_email_tool_schema()}, # 추가
                {'type': 'function', 'function': self._get_file_tool_schema()}   # 추가
            ]
        )
        self.assistant_id = assistant.id
        print(f'[*] 에이전트 생성 완료 (ID: {self.assistant_id})')
        return assistant

    # ------------------------------------------------------
    # [도구 스키마 정의]
    # ------------------------------------------------------
    def _get_url_tool_schema(self):
        # URL 악성 여부 분석 도구 스키마
        return {
            'name': 'predict_url_malicious',
            'description': 'URL 악성 여부 분석',
            'parameters': {
                'type': 'object',
                'properties': {'url': {'type': 'string'}},
                'required': ['url']
            }
        }

    def _get_intel_tool_schema(self):
        return {
            'name': 'search_threat_intel',
            # 설명을 더 구체적으로 변경 (2025-2026 최신 정보라는 키워드 포함)
            'description': '2025년 및 2026년의 실시간 보안 뉴스, 최신 랜섬웨어 트렌드, 새로운 취약점 정보를 인터넷에서 실시간으로 검색할 때 반드시 사용해야 하는 도구입니다. 모델 내부 지식에 없는 최신 위협 정보를 수집합니다.',
            'parameters': {
                'type': 'object',
                'properties': {'query': {'type': 'string', 'description': '검색할 보안 관련 키워드'}},
                'required': ['query']
            }
    }

    # [추가] path 파라미터 추가
    def _get_file_tool_schema(self):
        return {
            'name': 'predict_file_malicious',
            'description': '파일의 SHA-256 해시값 또는 로컬 경로를 기반으로 악성 여부를 분석합니다.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'file_hash': {'type': 'string', 'description': 'SHA-256 해시 문자열'},
                    'path': {'type': 'string', 'description': '파일의 로컬 임시 경로 (있을 경우 반드시 포함)'}
                },
                'required': ['file_hash']
            }
        }
    # [추가] email 스키마
    def _get_email_tool_schema(self):
        return {
            'name': 'predict_email_malicious',
            'description': '이메일 본문 텍스트의 스팸/피싱 여부를 분석합니다.',
            'parameters': {
                'type': 'object',
                'properties': {'text': {'type': 'string'}},
                'required': ['text']
            }
        }
    # ------------------------------------------------------
    # [핵심 로직] 에이전트 실행 루프
    # ------------------------------------------------------
    async def run_agent(self, user_message):
        ''' 사용자 메시지를 처리하고 필요한 도구를 호출하여 최종 답변 생성 '''
        # [수정] await 추가
        # 1. 대화 스레드 생성
        thread = await self.client.beta.threads.create()

        # 2. 사용자 메시지 추가
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

        # 4. 상태 감시 루프
        while True:
            run = await self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            if run.status == 'completed':
                messages = await self.client.beta.threads.messages.list(thread_id=thread.id)
                # 1차 분석 결과 (GPT의 Raw 답변)
                intermediate_response = messages.data[0].content[0].text.value
                
                # [수정] Advisor 엔진을 가동하여 '최종 보안 리포트'로 변환
                logger.info("[*] Advisor 엔진 가동: 리포트 최적화 중...")
                final_report = await self.advisor.generate_final_advice(intermediate_response)
                return final_report

            elif run.status == 'requires_action':
                # 도구 호출 요청 → 실행 후 결과 제출
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    print(f'[*] 도구 호출 실행: {fn_name}')

                    # [수정/추가] 각 도구 이름에 맞게 우리 모델 호출 연결
                    if fn_name == 'predict_url_malicious':
                        target_url = args.get('url')
                        # 1. 우리 모델 분석
                        # 2. 외부 인텔 조회
                        # [핵심] 두 결과를 합쳐서 전달 (덮어쓰지 않음)
                        result = {
                            "internal": self.analyzer.analyze_url(target_url),
                            "external": await self.intel_agent.search_web(target_url)
                        }
                        
                    elif fn_name == 'predict_email_malicious':
                        result = self.analyzer.analyze_email(args.get('text'))
                        
                    elif fn_name == 'predict_file_malicious':
                        # [하이브리드 분석] 내부 모델(mal.py) + 외부 인텔(VirusTotal 등)
                        target_hash = args.get('file_hash')
                        target_path = args.get('path')
                        
                        intel_res = await self.intel_agent.predict_file_malicious(target_hash)
                        ml_res = self.analyzer.analyze_file(target_path) if target_path else {"info": "path not provided"}
                        result = {"internal_ml_analysis": ml_res, "external_intelligence": intel_res}

                    elif fn_name == 'search_threat_intel':
                        # [수정] Tavily 검색 엔진(IntelAgent)을 실제로 호출하는 구간
                        # args.get('query')로 에이전트가 생성한 검색어를 넘겨줌
                        result = await self.intel_agent.search_web(args.get('query'))
    
                        # 만약 결과가 너무 길어서 에러나면 아래처럼 살짝 잘라야함 (선택사항)
                        # result = str(result)[:3000]

                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(result)
                    })

                await self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, 
                    run_id=run.id, 
                    tool_outputs=tool_outputs
                )

            elif run.status in ['failed', 'expired', 'cancelled']:  
                return f'[-] 에이전트 실행 실패: {run.status}'
            
            # 이벤트 루프 제어권 양보 (1초 대기)
            await asyncio.sleep(1) 

# ==========================================================
# 테스트 실행부
# ==========================================================
if __name__ == '__main__':
    async def main():
        # IntelAgent 초기화 (내부에서 환경변수 사용하므로 경로 확인 필수)
        print("[!] 프로그램을 시작합니다...") # 이 메시지가 뜨는지 확인!
        intel = IntelAgent()
        agent = AutoGuardAgent(intel_agent=intel)
        
        print("[!] OpenAI Assistant 생성 중... (잠시만 기다려주세요)")
        await agent.create_inspector()
        
        # [수정] 질문을 '검색'이 필요한 내용으로 변경!
        test_query = "최근 유행하는 LockBit 4.0 랜섬웨어에 대해 분석해줘. 내부 분석 엔진 결과랑 2026년 최신 보안 뉴스 증거를 합쳐서 보고해."
        
        # 파일 해시 분석 테스트용 질문 (필요시 사용)
        # test_query = "이 파일 해시 분석해줘: 275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f"
        print(f"\n[사용자 요청]: {test_query}")
        response = await agent.run_agent(test_query)
        print(f"\n[최종 분석 결과]\n{response}")
    asyncio.run(main())