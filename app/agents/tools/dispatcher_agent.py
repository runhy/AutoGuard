# 에이전트 기초 설계입니다.
import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv
from agents.intel_agent import IntelAgent # 분리한 에이전트 임포트


# 1. 환경 변수 로드
# 현재 실행 중인 core.py 파일의 절대 경로를 가져옵니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

# 그 폴더 안에 있는 .env 파일의 절대 경로를 만듭니다.
dotenv_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path)   # 경로 지정 api 로드

class AutoGuardAgent:
    def __init__(self, intel_agent):
        ''' OpenAI 클라이언트 초기화 및 에이전트 주입'''
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.assistant_id = None
        
        # 프롬프트 로드
        self.instruction = self._load_instruction()
        # 주입받은 인텔 에이전트 인스턴스 저장
        self.intel_agent = intel_agent

    def _load_instruction(self):
        ''' prompts/dispatcher.txt 파일을 읽어오는 내부 함수 '''
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'dispatcher.txt')      # window, linux 통합 경로
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def create_inspector(self):
        '''OpenAI 서버에 AutoGuard 에이전트를 실제로 생성합니다.'''
        assistant = self.client.beta.assistants.create(
            name='AutoGuard Dispatcher',
            instructions=self.instruction,
            model='gpt-4o',  # 성능을 위해 최신 모델 사용
            tools=[
                {'type': 'function', 'function': self._get_url_tool_schema()},
                {'type': 'function', 'function': self._get_email_tool_schema()},
                {'type': 'function', 'function': self._get_intel_tool_schema()}
            ]
        )
        self.assistant_id = assistant.id
        print(f'에이전트 생성 완료! ID: {self.assistant_id}')
        return assistant

    def _get_url_tool_schema(self):
        ''' 에이전트가 어떤 도구를 쓸 수 있는지 알려주는 명세서(Schema) '''
        return {
            'name': 'predict_url_malicious',
            'description': 'URL의 악성 여부를 ML 모델로 판별합니다.',
            'parameters': {     # 입력 값 목록
                'type': 'object',       # 피라미터들을 하나의 객체로 전달
                'properties': {         # 입력받을 변수들 나열
                    'url': {
                        'type': 'string', 'description': '분석할 URL 주소'
                    }
                },
                'required': ['url']     # url변수는 필수
            }
        }

    def _get_email_tool_schema(self):
        return {
            'name': 'predict_email_malicious',
            'description': '이메일 본문의 스팸/피싱 여부를 판별합니다.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string', 'description': '분석할 이메일 본문'}
                },
                'required': ['text']
            }
        }

    def _get_intel_tool_schema(self):
        """에이전트에게 검색 도구가 있음을 알리는 명세서"""
        return {
            'name': 'search_threat_intel',
            'description': '웹 검색을 통해 최신 보안 위협 정보 및 도메인 평판을 확인합니다.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': '검색할 악성코드 명칭, 파일 해시, 또는 URL'}
                },
                'required': ['query']
            }
        }

    
        
    # 에이전트 실행 처리 함수
    def run_agent(self, user_message):
        '''
        사용자 메시지를 처리하고 에이전트의 응답(또는 도구 호출 요청)을 받아옵니다.
        '''
        # 1. 새로운 채팅방(thread) 생성
        thread = self.client.beta.threads.create()      # 대화 내역 저장 목적
        
        # 2. 채팅방에 사용자 메시지 추가
        self.client.beta.threads.messages.create(
            thread_id=thread.id, 
            role='user',
            content=user_message
        )

        # 3. 에이전트 실행 (Run) 시작
        run = self.client.beta.threads.runs.create(     # 연결 -> 채팅창 메시지 + 에이전트
            thread_id=thread.id,
            assistant_id=self.assistant_id
        )

        print(f'[*] 분석 시작... (Run ID: {run.id})')    # 실행 출력문

        # 4. 루프 시작: 에이전트의 상태가 '완료'될 때까지 대기 및 도구 처리
        while True:
            # 상태 갱신(실시간 업데이트)
            run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            if run.status == 'completed':
                # 분석 완료 시 메시지 출력
                messages = self.client.beta.threads.messages.list(thread_id=thread.id)
                return messages.data[0].content[0].text.value       # 가장 최신 답변 텍스트 추출
            

            elif run.status == 'requires_action':
                # 에이전트가 도구 호출을 요청한 상태
                print('[!] 에이전트가 도구 호출을 요청했습니다.')
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    # 인텔 에이전트 도구 실행
                    if function_name == 'search_threat_intel':
                        result = self.intel_agent.search_web(arguments['query'])
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(result)
                        })
                    
                    # (계획)URL/Email 분석 도구 연결 로직

                # 결과 제출 후 루프 계속 진행
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                )

            elif run.status in ['failed', 'expired', 'cancelled']:
                # 실행 실패
                return f'[-] 에이전트 실행 실패: {run.status}'
            
            time.sleep(1)       # 자원 과소모 방지용

# 테스트 환경 실행
if __name__ == '__main__':
    intel = IntelAgent()
    agent = AutoGuardAgent(intel_agent=intel)
    agent.create_inspector()