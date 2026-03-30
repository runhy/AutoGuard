# app/agents/tools/advisor_agent.py

import os
import sys
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 시스템 경로 설정: 상위 폴더의 모듈이나 .env 파일을 참조하기 위해 경로를 동적으로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 환경 변수 로드: OpenAI API Key 등 보안이 필요한 설정값.
load_dotenv(os.path.join(project_root, '.env'))
logger = logging.getLogger("AutoGuard-Advisor")

class AdvisorAgent:
    """
    최종 의사결정 에이전트:
    Analyzer(내부 분석)와 Intel(외부 정보)이 도출한 JSON 형태의 파편화된 데이터를 
    사용자가 읽기 쉬운 '보안 권고 보고서'로 변환합니다.
    """
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY를 찾을 수 없습니다. .env 파일을 확인하세요.")
        
        # 비동기 통신을 위한 OpenAI 클라이언트 초기화
        self.client = AsyncOpenAI(api_key=api_key)
        # advisor.txt에 정의된 페르소나 및 지침 로드
        self.instruction = self._load_instruction()

    def _load_instruction(self):
        """ 
        prompts/advisor.txt에서 에이전트의 페르소나와 보고서 양식을 읽어옵니다. 
        이 파일의 내용에 따라 에이전트의 말투와 전문성이 결정됩니다.
        """
        prompt_path = os.path.join(current_dir, '..', 'prompts', 'advisor.txt')
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error("[-] advisor.txt 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
            return "너는 보안 컨설턴트야. 분석 결과를 요약하고 대응 방안을 제시해줘."       # 기본 설정값

    async def generate_final_advice(self, analysis_result: str) -> str:
        """
        핵심 로직:
        1. Analyzer/Intel이 뱉어낸 복잡한 raw 데이터를 입력받습니다.
        2. GPT-4o 모델에게 전달하여 '보안 전문가'의 관점으로 재해석하게 합니다.
        3. 마크다운 형식의 최종 리포트를 반환합니다.
        """
        try:
            # [Chat Completion API 사용 이유]
            # 도구를 직접 실행할 필요가 없으므로 무거운 Assistant API보다 
            # 빠르고 저렴하며 제어가 쉬운 Chat API를 사용합니다.
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    # 시스템 메시지에 페르소나(advisor.txt) 주입
                    {"role": "system", "content": self.instruction},
                    # 사용자 메시지에 수집된 데이터 주입
                    {"role": "user", "content": f"아래 분석 데이터를 바탕으로 최종 권고 리포트를 작성해줘:\n\n{analysis_result}"}
                ],
                # '창의성'보다는 '일관성'과 '냉정함'을 위해 낮게 설정
                temperature=0.2 
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[-] Advisor 리포트 생성 실패: {e}")
            return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"