import os
import sys
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 시스템 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 환경 변수 로드
load_dotenv(os.path.join(project_root, '.env'))
logger = logging.getLogger("AutoGuard-Advisor")

class AdvisorAgent:
    """
    최종 의사결정 에이전트:
    Analyzer/Intel의 분석 데이터와 KISA 보안 가이드를 결합하여 
    사용자가 읽기 쉬운 전문적인 '보안 권고 보고서'를 생성합니다.
    """
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY를 찾을 수 없습니다. .env 파일을 확인하세요.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.instruction = self._load_instruction()

    def _load_instruction(self):
        """ prompts/advisor.txt에서 페르소나 및 보고서 양식 로드 """
        prompt_path = os.path.join(current_dir, '..', 'prompts', 'advisor.txt')
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error("[-] advisor.txt 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
            return "너는 보안 컨설턴트야. 분석 결과와 가이드를 참고해 전문적인 대응 방안을 제시해줘."

    async def generate_final_advice(self, analysis_result: str, security_guide: str = "") -> str:
        """
        [Advisor Agent 최종 전략 로직]
        1. 위험도 점수(0-100) 자동 환산 및 등급 매칭
        2. 내부/외부 데이터의 기술적 근거를 사용자 친화적 용어로 번역
        3. KISA 가이드를 통한 실효성 있는 대응책 매칭
        """
        try:
            # KISA 가이드 및 전문 페르소나 주입을 위한 프롬프트 보강
            guide_context = ""
            if security_guide:
                guide_context = (
                    f"\n\n### [참조: KISA 보안 가이드라인 전문 데이터]\n{security_guide}\n"
                    f"**전략 지침:** 위 가이드에서 유사도 점수 등 시스템 메시지는 폐기하십시오. "
                    f"현재 탐지된 위협 유형(피싱/악성코드 등)에 직결되는 '실행 지침'만 발췌하여 리포트에 녹여내십시오."
                )

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.instruction},
                    {
                        "role": "user", 
                        "content": (
                            f"다음 분석 데이터를 바탕으로 [Output Format]에 맞춰 최종 보안 전략 리포트를 작성하십시오.\n\n"
                            f"[Raw 분석 데이터]\n{analysis_result}"
                            f"{guide_context}\n\n"
                            f"**핵심 요구사항:**\n"
                            f"1. 첫 줄에 반드시 '[위험도 점수]: XX/100' 형식을 유지할 것.\n"
                            f"2. 등급은 점수에 따라 Critical(90+), High(70+), Medium(40+), Low(40미만)로 분류할 것.\n"
                            f"3. 엔트로피는 '파일 구조의 복잡성/암호화', 섹션명은 '코드 변조 흔적'으로 풀어서 설명할 것."
                        )
                    }
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[-] Advisor 리포트 생성 실패: {e}")
            return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"

    async def summarize_rag_chunks(self, chunks: list[str], threat_type: str = "보안 위협") -> str:
        """
        [RAG 결과 요약 메서드]
        벡터 DB에서 검색된 파편화된 PDF 청크들을
        GPT-4o-mini를 통해 자연스러운 KISA 보안 권고문으로 재가공합니다.

        Args:
            chunks: RAG 검색 결과 청크 텍스트 목록
            threat_type: 현재 분석 유형 (URL / Email / File 등)

        Returns:
            자연스럽게 재작성된 보안 권고문 문자열
        """
        try:
            # 청크가 없으면 바로 반환
            if not chunks:
                return ""

            # 청크 목록을 하나의 텍스트로 합치기
            joined = "\n\n".join(chunks)

            response = await self.client.chat.completions.create(
                # 요약 작업은 mini 모델로 충분 (비용 절감)
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "너는 KISA(한국인터넷진흥원) 보안 가이드 전문가야. "
                            "제공된 보안 가이드 원문 파편들을 분석하여, "
                            "현재 위협 유형과 관련된 핵심 내용만 추려서 "
                            "3~5줄의 자연스러운 한국어 권고문으로 재작성해줘. "
                            "아래 규칙을 반드시 지켜:\n"
                            "1. '유사도 점수', '검색 결과', '청크 번호' 등 시스템 내부 정보는 절대 포함하지 마.\n"
                            "2. 문장은 '~하십시오', '~을 권장합니다' 형식의 신뢰감 있는 어조로 작성해.\n"
                            "3. KISA 공식 사이트(seed.kisa.or.kr 등) 언급이 원문에 있으면 포함해.\n"
                            "4. 현재 위협 유형과 무관한 내용은 과감히 제외해."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"위협 유형: {threat_type}\n\n"
                            f"[KISA 가이드 원문 파편]\n{joined}\n\n"
                            "위 내용을 바탕으로 핵심 보안 권고문을 작성해줘."
                        )
                    }
                ],
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"[-] RAG 요약 실패: {e}")
            # 실패 시 첫 번째 청크 300자로 fallback
            return chunks[0][:300] if chunks else ""