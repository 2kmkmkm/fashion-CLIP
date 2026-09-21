import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv

# 1단계: LLM 질의 분석기

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. 단품 슬롯 스키마 정의
class SlotQuery(BaseModel):
    slot_name: str = Field(description="슬롯명: outer, top, bottom, shoes, hat 중 하나")
    category: Literal["outer", "top", "bottom", "shoes", "hat"] = Field(description="무신사 5대 대분류 카테고리")
    clip_query_en: str = Field(description="Fashion-CLIP 검색용 패션 영문 시각 묘사 (소재, 핏, 실루엣, 디테일)")
    color: Optional[str] = Field(None, description="색상 제약조건 (예: black, white, brown, blue, gray)")
    max_price: Optional[int] = Field(None, description="가격 상한선 (원 단위 정수)")

# 2. 전체 의도 스키마 정의
class ParsedIntent(BaseModel):
    search_type: Literal["single", "coordination"] = Field(description="단품 검색(single) vs 코디 세트 추천(coordination)")
    tpo_summary: str = Field(description="TPO 및 무드 요약 (한국어)")
    slots: List[SlotQuery] = Field(description="검색할 슬롯 리스트 (단품은 1개, 코디는 2개 이상)")

class QueryParser:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    def parse(self, user_query: str) -> ParsedIntent:
        system_prompt = """
        당신은 패션 커머스 플랫폼의 수석 AI 스타일리스트입니다.
        사용자의 한국어 질의를 분석하여 [단품 검색] 또는 [코디 세트 추천]으로 엄격하게 분류하고,
        패션 특화 임베딩 모델(Fashion-CLIP)에 입력할 최적의 영문 시각 키워드로 변환하세요.

        [핵심 분류 규칙]:
        1. search_type:
           - 사용자가 특정 단일 품목(아우터, 자켓, 상의, 셔츠, 바지, 신발, 모자 등) 1개만 명확히 찾고 있다면
             질의에 '오늘 날씨', '출근할 때', '여행 갈 때' 같은 상황 수식어가 붙어 있어도
             무조건 **"single"**로 분류해야 합니다! (예: "오늘 날씨에 맞는 아우터", "피크닉 갈 때 신을 신발")
           - search_type이 "single"일 때는 slots 리스트에 **반드시 1개의 슬롯만** 포함해야 합니다.
           - 머리부터 발끝까지 전신 착장이나 2개 이상의 품목 조합을 원할 때만 **"coordination"**입니다.
             (예: "나트랑 여행룩", "결혼식 하객룩 세트", "셔츠에 어울리는 바지 추천")
           - search_type이 "coordination"일 때는 slots 리스트에 **최소 2개 이상의 슬롯**을 구성해야 합니다.

        2. category: 반드시 ["outer", "top", "bottom", "shoes", "hat"] 5개 중 하나로 매핑.
        3. clip_query_en: 소재, 실루엣, 핏, 용도 등 시각적 패션 어휘 위주로 작성.
        """

        completion = client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            response_format=ParsedIntent,
            temperature=0.1
        )
        return completion.choices[0].message.parsed