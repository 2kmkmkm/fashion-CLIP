import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SlotQuery(BaseModel):
    slot_name: str = Field(description="슬롯명: outer, top, bottom, shoes, hat 중 하나")
    category: Literal["outer", "top", "bottom", "shoes", "hat"] = Field(description="무신사 5대 대분류 카테고리")
    clip_query_en: str = Field(description="Fashion-CLIP 검색용 패션 영문 시각 묘사")
    max_price: Optional[int] = Field(description="Soft-Margin 버퍼(1.3배)가 곱해진 해당 슬롯의 가격 상한선")
    color: Optional[str] = Field(description="질의에서 언급된 색상 (예: black, blue, white, gray, navy, red 등, 없으면 null)")

class ParsedIntent(BaseModel):
    search_type: Literal["single", "coordination"] = Field(description="단품 검색 vs 코디 세트 추천")
    total_budget: Optional[int] = Field(description="사용자가 언급한 총예산 (원 단위 정수, 없으면 null)")
    season: Literal["SS", "FW", "ALL"] = Field(description="질의에서 파악된 계절감 ('SS', 'FW', 'ALL' 중 하나)")
    tpo_summary: str = Field(description="TPO 및 무드 요약")
    slots: List[SlotQuery]

class QueryParser:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    def parse(self, user_query: str) -> ParsedIntent:
        system_prompt = """
        당신은 패션 커머스 플랫폼의 수석 AI 스타일리스트입니다.
        사용자의 한국어 질의를 분석하여 [단품 검색] 또는 [코디 세트 추천]으로 분류하세요.
        
        [추가 규칙]:
        1. total_budget: 사용자가 언급한 예산(예: "20만원대")을 정수(200000)로 추출하세요. 없으면 null.
        2. slots의 max_price: 슬롯별 예산을 나눌 때, Qdrant 검색 시 여유를 주기 위해 **산정된 기준 예산의 1.3배(30% 소프트 버퍼)**를 곱한 값을 정수로 넣으세요.
        3. season: 여름/휴양지면 'SS', 가을/겨울/쌀쌀한 날씨면 'FW', 사계절이나 중립적이면 'ALL'로 분류하세요.
        4. slots의 color: 질의에 특정 색상(파란색 -> 'blue', 검은색 -> 'black', 회색 -> 'gray' 등)이 명시되어 있다면 영어 소문자로 추출하세요. 언급이 없으면 null.
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