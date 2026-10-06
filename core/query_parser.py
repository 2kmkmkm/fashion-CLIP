import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from dotenv import load_dotenv

# OpenAI를 활용해 사용자의 한국어 질의를 파싱

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 코디 추천 시 개별 아이템(슬롯)이 가져야 할 구체적인 세부 검색 조건
# 카테고리, TPO, 가격, 색상
class SlotQuery(BaseModel):
    slot_name: str = Field(description="슬롯명: outer, top, bottom, shoes, hat 중 하나")
    category: Literal["outer", "top", "bottom", "shoes", "hat"] = Field(description="무신사 5대 대분류 카테고리")
    clip_query_en: str = Field(description="Fashion-CLIP 검색용 패션 영문 시각 묘사")
    max_price: Optional[int] = Field(description="Soft-Margin 버퍼(1.3배)가 곱해진 해당 슬롯의 가격 상한선")
    color: Optional[str] = Field(description="질의에서 언급된 색상 (예: black, blue, white, gray, navy, red 등, 없으면 null)")
    fit_type: Optional[str] = Field(description="질의에서 언급된 핏 성향 (예: slim, regular, relaxed, oversized 등, 없으면 null)")

# 전체 질의의 거시적인 맥락과 공통 조건을 담는 최상위 컨테이너
# 단품/코디, 전체 예산, 계절감, TPO
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
        2. slots의 max_price: 슬롯별 예산을 나눌 때, Qdrant 검색 시 여유를 주기 위해 산정된 기준 예산의 1.3배(30% 소프트 버퍼)를 곱한 값을 정수로 넣으세요.
        3. season: 여름/휴양지면 'SS', 가을/겨울/쌀쌀한 날씨면 'FW', 사계절이나 중립적이면 'ALL'로 분류하세요.
        4. slots의 color: 질의에 특정 색상(파란색 -> 'blue', 검은색 -> 'black', 회색 -> 'gray' 등)이 명시되어 있다면 영어 소문자로 추출하세요. 언급이 없으면 null.
        5. "A에 어울리는 B", "A에 신을 B 추천해줘"처럼 특정 아이템(A)과 조합할 타겟 단품(B) 하나를 묻는 질의는 코디가 아니라 'single'로 분류하고, 사용자가 구매하고자 하는 핵심 타겟 아이템 1개만 슬롯으로 추출하세요.
        6. slots의 fit_type: 질의에 특정 핏(오버핏/루즈핏 -> 'relaxed' 또는 'oversized', 슬림핏 -> 'slim', 기본/레귤러 -> 'regular' 등)이 명시되어 있다면 영어 소문자로 추출하세요. 언급이 없으면 null.
        """

        # Few-shot 예시
        messages = [
            {"role": "system", "content": system_prompt},

            # 예시 1: 단품 검색 (기본)
            {"role": "user", "content": "연청색 바지에 신을 실버 스니커즈 추천해줘"},
            {"role": "assistant", "content": '{"search_type": "single", "total_budget": null, "season": "ALL", "tpo_summary": "연청색 바지에 어울리는 포인트 실버 스니커즈 스타일링", "slots": [{"slot_name": "shoes", "category": "shoes", "clip_query_en": "silver metallic sneakers, modern trendy shoes", "max_price": null, "color": "silver", "fit_type": null}]}'},
            
            # 예시 2: 코디 추천 (기본)
            {"role": "user", "content": "하객룩으로 입을 깔끔한 블레이저랑 슬랙스 세트 찾아줘"},
            {"role": "assistant", "content": '{"search_type": "coordination", "total_budget": null, "season": "ALL", "tpo_summary": "결혼식 하객룩, 단정하고 깔끔한 포멀 무드", "slots": [{"slot_name": "outer", "category": "outer", "clip_query_en": "clean formal blazer jacket", "max_price": null, "color": null, "fit_type": null}, {"slot_name": "bottom", "category": "bottom", "clip_query_en": "formal dress pants slacks", "max_price": null, "color": null, "fit_type": null}]}'},
            
            # 예시 3: 핏이 명시된 단품 검색 (오버사이즈/루즈핏)
            {"role": "user", "content": "스트릿하게 입기 좋은 오버사이즈 후드티 찾아줘"},
            {"role": "assistant", "content": '{"search_type": "single", "total_budget": null, "season": "ALL", "tpo_summary": "스트릿한 무드의 오버사이즈 후드티 스타일링", "slots": [{"slot_name": "top", "category": "top", "clip_query_en": "street style oversized hoodie", "max_price": null, "color": null, "fit_type": "relaxed"}]}'},

            # 예시 4: 총예산과 계절감이 포함된 코디 추천 (예산 안분 및 FW 시즌)
            {"role": "user", "content": "15만원 이하로 가을에 입기 좋은 니트랑 데님 팬츠 코디 추천해줘"},
            {"role": "assistant", "content": '{"search_type": "coordination", "total_budget": 150000, "season": "FW", "tpo_summary": "가을철 캐주얼한 니트와 데님 팬츠 데일리 코디", "slots": [{"slot_name": "top", "category": "top", "clip_query_en": "cozy knitwear sweater", "max_price": 97500, "color": null, "fit_type": null}, {"slot_name": "bottom", "category": "bottom", "clip_query_en": "denim jeans pants", "max_price": 97500, "color": null, "fit_type": null}]}'},

            # 예시 5: 색상과 핏이 동시에 지정된 단품 검색 (SS 시즌 + 네이비 + 루즈핏)
            {"role": "user", "content": "시원하게 입기 좋은 루즈핏 네이비 반팔 티셔츠 찾아줘"},
            {"role": "assistant", "content": '{"search_type": "single", "total_budget": null, "season": "SS", "tpo_summary": "여름철 시원하고 편안한 루즈핏 네이비 반팔 티셔츠", "slots": [{"slot_name": "top", "category": "top", "clip_query_en": "casual short sleeve t-shirt", "max_price": null, "color": "navy", "fit_type": "relaxed"}]}'},

            # 실제 사용자 질의
            {"role": "user", "content": user_query}
        ]

        completion = client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=ParsedIntent,
            temperature=0.1
        )
        
        return completion.choices[0].message.parsed