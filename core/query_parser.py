import os
import datetime
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
    category: Literal["outer", "top", "bottom", "shoes", "hat"] = Field(description="무신사 5대 대분류 카테고리")
    clip_query_en: str = Field(description="Fashion-CLIP 검색용 패션 영문 시각 묘사")
    max_price: Optional[int] = Field(description="Soft-Margin 버퍼(1.3배)가 곱해진 해당 슬롯의 가격 상한선")
    color: Optional[str] = Field(description="질의에서 언급된 색상 (예: black, blue, white, gray, navy, red 등, 없으면 null)")
    fit_type: Optional[Literal["slim", "regular", "relaxed", "oversized"]] = Field(description="질의에서 언급된 핏 (slim, regular, relaxed, oversized 중 택 1. 없으면 null)")

# 전체 질의의 거시적인 맥락과 공통 조건을 담는 최상위 컨테이너
# 단품/코디, 전체 예산, 계절감, TPO
class ParsedIntent(BaseModel):
    search_type: Literal["single", "coordination"] = Field(description="단품 검색 vs 코디 세트 추천")
    total_budget: Optional[int] = Field(description="사용자가 언급한 총예산 (원 단위 정수, 없으면 null)")
    season_preferences: List[Literal["봄", "여름", "가을", "겨울"]] = Field(
        description="사용자 질의에서 파악된 계절감 우선순위 배열 (가장 핵심이 되는 계절을 첫 번째(index 0)로 배치)"
    )
    tpo_summary: str = Field(descri11ption="TPO 및 무드 요약")
    slots: List[SlotQuery]

class QueryParser:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    def parse(self, user_query: str) -> ParsedIntent:
        current_month = datetime.datetime.now().month
        
        system_prompt = f"""
        당신은 패션 커머스 플랫폼의 수석 AI 스타일리스트이자 데이터 파서입니다.
        사용자의 한국어 질의를 분석하여 [단품 검색] 또는 [코디 세트 추천]으로 분류하세요.

        현재 시점은 {current_month}월입니다.
        
        [핵심 파싱 규칙 및 예외 방어 지침]:
        
        1. search_type 및 slots 분기:
           - "A에 어울리는 B", "A에 신을 B 추천해줘"처럼 특정 기준 아이템(A)과 조합할 타겟 단품(B) 하나를 묻는 질의는 코디가 아니라 'single'로 분류하고, 사용자가 구매하고자 하는 핵심 타겟 아이템 1개만 슬롯으로 추출하세요.
           - 여러 아이템의 조화를 동시에 물어볼 때만 'coordination'으로 분류하세요.

        2. category 매정밀 지정:
           - 무신사 5대 대분류("outer", "top", "bottom", "shoes", "hat") 중 정확히 하나를 매칭하세요.
           - [예외 방어]: "패딩", "코트", "자켓", "점퍼", "가디건" 등은 상의가 아니라 무조건 **"outer"**로 분류해야 합니다. "바지", "슬랙스", "데님", "스커트"는 **"bottom"**입니다.

        3. clip_query_en (Fashion-CLIP 최적화 영문 묘사):
           - 단순 단어 직역이 아니라, 패션 비전 모델이 이해하기 좋은 구체적인 시각적 스타일 묘사로 확장하세요. (예: "검은색 패딩" -> "heavy winter black puffer down jacket, warm outerwear")

        4. season_preferences (계절 우선순위 배열):
           - 사용자의 질의에서 느껴지는 계절감을 우선순위가 높은 순서대로 '봄', '여름', '가을', '겨울' 중 선택하여 배열로 담으세요.
           - [강력 예외 방어]: 질의에 "패딩", "코트", "다운", "기모", "울 코트" 등 명백한 겨울 아우터/방한용품 키워드가 포함되어 있다면, **현재 월과 상관없이 무조건 배열의 첫 번째(index 0) 값은 '겨울'**이어야 합니다.
           - "봄가을용", "간절기" 같은 복합 니즈라면 핵심 계절을 앞에 두고 순서대로 배열하세요 (예: ['가을', '봄']).
           - 계절감이 전혀 없고 범용적인 아이템(예: "무지 티셔츠", "스니커즈", "데님 팬츠")이라면 4가지 계절 전체를 유연하게 모두 담아내세요 (예: ['봄', '여름', '가을', '겨울']).

        5. total_budget 및 max_price:
           - total_budget: 사용자가 언급한 총예산(예: "20만원대")을 정수(200000)로 추출하세요. 없으면 null.
           - slots의 max_price: 슬롯별 예산을 나눌 때, Qdrant 검색 시 여유를 주기 위해 산정된 기준 예산의 1.3배(30% 소프트 버퍼)를 곱한 값을 정수로 넣으세요. 예산이 없으면 null.

        6. color (색상 정규화):
           - 질의에 특정 색상(검은색->'black', 파란색->'navy' 또는 'blue', 흰색->'white', 회색->'gray', 아이보리->'ivory' 등)이 명시되어 있다면 표준 영문 소문자로 추출하세요. 언급이 없으면 null.

        7. fit_type (핏 성향 제한):
           - 사용자가 핏을 언급하면 반드시 'slim', 'regular', 'relaxed', 'oversized' 4개 중 하나로만 추출하세요. 오버핏/루즈핏은 'relaxed' 또는 'oversized'로 매핑하고, 언급이 없으면 null.
        """

        # Few-shot 예시
        messages = [
            {"role": "system", "content": system_prompt},

            # 예시 1: 단품 검색 (계절 무관)
            {"role": "user", "content": "연청색 바지에 신을 실버 스니커즈 추천해줘"},
            {"role": "assistant", "content": '{"search_type": "single", "total_budget": null, "season_preferences": ["봄", "여름", "가을", "겨울"], "tpo_summary": "연청색 바지에 어울리는 포인트 실버 스니커즈 스타일링", "slots": [{"category": "shoes", "clip_query_en": "silver metallic sneakers", "max_price": null, "color": "silver", "fit_type": null}]}'},
            
            # 예시 2: 코디 추천 (계절 무관)
            {"role": "user", "content": "하객룩으로 입을 깔끔한 블레이저랑 슬랙스 세트 찾아줘"},
            {"role": "assistant", "content": '{"search_type": "coordination", "total_budget": null, "season_preferences": ["봄", "여름", "가을", "겨울"], "tpo_summary": "결혼식 하객룩, 단정하고 깔끔한 포멀 무드", "slots": [{"category": "outer", "clip_query_en": "clean formal blazer jacket", "max_price": null, "color": null, "fit_type": null}, {"category": "bottom", "clip_query_en": "formal dress pants slacks", "max_price": null, "color": null, "fit_type": null}]}'},
            
            # 예시 3: 핏이 명시된 단품 검색
            {"role": "user", "content": "스트릿하게 입기 좋은 오버사이즈 후드티 찾아줘"},
            {"role": "assistant", "content": '{"search_type": "single", "total_budget": null, "season_preferences": ["봄", "여름", "가을", "겨울"], "tpo_summary": "스트릿한 무드의 오버사이즈 후드티 스타일링", "slots": [{"category": "top", "clip_query_en": "street style oversized hoodie", "max_price": null, "color": null, "fit_type": "relaxed"}]}'},

            # 예시 4: 예산과 가을 계절감이 포함된 코디 추천
            {"role": "user", "content": "15만원 이하로 가을에 입기 좋은 니트랑 데님 팬츠 코디 추천해줘"},
            {"role": "assistant", "content": '{"search_type": "coordination", "total_budget": 150000, "season_preferences": ["가을", "겨울"], "tpo_summary": "가을철 캐주얼한 니트와 데님 팬츠 데일리 코디", "slots": [{"category": "top", "clip_query_en": "cozy knitwear sweater", "max_price": 97500, "color": null, "fit_type": null}, {"category": "bottom", "clip_query_en": "denim jeans pants", "max_price": 97500, "color": null, "fit_type": null}]}'},

            # 예시 5: 봄가을 간절기 복합 니즈 질의
            {"role": "user", "content": "봄가을용으로 입기 좋은 트렌치코트 찾아줘"},
            {"role": "assistant", "content": '{"search_type": "single", "total_budget": null, "season_preferences": ["가을", "봄"], "tpo_summary": "간절기 시즌에 활용하기 좋은 클래식 트렌치코트", "slots": [{"category": "outer", "clip_query_en": "classic trench coat jacket", "max_price": null, "color": null, "fit_type": null}]}'},

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