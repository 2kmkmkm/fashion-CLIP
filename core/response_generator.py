import os
from openai import OpenAI
from dotenv import load_dotenv

# 4단계: 자연어 응답 생성기

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ResponseGenerator:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    def generate_single_response(self, query: str, tpo: str, items: list) -> str:
        # 1. 불필요한 URL/ID 데이터 제거 및 텍스트 요약
        item_summaries = []
        for item in items:
            brand = item.get('brand_name', '브랜드')
            name = item.get('product_name', '상품명')
            price = item.get('price', '가격미상')
            item_summaries.append(f"- {brand} {name} ({price}원)")
        
        summary_text = "\n".join(item_summaries)

        prompt = f"""
        사용자 질의: "{query}"
        상황(TPO): {tpo}
        추천된 단품 목록:
        {summary_text}

        [엄격한 제약 사항]:
        1. 반드시 위 '추천된 단품 목록'에 실제로 포함된 상품명과 브랜드만 언급하세요. 목록에 없는 상품(예: 바람막이, 후드티 등)을 임의로 지어내거나 추천하지 마세요.
        2. 위 상품들을 바탕으로 사용자의 요청에 딱 맞춘 추천 코멘트를 정중하고 센스 있는 패션 매니저 말투(한국어)로 2~3문장 작성하세요.
        """
        res = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content

    def generate_coordination_response(self, query: str, tpo: str, outfits: list) -> str:
        # 1. 2단계 파이프라인을 거친 코디 세트 정보 텍스트 요약
        outfit_summaries = []
        for idx, outfit in enumerate(outfits):
            outfit_text = f"[[ 코디 세트 {idx+1} ]]\n"
            
            # items가 리스트이므로 바로 순회하며 속성 추출
            for item in outfit.get("items", []):
                brand = item.get('brand_name', '브랜드')
                name = item.get('product_name', '상품명')
                category = item.get('category', '의류')
                outfit_text += f"- [{category}] {brand} {name}\n"
            
            # 2. Vision LLM이 작성한 평가 내용(verdict 또는 styling_tip) 반영
            verdict = outfit.get("verdict", "톤과 실루엣이 자연스럽게 어우러지는 조합입니다.")
            tip = outfit.get("styling_tip", "")
            outfit_text += f"* AI 에디터 평가: {verdict} / {tip}\n"
            
            outfit_summaries.append(outfit_text)
        
        summary_text = "\n\n".join(outfit_summaries)

        prompt = f"""
        사용자 질의: "{query}"
        상황(TPO): {tpo}
        추천 코디 세트 정보:
        {summary_text}

        [엄격한 제약 사항]:
        1. 반드시 위 '추천 코디 세트 정보'에 명시된 실제 상품들(브랜드명, 상품명, 카테고리)만 정확히 인용하세요. 목록에 없는 신발, 가방, 액세서리 등의 아이템을 임의로 지어내어 추천하지 마세요.
        2. AI 에디터 평가 포인트에 기반하여 각 코디 세트의 매력을 설명하되, 존재하지 않는 스펙이나 가격을 만들어내지 마세요.
        
        당신은 최고의 패션 매거진 에디터입니다. 위 제약 사항을 준수하며 각 코디 세트의 매력과 TPO 적합성을 설명하는 추천 코멘트를 매끄럽고 세련되게 작성하세요.
        """
        res = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content