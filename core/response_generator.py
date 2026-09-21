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
        prompt = f"""
        사용자 질의: "{query}"
        상황(TPO): {tpo}
        추천된 단품 목록:
        {items}

        위 상품들을 바탕으로 사용자의 요청에 딱 맞춘 추천 코멘트를 정중하고 센스 있는 패션 매니저 말투(한국어)로 2~3문장 작성하세요.
        """
        res = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content

    def generate_coordination_response(self, query: str, tpo: str, outfits: list) -> str:
        prompt = f"""
        사용자 질의: "{query}"
        상황(TPO): {tpo}
        추천 코디 세트 정보:
        {outfits}

        위 세트 조합의 스타일 포인트와 TPO 적합성을 설명하는 추천 코멘트를 패션 매거진 에디터 어조로 매끄럽게 작성하세요.
        """
        res = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content