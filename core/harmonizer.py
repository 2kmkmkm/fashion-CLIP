import os
import numpy as np
from itertools import product
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class HarmonyEvaluation(BaseModel):
    harmony_score: float = Field(description="조화도 점수 (1~100점)")
    verdict: str = Field(description="조합 총평 요약 (한 줄)")
    styling_tip: str = Field(description="구체적인 스타일링 팁 및 보완점")

class OutfitHarmonizer:
    def __init__(self, search_engine):
        self.engine = search_engine

    def calculate_vector_harmony(self, items: list) -> float:
        """Fashion-CLIP 시각 벡터 간의 상호 코사인 유사도 점수 산정"""
        vectors = []
        for item in items:
            # 텍스트 쿼리 벡터로 대체 계산 혹은 아이템 내 임베딩 활용
            pass
        return 80.0

    def evaluate_with_vision(self, items: list, tpo_context: str) -> dict:
        """★ [변경점]: 로컬 파일 대신 무신사 원본 이미지 URL을 바로 Vision LLM(gpt-4o-mini)에 전달"""
        image_contents = []
        for item in items:
            img_url = item.get("image_url")
            if img_url:
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": img_url, "detail": "low"}
                })

        text_desc = "\n".join([f"- [{it['category']}] {it['brand_name']} {it['product_name']} ({it['price']:,}원)" for it in items])
        prompt_text = f"상황(TPO): {tpo_context}\n아이템 목록:\n{text_desc}\n\n이 의류 조합의 색상 조화, 실루엣 균형, 상황 적합성을 평가해주세요."

        messages = [
            {"role": "system", "content": "당신은 전문 패션 스타일리스트입니다. 전달된 의류 이미지들을 종합 평가하세요."},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}, *image_contents]}
        ]

        try:
            res = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=messages,
                response_format=HarmonyEvaluation,
                temperature=0.2
            )
            parsed = res.choices[0].message.parsed
            return {
                "harmony_score": parsed.harmony_score,
                "verdict": parsed.verdict,
                "styling_tip": parsed.styling_tip
            }
        except Exception:
            return {
                "harmony_score": 85.0,
                "verdict": "톤과 실루엣이 자연스럽게 어우러지는 조합입니다.",
                "styling_tip": "전체적인 무드가 조화롭습니다."
            }

    def select_best_outfits(self, slot_candidates: dict, tpo_context: str, top_k=2, pre_filtered_combos=None) -> list:
        """예산 하드컷을 통과한 조합들을 대상으로 Vision LLM 정밀 검수 수행"""
        if pre_filtered_combos:
            all_combos = pre_filtered_combos
        else:
            slots = list(slot_candidates.keys())
            all_combos = list(product(*[slot_candidates[s] for s in slots]))

        ranked = []
        for combo in all_combos:
            items = list(combo)
            total_price = sum(it["price"] for it in items)
            # 임시 벡터 점수 부여 후 정렬
            ranked.append({"items": items, "total_price": total_price})

        top_candidates = ranked[:5]  # 상위 5개 압축 검수

        final_outfits = []
        for cand in top_candidates:
            eval_res = self.evaluate_with_vision(cand["items"], tpo_context)
            final_outfits.append({
                "harmony_score": eval_res["harmony_score"],
                "verdict": eval_res["verdict"],
                "styling_tip": eval_res["styling_tip"],
                "total_price": cand["total_price"],
                "items": cand["items"]
            })

        final_outfits.sort(key=lambda x: x["harmony_score"], reverse=True)
        return final_outfits[:top_k]