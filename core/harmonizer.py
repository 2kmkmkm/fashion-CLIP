import os
import base64
import numpy as np
from itertools import product
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 3단계: 조화도 채점 & Vision LLM 검증기

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class HarmonyEvaluation(BaseModel):
    harmony_score: float = Field(description="조화도 점수 (1~100점)")
    verdict: str = Field(description="조합 총평 요약 (한 줄)")
    styling_tip: str = Field(description="구체적인 스타일링 팁 및 보완점")

class OutfitHarmonizer:
    def __init__(self, search_engine):
        self.engine = search_engine

    def _encode_image_b64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def calculate_vector_harmony(self, items: list) -> float:
        """Fashion-CLIP 시각 벡터 간의 상호 코사인 유사도 + 색상 규칙 채점"""
        vectors = []
        for item in items:
            img_path = item.get("local_image_path")
            if img_path and os.path.exists(img_path):
                vectors.append(np.array(self.engine.encode_image(img_path)))

        if len(vectors) < 2:
            return 70.0

        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(np.dot(vectors[i], vectors[j]))

        avg_sim = float(np.mean(sims))
        visual_score = np.clip((avg_sim - 0.15) / (0.60 - 0.15) * 100, 40, 95)

        # 색상 안정성 가산점
        color_bonus = 0
        all_colors = [c for item in items for c in (item.get("colors") or [])]
        if any(c in {"black", "white", "gray", "charcoal"} for c in all_colors):
            color_bonus += 5
        if 1 <= len(set(all_colors)) <= 3:
            color_bonus += 5

        return round(float(np.clip(visual_score + color_bonus, 50.0, 99.0)), 1)

    def evaluate_with_vision(self, items: list, tpo_context: str) -> dict:
        """최종 상위 조합의 누끼 이미지를 Vision LLM(gpt-4o-mini)에 전달해 정밀 검증"""
        image_contents = []
        for item in items:
            path = item.get("local_image_path")
            if path and os.path.exists(path):
                b64 = self._encode_image_b64(path)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}
                })

        text_desc = "\n".join([f"- [{it['category']}] {it['brand_name']} {it['product_name']} ({it['colors']})" for it in items])
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
            # Vision 호출 실패 시 벡터 점수로 fallback
            base_score = self.calculate_vector_harmony(items)
            return {
                "harmony_score": base_score,
                "verdict": "톤과 실루엣이 자연스럽게 어우러지는 조합",
                "styling_tip": "무채색 신발이나 가방을 함께 매치하면 깔끔합니다."
            }

    def select_best_outfits(self, slot_candidates: dict, tpo_context: str, top_k=2) -> list:
        """슬롯별 후보군을 조합하여 최적의 코디 셋 선별"""
        slots = list(slot_candidates.keys())
        all_combos = list(product(*[slot_candidates[s] for s in slots]))

        # 1차: 벡터 기반 고속 채점
        ranked = []
        for combo in all_combos:
            items = list(combo)
            vec_score = self.calculate_vector_harmony(items)
            total_price = sum(it["price"] for it in items)
            ranked.append({"items": items, "vec_score": vec_score, "total_price": total_price})

        ranked.sort(key=lambda x: x["vec_score"], reverse=True)
        top_candidates = ranked[:top_k]

        # 2차: 상위 조합 Vision LLM 정밀 검증
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
        return final_outfits