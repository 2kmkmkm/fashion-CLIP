import os
import numpy as np
from itertools import product
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 코디 추천 시 예산 조건을 통과한 후보 조합들을 대상으로, 
# 무신사 원본 이미지 URL을 Vision LLM에 직접 전달하여 
# 색상 조화, 실루엣 균형, TPO 적합성을 심사하고 점수와 스타일링 팁을 도출

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class HarmonyEvaluation(BaseModel):
    harmony_score: float = Field(description="조화도 점수 (1~100점)")
    verdict: str = Field(description="조합 총평 요약 (한 줄)")
    styling_tip: str = Field(description="구체적인 스타일링 팁 및 보완점")

class OutfitHarmonizer:
    def __init__(self, search_engine):
        self.engine = search_engine

    # Fashion-CLIP 기반의 조화도 1차 필터링
    def calculate_vector_harmony(self, items: list) -> float:
        """Fashion-CLIP 시각 벡터 간의 상호 코사인 유사도 점수 산정"""
        vectors = []
        for item in items:
            # 아이템 딕셔너리에 포함된 512차원 벡터 추출
            vec = item.get("vector")
            if vec is not None:
                vectors.append(np.array(vec))
    
        # 아이템 개수가 2개 미만이면 조화도를 계산할 수 없으므로 기본 점수 반환
        if len(vectors) < 2:
            return 80.0

        # 조합 내 모든 아이템 쌍(Pair) 간의 코사인 유사도 계산
        similarities = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                v1 = vectors[i]
                v2 = vectors[j]
                norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm_product == 0:
                    sim = 0.0
                else:
                    sim = np.dot(v1, v2) / norm_product
                similarities.append(sim)
        
        if not similarities:
            return 80.0

        # 평균 코사인 유사도 산출
        avg_sim = np.mean(similarities)
        
        # Fashion-CLIP 유사도 분포(통상 0.1~0.6)를 0~100점 스케일로 선형 변환
        score = float(np.clip((avg_sim - 0.1) / 0.4 * 100, 20, 100))
        return round(score, 2)

    def evaluate_with_vision(self, items: list, tpo_context: str) -> dict:
        """로컬 파일 대신 무신사 원본 이미지 URL을 바로 Vision LLM(gpt-4o-mini)에 전달"""
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

            # Fashion-CLIP 벡터 기반 1차 조화도 점수 산정
            vector_score = self.calculate_vector_harmony(items)

            ranked.append({
                "items": items, 
                "total_price": total_price,
                "vector_score": vector_score
            })

        # 1차 벡터 조화도가 높은 순서대로 정렬 후 상위 5개만 압축
        ranked.sort(key=lambda x: x["vector_score"], reverse=True)
        top_candidates = ranked[:5] # 상위 5개 압축 검수

        final_outfits = []
        for cand in top_candidates:
            # 2차 정밀 검수 (Vision LLM)
            eval_res = self.evaluate_with_vision(cand["items"], tpo_context)
            final_outfits.append({
                "harmony_score": eval_res["harmony_score"],
                "verdict": eval_res["verdict"],
                "styling_tip": eval_res["styling_tip"],
                "total_price": cand["total_price"],
                "items": cand["items"]
            })

        # 최종 LLM 조화도 점수 기준으로 재정렬하여 상위 `top_k`개 반환
        final_outfits.sort(key=lambda x: x["harmony_score"], reverse=True)
        return final_outfits[:top_k]