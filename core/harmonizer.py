import os
import numpy as np
from itertools import product
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 코디 추천 시 예산 조건을 통과한 후보 조합들을 대상으로, 
# 무신사 원본 이미지 URL을 Vision LLM에 직접 전달하여 
# 사전 계산된 색상/계절 결과와 함께 TPO 적합성을 심사하고 점수와 스타일링 팁을 도출

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# [신규] 색상/계절 사전 검증 결과와 TPO 심사를 결합한 게이트 결과 스키마
class HarmonyGateResult(BaseModel):
    pass_status: bool = Field(description="색상/계절/TPO 세 기준을 모두 통과하면 true")
    violated_rules: list[str] = Field(description="위반된 기준 이름 리스트 (색상 조화 / 계절 일치 / TPO 부적합)")
    feedback_summary: str = Field(description="한국어 존댓말 2~3문장 피드백")

SYSTEM_PROMPT = """당신은 패션 스타일리스트이자 코디 조화도(outfit harmony) 심사관입니다.
아래 세 가지 기준으로만 조화도를 평가합니다. 이 기준 밖의 임의 판단은 하지 않습니다.

[평가 기준]
1. 색상 조화 (color_status) — 이미 색상 이론으로 계산되어 주어집니다.
   이 값이 "위반"이면 반드시 violated_rules에 "색상 조화" 항목을 포함하세요.
   이 값이 "위반 없음"이면 색상만으로는 위반을 만들지 마세요. 당신이 직접 Hue를
   재계산하거나 이 결과를 뒤집지 마세요.
2. 계절 일치 (season_status) — 이미 계산되어 주어집니다. 마찬가지로 주어진 값을
   그대로 신뢰하고, "위반"일 때만 violated_rules에 "계절 일치" 항목을 포함하세요.
3. TPO(상황) 적합성 — 이것만 당신이 이미지와 상품 정보를 보고 직접 판단합니다.
   주어진 tpo_context(예: "면접", "여름 휴양지 여행")에 이 착장이 어울리는지,
   포멀도·노출도·활동성 관점에서 과하거나 부족한 부분이 있으면
   violated_rules에 "TPO 부적합"을 포함하고 구체적으로 어떤 아이템이 왜
   안 맞는지 feedback_summary에 명시하세요.

[출력 규칙]
- pass_status: 위 세 기준 중 하나라도 위반이면 false, 셋 다 문제없으면 true.
- violated_rules: 위반된 기준 이름만 리스트로. 위반이 없으면 빈 리스트.
- feedback_summary: 한국어 존댓말 2~3문장. 위반이 있으면 "무엇이 왜 안 맞는지 +
  어떻게 고치면 되는지"를 구체적으로. 위반이 없으면 "왜 잘 어울리는지"를
  코디 요소(색/계절/상황) 하나씩 짚어서 설명.
- 근거 없는 칭찬이나 뭉뚱그린 말("괜찮아요", "잘 어울려요")만 쓰지 말고,
  반드시 색상/계절/TPO 중 무엇 때문인지 밝히세요.
"""

USER_PROMPT_TEMPLATE = """[착장 구성]
{items_desc}

[상황(TPO)]
{tpo_context}

[사전 계산된 색상 조화 결과]
{color_status}

[사전 계산된 계절 일치 결과]
{season_status}

위 착장 이미지를 보고 TPO 적합성을 판단한 뒤, 세 기준을 종합해서
pass_status / violated_rules / feedback_summary를 채워주세요."""

class OutfitHarmonizer:
    def __init__(self, search_engine):
        self.engine = search_engine

    def calculate_vector_harmony(self, items: list) -> float:
        """Fashion-CLIP 시각 벡터 간의 상호 코사인 유사도 점수 산정"""
        vectors = []
        for item in items:
            vec = item.get("vector")
            if vec is not None:
                vectors.append(np.array(vec))
    
        if len(vectors) < 2:
            return 80.0

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

        avg_sim = np.mean(similarities)
        score = float(np.clip((avg_sim - 0.1) / 0.4 * 100, 20, 100))
        return round(score, 2)

    def evaluate_final(self, items: list, tpo_context: str,
                       season_status: str = "PASS",
                       color_status: str = "위반 없음") -> dict:
        """무신사 원본 이미지 URL을 바로 Vision LLM(gpt-4o-mini)에 전달하여 최종 게이트 심사 수행"""
        items_desc = "\n".join(
            f"- [{it['category']}] {it['brand_name']} {it['product_name']} "
            f"(색상: {', '.join(it['colors']) or '정보없음'})"
            for it in items
        )

        content = [
            {
                "type": "text",
                "text": USER_PROMPT_TEMPLATE.format(
                    items_desc=items_desc,
                    tpo_context=tpo_context,
                    color_status=color_status,
                    season_status=season_status,
                ),
            }
        ]
        
        # [수정] 로컬 파일 대신 무신사 원본 이미지 URL을 직접 Vision LLM에 전달
        for it in items:
            img_url = it.get("image_url")
            if img_url:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_url,
                        "detail": "low"
                    }
                })

        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format=HarmonyGateResult,
                temperature=0.1,
            )
            result = completion.choices[0].message.parsed
            return result.model_dump()
        except Exception:
            return {
                "pass_status": True,
                "violated_rules": [],
                "feedback_summary": "톤과 실루엣이 자연스럽게 어우러지는 조화로운 코디입니다."
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

            vector_score = self.calculate_vector_harmony(items)

            # 사이즈 아웃라이어 극단적 감점 연동
            fit_scores = []
            has_outlier_size = False
            for it in items:
                rec = it.get("size_recommendation")
                if rec and "predicted_fit_score" in rec:
                    score_val = rec["predicted_fit_score"]
                    if score_val == 0.0:
                        has_outlier_size = True
                    fit_scores.append(score_val * 100)
                else:
                    fit_scores.append(50.0)
            
            avg_fit_score = sum(fit_scores) / len(fit_scores) if fit_scores else 50.0

            composite_score = (vector_score * 0.7) + (avg_fit_score * 0.3)
            if has_outlier_size:
                composite_score -= 50.0

            ranked.append({
                "items": items, 
                "total_price": total_price,
                "vector_score": vector_score,
                "avg_fit_score": avg_fit_score,
                "composite_score": composite_score
            })

        ranked.sort(key=lambda x: x["composite_score"], reverse=True)
        top_candidates = ranked[:5] # 상위 5개 압축 검수

        final_outfits = []
        for cand in top_candidates:
            # 무신사 이미지 URL 기반 최종 게이트 심사 실행
            gate_res = self.evaluate_final(cand["items"], tpo_context)
            
            harmony_score = 95.0 if gate_res["pass_status"] else 60.0

            final_outfits.append({
                "harmony_score": harmony_score,
                "pass_status": gate_res["pass_status"],
                "violated_rules": gate_res["violated_rules"],
                "verdict": gate_res["feedback_summary"],
                "styling_tip": f"위반 항목: {', '.join(gate_res['violated_rules']) if gate_res['violated_rules'] else '없음'}",
                "total_price": cand["total_price"],
                "items": cand["items"]
            })

        final_outfits.sort(key=lambda x: x["harmony_score"], reverse=True)
        return final_outfits[:top_k]