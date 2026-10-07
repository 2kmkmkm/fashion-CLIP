# -*- coding: utf-8 -*-
"""
색상/계절은 사전 계산된 값을 그대로 신뢰하고, TPO 적합성만 Vision LLM이
직접 이미지를 보고 판단하게 하는 조화도 최종 게이트.
"""
import base64
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class HarmonyGateResult(BaseModel):
    pass_status: bool = Field(description="색상/계절/TPO 세 기준을 모두 통과하면 true")
    violated_rules: list[str] = Field(description="위반된 기준 이름 리스트 (색상 조화 / 계절 일치 / TPO 부적합)")
    feedback_summary: str = Field(description="한국어 존댓말 2~3문장 피드백")


SYSTEM_PROMPT = """당신은 패션 스타일리스트이자 코디 조화도(outfit harmony) 심사관입니다.
아래 세 가지 기준으로만 조화도를 평가합니다. 이 기준 밖의 임의 판단은 하지 않습니다.

[평가 기준]
1. 색상 조화 (color_status) — 이미 색상 이론(Hue 차이)으로 계산되어 주어집니다.
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


def encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def evaluate_final(items: list, tpo_context: str,
                    season_status: str = "PASS",
                    color_status: str = "위반 없음") -> dict:
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
    for it in items:
        img_path = it.get("local_image_path")
        if img_path and os.path.exists(img_path):
            b64 = encode_image_b64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "low",
                },
            })

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
