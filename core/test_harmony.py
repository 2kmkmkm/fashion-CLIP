# -*- coding: utf-8 -*-
"""
무작위로 옷(상품)을 골라서 조화도 파이프라인(harmony_gate.evaluate_final)을
반복 테스트하는 스크립트.

- products.jsonl에서 무작위로 2~3개 아이템을 뽑는다.
- season/color는 harmony_gate가 기대하는 "사전 계산된 데이터" 형태의
  문자열로 만들어서 넘긴다 (season_status, color_status).
- TPO(착용 목적)도 무작위로 하나 골라서 넘긴다.
- evaluate_final()을 호출해서 GPT-4o-mini의 pass/violated_rules/feedback을 출력한다.

사용법:
    python test_harmony_random.py            # 기본 5회 반복
    python test_harmony_random.py --n 10      # 10회 반복
    python test_harmony_random.py --seed 42   # 재현 가능하게 시드 고정
"""
import argparse
import json
import os
import random

from harmony_gate import evaluate_final

PRODUCTS_PATH = "products.jsonl"
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images_cutout")

COLOR_HUE = {
    "red": 0, "orange": 30, "brown": 25, "beige": 40, "khaki": 60,
    "olive": 75, "yellow": 55, "mint": 150, "green": 120, "skyblue": 200,
    "sky": 200, "blue": 220, "navy": 230, "purple": 280, "violet": 275,
    "pink": 330, "burgundy": 350, "wine": 350, "ivory": 45,
}
NEUTRAL_COLORS = {"black", "white", "gray", "grey", "silver", "charcoal"}

TPO_CONTEXTS = [
    "정장 미팅",
    "친구와 캐주얼 약속",
    "면접",
    "여름 휴양지 여행",
    "출근룩",
    "데이트",
    "운동/헬스장",
    "결혼식 하객",
]


def load_products(path=PRODUCTS_PATH):
    products = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            products.append(json.loads(line))
    return products


def resolve_image_path(p: dict):
    path = p.get("local_image_path")
    if path and os.path.exists(path):
        return path
    product_id = p.get("product_id")
    if product_id:
        candidate = os.path.join(IMAGES_DIR, f"{product_id}.png")
        if os.path.exists(candidate):
            return candidate
    return None


def get_colors(p: dict):
    colors = p.get("color") or p.get("color_normalized") or p.get("color_from_name") or []
    if isinstance(colors, str):
        colors = [colors]
    return [c.lower() for c in colors if c]


def get_season(p: dict):
    return (p.get("season") or "").upper() or None


def _hue_diff(h1, h2):
    d = abs(h1 - h2) % 360
    return min(d, 360 - d)


def build_color_status(items: list) -> str:
    """모든 아이템 쌍 중 가장 부조화한 조합을 기준으로 color_status 문자열 생성."""
    worst = None
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            colors_a, colors_b = get_colors(items[i]), get_colors(items[j])
            if not colors_a or not colors_b:
                continue
            for ca in colors_a:
                for cb in colors_b:
                    if ca in NEUTRAL_COLORS or cb in NEUTRAL_COLORS:
                        continue
                    if ca not in COLOR_HUE or cb not in COLOR_HUE:
                        continue
                    diff = _hue_diff(COLOR_HUE[ca], COLOR_HUE[cb])
                    if worst is None or diff > worst[0]:
                        worst = (diff, f"{ca}-{cb}")

    if worst is None:
        return "위반 없음 (무채색 위주이거나 색상 정보 부족)"

    diff, pair = worst
    if diff <= 60:
        return f"위반 없음 ({pair}, Hue차 {diff:.0f}도로 유사색/인접색 범위)"
    if diff > 150:
        return f"위반 없음 ({pair}, Hue차 {diff:.0f}도로 보색 대비 범위 내 포인트 배색)"
    return f"위반: {pair}, Hue 차이 약 {diff:.0f}도로 유사색/삼색조화/보색 범위 밖"


def build_season_status(items: list) -> str:
    seasons = {get_season(p) for p in items if get_season(p)}
    if len(seasons) <= 1:
        s = seasons.pop() if seasons else "정보없음"
        return f"위반 없음 (모두 season={s}로 일치)"
    return f"위반: 계절 불일치 ({', '.join(sorted(seasons))})"


def pick_random_outfit(products: list, k: int):
    candidates = [p for p in products if resolve_image_path(p) is not None]
    if len(candidates) < k:
        raise RuntimeError(
            f"이미지가 있는 상품이 {len(candidates)}개뿐이라 {k}개 조합을 못 만듦. "
            f"images_cutout/ 폴더 확인 필요."
        )
    return random.sample(candidates, k)


def to_harmony_items(picked: list) -> list:
    items = []
    for p in picked:
        items.append({
            "category": p.get("category") or p.get("category_l1") or "unknown",
            "brand_name": p.get("brand") or p.get("brand_name") or "브랜드명확인필요",
            "product_name": p.get("name") or p.get("product_name") or p.get("product_id"),
            "colors": get_colors(p),
            "local_image_path": resolve_image_path(p),
        })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="반복 테스트 횟수")
    parser.add_argument("--k", type=int, default=2, help="한 조합에 넣을 아이템 개수")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드 (재현용)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    products = load_products()

    for i in range(args.n):
        picked = pick_random_outfit(products, args.k)
        items = to_harmony_items(picked)

        season_status = build_season_status(picked)
        color_status = build_color_status(picked)
        tpo_context = random.choice(TPO_CONTEXTS)

        print(f"\n===== 테스트 {i + 1}/{args.n} =====")
        for it in items:
            print(f"  - [{it['category']}] {it['brand_name']} {it['product_name']} ({it['colors']})")
        print(f"  TPO: {tpo_context}")
        print(f"  season_status(사전계산): {season_status}")
        print(f"  color_status(사전계산): {color_status}")

        result = evaluate_final(
            items,
            tpo_context=tpo_context,
            season_status=season_status,
            color_status=color_status,
        )

        print(f"  -> pass={result['pass_status']} | violated={result['violated_rules']}")
        print(f"     feedback: {result['feedback_summary']}")


if __name__ == "__main__":
    main()