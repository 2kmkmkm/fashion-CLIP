"""크롤링 데이터의 color_raw를 고정 색상 목록(core/color_palette.py)으로 바꾸는 매핑표를 Jev(TypeSafe)로 만든다.

결과 config/color_map.json의 형식(키: color_raw를 소문자·앞뒤 공백 제거한 값, 값: 기본 색상)은 이전 GPT 버전과 같아서
index_to_qdrant.py는 그대로 쓴다.

- color_raw마다 Jev에 보기 고르기(choice) 질문을 한 번씩 보낸다. 같은 이름은 한 번만 묻는다.
- 응답은 config/color_map_jev_results.jsonl에 남겨, 다시 실행하면 새로 생긴 이름만 묻는다(--fresh면 전부 다시).
- 확신(confidence)이 --threshold보다 낮으면 color_map.json에는 etc로 넣고 config/color_map_review.csv에 모은다.
- scripts/color_map_overrides.json에 사람이 정한 값이 있으면 Jev 결과보다 우선한다(약어처럼 Jev가 추측할 수밖에 없는 것).
- API 키는 TYPESAFE_API_KEY 환경 변수(.env 가능)에서 읽기만 하고 출력하지 않는다.

실행(저장소 루트에서): python scripts/generate_color_map.py
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.color_palette import INSTRUCTIONS, PALETTE, normalize_key  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
OVERRIDES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "color_map_overrides.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Jev로 color_raw → 기본 색상 매핑표(config/color_map.json) 만들기")
    parser.add_argument("--data", default="./data", help="크롤링 상품 JSON 폴더")
    parser.add_argument("--config", default="./config", help="결과 파일을 둘 폴더")
    parser.add_argument("--threshold", type=float, default=0.6, help="이 값보다 확신이 낮으면 etc로 두고 review에 넣음")
    parser.add_argument("--workers", type=int, default=8, help="동시에 보낼 Jev 요청 수")
    parser.add_argument("--fresh", action="store_true", help="이전 응답을 무시하고 전부 다시 묻기(색상 목록·지시문을 바꿨을 때)")
    return parser.parse_args()


def collect_raw_colors(data_root):
    """color_raw별로 처음 나온 상품명·카테고리와 같은 상품의 다른 색상 이름을 함께 모은다."""
    contexts = {}
    for root, _, files in os.walk(data_root):
        for name in sorted(files):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8") as file:
                    product = json.load(file)
            except (OSError, json.JSONDecodeError) as error:
                print(f"[건너뜀] {name}: {error}")
                continue
            raws = [m.get("color_raw") for m in product.get("color_image_mappings") or [] if m.get("color_raw")]
            for raw in raws:
                key = normalize_key(raw)
                if not key or key in contexts:
                    continue
                contexts[key] = {
                    "raw": raw.strip(),
                    "product_name": product.get("name", ""),
                    "category": (product.get("dataset") or {}).get("logical_category", ""),
                    "siblings": [r for r in raws if r != raw][:10],
                }
    return contexts


def describe(context):
    lines = [
        f"색상 옵션 이름: {context['raw']}",
        f"상품명: {context['product_name']}",
        f"카테고리: {context['category']}",
    ]
    if context["siblings"]:
        lines.append("같은 상품의 다른 색상 옵션: " + ", ".join(context["siblings"]))
    return "\n".join(lines)


def ask_jev(api_key, state):
    body = {
        "model": MODEL,
        "state": state,
        "questions": {"color": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": PALETTE}},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            # 요청 한도(429)나 서버 오류는 잠시 기다렸다 다시 보낸다.
            if error.code in (429, 500, 502, 503, 504) and attempt < 4:
                retry_after = error.headers.get("Retry-After") if error.headers else None
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt)
                continue
            raise RuntimeError(f"Jev 요청 실패 HTTP {error.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Jev에 연결하지 못했습니다: {error}") from None
    raise RuntimeError("Jev 요청 재시도 횟수를 넘었습니다.")


def classify(api_key, key, context):
    response = ask_jev(api_key, describe(context))
    answer = (response.get("answers") or {}).get("color") or {}
    choice, confidence = answer.get("choice"), answer.get("confidence")
    if choice not in PALETTE or not isinstance(confidence, (int, float)):
        raise RuntimeError("Jev 응답 형식이 예상과 다릅니다: " + json.dumps(answer, ensure_ascii=False)[:300])
    top = sorted((answer.get("probabilities") or {}).items(), key=lambda item: -item[1])[:3]
    usage = response.get("usage") or {}
    return {
        "key": key,
        "raw": context["raw"],
        "choice": choice,
        "confidence": float(confidence),
        "top3": top,
        "product_name": context["product_name"],
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
    }


def load_overrides():
    if not os.path.exists(OVERRIDES_FILE):
        return {}
    with open(OVERRIDES_FILE, encoding="utf-8") as file:
        overrides = {normalize_key(k): v for k, v in json.load(file).items() if not k.startswith("_")}
    invalid = {k: v for k, v in overrides.items() if v not in PALETTE}
    if invalid:
        sys.exit(f"color_map_overrides.json에 색상 목록에 없는 값이 있습니다: {invalid}")
    return overrides


def main():
    args = parse_args()
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        sys.exit("TYPESAFE_API_KEY 환경 변수가 없습니다(.env에 넣어도 됩니다).")
    if not os.path.isdir(args.data):
        sys.exit(f"데이터 폴더가 없습니다: {args.data}")
    os.makedirs(args.config, exist_ok=True)
    results_path = os.path.join(args.config, "color_map_jev_results.jsonl")
    map_path = os.path.join(args.config, "color_map.json")
    review_path = os.path.join(args.config, "color_map_review.csv")

    contexts = collect_raw_colors(args.data)
    overrides = load_overrides()
    previous = {}
    if os.path.exists(results_path) and not args.fresh:
        with open(results_path, encoding="utf-8") as file:
            previous = {r["key"]: r for r in (json.loads(line) for line in file if line.strip())}
    todo = [k for k in sorted(contexts) if k not in previous and k not in overrides]
    overridden = sum(1 for k in contexts if k in overrides)
    reused = sum(1 for k in contexts if k in previous and k not in overrides)
    print(f">> 고유 color_raw {len(contexts)}개 · 사람이 정한 값 {overridden}개 · 이전 응답 재사용 {reused}개 · 새로 물어볼 것 {len(todo)}개")

    started = time.time()
    tokens = Counter()
    with open(results_path, "w" if args.fresh else "a", encoding="utf-8") as results_file, \
            ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(classify, api_key, key, contexts[key]) for key in todo]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            tokens.update(input=record["input_tokens"], output=record["output_tokens"])
            previous[record["key"]] = record
            results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_file.flush()
            if done % 50 == 0 or done == len(todo):
                print(f"   {done}/{len(todo)} 완료")

    color_map, review, counts = {}, [], Counter()
    for key in sorted(contexts):
        if key in overrides:
            color_map[key] = overrides[key]
        else:
            record = previous[key]
            confident = record["confidence"] >= args.threshold
            color_map[key] = record["choice"] if confident else "etc"
            if not confident or record["choice"] in ("etc", "multi"):
                review.append(record)
        counts[color_map[key]] += 1

    with open(map_path, "w", encoding="utf-8") as file:
        json.dump(color_map, file, ensure_ascii=False, indent=4)
    with open(review_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["color_raw", "Jev 선택", "확신", "상위 후보", "상품명"])
        for record in sorted(review, key=lambda r: r["confidence"]):
            writer.writerow([record["raw"], record["choice"], f"{record['confidence']:.2f}",
                             ", ".join(f"{label} {prob:.2f}" for label, prob in record["top3"]), record["product_name"]])

    print()
    if todo:
        print(f">> 새로 물어본 것 {len(todo)}개, {time.time() - started:.1f}초, "
              f"토큰 입력 {tokens['input']} / 출력 {tokens['output']}")
    print(">> 색상별 개수: " + ", ".join(f"{c} {n}" for c, n in counts.most_common()))
    print(f">> 사람이 정한 값 {overridden}개 적용 ({os.path.relpath(OVERRIDES_FILE)})")
    print(f">> 매핑표: {map_path}")
    print(f">> 확인 필요 {len(review)}개(확신 {args.threshold} 미만 또는 etc·multi): {review_path}")
    for record in sorted(review, key=lambda r: r["confidence"])[:15]:
        top = ", ".join(f"{label} {prob:.2f}" for label, prob in record["top3"])
        print(f"   {record['raw']:<28} → {record['choice']:<8} ({record['confidence']:.2f}) | {top}")
    print(">> 확인 필요 항목은 사람이 보고 scripts/color_map_overrides.json에 넣은 뒤 다시 실행하면 반영됩니다.")


if __name__ == "__main__":
    main()
