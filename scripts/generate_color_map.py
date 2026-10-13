import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_ROOT = "./data"

def extract_unique_colors():
    print(">> 데이터셋에서 고유 색상 추출 중...")
    unique_colors = set()
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # color_image_mappings 안의 color_raw 추출
                        mappings = data.get("color_image_mappings", [])
                        for mapping in mappings:
                            raw_color = mapping.get("color_raw")
                            if raw_color:
                                unique_colors.add(raw_color.strip().lower())
                except Exception:
                    pass
    return list(unique_colors)

def generate_color_mapping(raw_colors):
    print(f">> 총 {len(raw_colors)}개의 고유 색상 발견. GPT-4o-mini로 매핑 시작...")
    
    prompt = f"""
    당신은 패션 데이터 엔지니어입니다.
    아래 제공된 쇼핑몰 색상명 리스트(한국어/영어/혼종)를 분석하여, 
    가장 가까운 기본 영문 소문자 색상(black, white, gray, navy, blue, red, green, brown, beige, pink, purple, ivory, khaki 등) 15~20개 중 하나로 매핑해주세요.
    결과는 반드시 JSON 형태의 딕셔너리로만 반환해야 합니다. (Key: 원래 색상, Value: 매핑된 기본 영문 색상)
    
    [색상명 리스트]
    {raw_colors}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={ "type": "json_object" }
    )
    
    return json.loads(res.choices[0].message.content)

if __name__ == "__main__":
    raw_color_list = extract_unique_colors()
    if raw_color_list:
        color_map_json = generate_color_mapping(raw_color_list)
        
        # 파일로 저장
        with open("./config/color_map.json", "w", encoding="utf-8") as f:
            json.dump(color_map_json, f, ensure_ascii=False, indent=4)
        print(">> 🎉 'color_map.json' 파일 생성 완료!")
    else:
        print(">> 추출된 색상이 없습니다.")