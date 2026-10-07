import os
import json
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models

class FashionCLIPRunner:
    def __init__(self, model_name="patrickjohncyh/fashion-clip"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f">> Fashion-CLIP 모델 로딩 중... (Device: {self.device})")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_images(self, pil_images, batch_size=16):
        embeddings = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                elif hasattr(feats, "last_hidden_state"):
                    feats = feats.last_hidden_state[:, 0, :]
                
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                embeddings.extend(feats.cpu().numpy())
        return np.array(embeddings)

# 상품 실측에 따른 정규화 함수
def normalize_measurements(category_name: str, raw_measurements: dict) -> dict:
    mapped = {}
    if category_name in ["상의", "아우터"]:
        key_map = {
            "총장": "top_length", 
            "어깨너비": "shoulder", 
            "가슴단면": "chest", 
            "소매길이": "sleeve", 
            "밑단단면": "top_hem", 
            "소매부리단면": "cuff", 
            "암홀": "armhole"
        }
    elif category_name == "하의":
        key_map = {
            "총장": "bottom_length", 
            "허리단면": "waist", 
            "엉덩이단면": "hip", 
            "허벅지단면": "thigh", 
            "밑위": "rise", 
            "밑단단면": "bottom_hem"
        }
    elif category_name == "신발":
        key_map = {
            "발길이": "foot_length", 
            "발볼": "foot_width", 
            "발목높이": "ankle_height", 
            "굽높이": "heel_height"
        }
    elif category_name == "모자":
        key_map = {
            "머리둘레": "head_circumference", 
            "챙길이": "brim_length", 
            "깊이": "depth"
        }
    else:
        key_map = {}
        
    for k, v in raw_measurements.items():
        en_key = key_map.get(k)
        if en_key is not None:
            try:
                val = float(v)
                if val > 0.0:
                    mapped[en_key] = val
            except (ValueError, TypeError):
                pass
    return mapped

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "musinsa_products"

print(f">> Qdrant 서버 연결 시도: {QDRANT_URL}")
qdrant = QdrantClient(url=QDRANT_URL)

if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(COLLECTION_NAME)
    print(f">> 기존 컬렉션 '{COLLECTION_NAME}' 삭제 및 초기화 완료.")

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
)
print(f">> 신규 컬렉션 '{COLLECTION_NAME}' 생성 완료")

# data 폴더의 하위 디렉토리 순회하며 JSON 로드
DATA_ROOT = "data"
if not os.path.exists(DATA_ROOT):
    raise FileNotFoundError(f"🚨 'data' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")

products = []
print(f">> '{DATA_ROOT}' 폴더 내 JSON 파일 탐색 시작...")
for root, dirs, files in os.walk(DATA_ROOT):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    products.append(data)
            except Exception as e:
                print(f"[에러] 파일 읽기 실패 ({file_path}): {e}")

print(f">> 총 {len(products)}개의 상품 JSON 파일 로드 완료.")

valid_records = []
pil_images_to_embed = []
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'}

for p in tqdm(products, desc="상품 이미지 안전 로딩 중"):
    p_id = str(p.get("product_id"))
    image_url = ""
    
    # 대표 이미지 URL 추출
    images_list = p.get("images", [])
    if images_list:
        image_url = images_list[0] if isinstance(images_list[0], str) else images_list[0].get("image_url", "")

    # 이미지 다운로드 및 파싱 시도
    img = None
    if image_url:
        try:
            response = requests.get(image_url, headers=headers, timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            pass

    # 유효성 검증 및 리스트 적재
    if img is not None:
        valid_records.append((p, image_url))
        pil_images_to_embed.append(img)
    else:
        print(f"\n[스킵] 상품 ID {p_id}: 이미지를 로드할 수 없습니다.")

print(f">> 최종 유효 이미지 로드 성공: {len(pil_images_to_embed)} / {len(products)} 건")

if not pil_images_to_embed:
    print(">> 🚨 임베딩할 이미지가 0건입니다. 이미지 URL을 확인해주세요.")
    exit()

fclip = FashionCLIPRunner()
BATCH_SIZE = 16
all_embeddings = []
for i in tqdm(range(0, len(pil_images_to_embed), BATCH_SIZE), desc="Fashion-CLIP 임베딩 추출"):
    batch_imgs = pil_images_to_embed[i:i + BATCH_SIZE]
    batch_vecs = fclip.encode_images(batch_imgs, batch_size=len(batch_imgs))
    all_embeddings.extend(batch_vecs)

points = []
for idx, (p, img_url) in enumerate(valid_records):
    p_id = p.get("product_id")
    
    # 가격
    price_info = p.get("price", {})
    price_val = int(price_info.get("normal", 0))

    # 카테고리
    logical_cat = p.get("dataset", {}).get("logical_category")
    if not logical_cat or logical_cat not in ["상의", "하의", "아우터", "모자", "신발"]:
        print(f"\n[스킵] 상품 ID {p_id}: 유효하지 않거나 누락된 카테고리입니다. (값: {logical_cat})")
        continue

    # 서브 카테고리
    subcat_title = p.get("category", {}).get("categoryDepth2Title", "")

    # 계절감
    season_info = p.get("season", {})
    primary_season = season_info.get("primary", "ALL")
    season_candidates = season_info.get("primary_candidates", [primary_season] if primary_season else [])

    # 상품 실측
    size_info = p.get("size", {})
    raw_measurements_list = size_info.get("measurements", [])
    normalized_measurements_list = []

    for size_item in raw_measurements_list:
        size_label = size_item.get("size")
        raw_m = size_item.get("measurements", {})
        norm_m = normalize_measurements(logical_cat, raw_m)
        
        if norm_m:
            normalized_measurements_list.append({
                "size": size_label,
                "sequence": size_item.get("sequence", 0),
                "measurements": norm_m
            })
    
    # Qdrant Payload 구성 (색상 필드는 빈 배열 처리)
    payload = {
        "product_id": int(p["product_id"]),
        "product_name": p.get("name", ""),
        "brand_name": p.get("brand", {}).get("name", ""),
        "category": logical_cat,
        "subcategory": subcat_title,
        "price": price_val,
        "season": primary_season,
        "season_candidates": season_candidates,
        "color_normalized": [],
        "features": p.get("features", {}),
        "sizes": normalized_measurements_list,
        "image_url": img_url,
        "product_url": p.get("source_url", "")
    }

    points.append(
        models.PointStruct(
            id=int(p["product_id"]),
            vector=all_embeddings[idx].tolist(),
            payload=payload
        )
    )

print(">> Qdrant DB에 포인트 배치 업서트 시작...")
UPSERT_BATCH_SIZE = 100
for i in range(0, len(points), UPSERT_BATCH_SIZE):
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points[i:i + UPSERT_BATCH_SIZE]
    )

print(f"🎉 인덱싱 성공! 총 {len(points)}건 적재 완료.")