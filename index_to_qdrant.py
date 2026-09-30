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


DATASET_PATH = "outputs/products_dataset_completed_v2.jsonl" 
if not os.path.exists(DATASET_PATH):
    # 만약 v2 파일이 없다면 v1 등으로 변경 가능
    DATASET_PATH = "data/products_dataset_completed_v1.jsonl"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"🚨 데이터셋 파일을 찾을 수 없습니다: {DATASET_PATH}")

products = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            products.append(json.loads(line))

print(f">> 데이터셋 파일 로드 완료: 총 {len(products)}개 상품")


valid_records = []
pil_images_to_embed = []
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'}

for p in tqdm(products, desc="상품 이미지 안전 로딩 중"):
    p_id = str(p.get("product_id"))
    image_url = ""
    local_path = ""
    
    # 1순위: selected_images 안의 front local_path나 image_url 확인
    selected_front = (p.get("selected_images") or {}).get("front") or {}
    local_path = selected_front.get("local_path")
    image_url = selected_front.get("image_url")
    
    # 2순위: 만약 위 정보가 없으면 images 배열의 첫 번째 항목(썸네일 등) 활용
    if not image_url:
        images_list = p.get("images", [])
        if images_list:
            image_url = images_list[0].get("image_url", "")

    img = None
    # 로컬 캐시에 파일이 존재하면 즉시 로드
    if local_path and os.path.exists(local_path):
        try:
            img = Image.open(local_path).convert("RGB")
        except Exception:
            pass

    # 로컬에 없으면 원본 URL을 통해 다운로드 시도
    if img is None and image_url:
        try:
            response = requests.get(image_url, headers=headers, timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            pass

    if img is not None:
        valid_records.append((p, image_url))
        pil_images_to_embed.append(img)
    else:
        print(f"\n[스킵] 상품 ID {p_id}: 이미지를 로드할 수 없습니다.")

print(f">> 최종 유효 이미지 로드 성공: {len(pil_images_to_embed)} / {len(products)} 건")

if not pil_images_to_embed:
    print(">> 🚨 임베딩할 이미지가 0건입니다. 데이터셋 경로를 확인해주세요.")
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
    price_val = int(p.get("price") or p.get("sale_price") or 0)
    
    payload = {
        "product_id": int(p["product_id"]),
        "product_name": p.get("product_name", ""),
        "brand_name": "brand_name" in p and p["brand_name"] or "",
        "category": p.get("category") or p.get("source_category", ""),
        "subcategory": p.get("subcategory", ""),
        "price": price_val,
        "season": p.get("season", "ALL"),
        "color_normalized": p.get("color_normalized", []),
        "fit_normalized": p.get("fit_normalized", []),
        "image_url": img_url,
        "product_url": p.get("product_url", "")
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