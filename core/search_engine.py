import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Fashion-CLIP + Qdrant
# 사용자가 요청한 패션 키워드를 Fashion-CLIP 모델을 통해 512차원 벡터로 변환
# Qdrant 벡터 DB에서 카테고리, 예산 상한선, 계절, 색상 등의 하드 필터를 적용
# 1차 후보 추출: 카테고리별 5개씩

class FashionSearchEngine:
    def __init__(self, qdrant_url="http://localhost:6333", collection_name="musinsa_products"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_name = "patrickjohncyh/fashion-clip"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(p=2, dim=-1, keepdim=True)

    def encode_text(self, text: str) -> list:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0, :]
            feats = self._normalize(feats)
        return feats.cpu().numpy()[0].tolist()

    def search(self,
               query_text: str = None, 
               category: str = None,
               season: str = "ALL",
               max_price: int = None, 
               color: str = None,
               top_k: int = 3) -> list:
        
        query_vector = self.encode_text(query_text) # 질의 텍스트를 Fashion-CLIP 벡터로 변환

        # 데이터베이스 레벨에서 반드시 만족해야 하는 하드 필터 구성 (카테고리 + 예산 + 색상)
        must_conditions = [
            models.FieldCondition(key="category", match=models.MatchValue(value=category))
        ]

        if max_price is not None:
            must_conditions.append(models.FieldCondition(key="price", range=models.Range(lte=max_price)))

        if color is not None:
            must_conditions.append(models.FieldCondition(key="color_normalized", match=models.MatchValue(value=color)))
            
        # 계절 하이브리드 필터 (요청된 계절 + 'ALL' 우선 인출)
        should_conditions = [
            models.FieldCondition(key="season", match=models.MatchValue(value=season)),
            models.FieldCondition(key="season", match=models.MatchValue(value="ALL"))
        ]

        query_filter = models.Filter(must=must_conditions, should=should_conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k * 2,
            with_vectors=True
        )

        results = []
        for hit in response.points:
            # 유사도 0.2 미만은 컷오프
            # 데이터셋 늘어나면 수정
            if hit.score < 0.2:
                continue
            p = hit.payload
            results.append({
                "similarity_score": round(hit.score, 4),
                "product_id": p.get("product_id"),
                "product_name": p.get("product_name"),
                "brand_name": p.get("brand_name"),
                "category": p.get("category"),
                "price": p.get("price"),
                "colors": p.get("color_normalized"),
                "fit_type": p.get("fit_type"),
                "image_url": p.get("image_url"),
                "product_url": p.get("product_url"),
                "vector": hit.vector
            })
        
        # 컷오프 때문에 후보가 비어버릴 경우 비상 방어 (Top 1 반환)
        if not results and response.points:
            hit = response.points[0]
            p = hit.payload
            results.append({
                "similarity_score": round(hit.score, 4),
                "product_id": p.get("product_id"),
                "product_name": p.get("product_name"),
                "brand_name": p.get("brand_name"),
                "category": p.get("category"),
                "price": p.get("price"),
                "colors": p.get("color_normalized"),
                "fit_type": p.get("fit_type"),
                "image_url": p.get("image_url"),
                "product_url": p.get("product_url"),
                "vector": hit.vector
            })

        return results[:top_k]