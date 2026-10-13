import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Fashion-CLIP + Qdrant
# 사용자가 요청한 패션 키워드를 Fashion-CLIP 모델을 통해 512차원 벡터로 변환
# Qdrant 벡터 DB에서 카테고리, 예산 상한선, 계절 등의 하드 필터를 적용
# 1차 후보 추출: 카테고리별 5개씩

# top_k = 3
# DB 레벨 (2단계): 수만 개의 상품 중 15개(top_k * 5)만 추려내어 대규모 필터링 
# 노이즈 컷오프 (3단계): 유사도 0.2 미만 필터링
# Re-ranking (4~5단계): 색상, 핏, 계절 우선순위, 예산 초과 여부를 따져 상위 3개(top_k)만 최종 생존시킴

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
               season_preferences: list = None,
               max_price: int = None, 
               color: str = None,
               fit_type: str = None,
               top_k: int = 3) -> list:
        
        query_vector = self.encode_text(query_text) # 질의 텍스트를 Fashion-CLIP 벡터로 변환

        # 영문 카테고리를 Qdrant에 저장된 한글 카테고리로 변환하는 매핑
        category_map = {
            "top": "상의",
            "outer": "아우터",
            "bottom": "하의",
            "shoes": "신발",
            "hat": "모자"
        }
        db_category = category_map.get(category, category) if category else None

        # 데이터베이스 레벨에서 반드시 만족해야 하는 하드 필터 (카테고리)
        must_conditions = []
        if db_category is not None:
            must_conditions.append(
                models.FieldCondition(key="category", match=models.MatchValue(value=db_category))
            )

        # 사용자가 선호하는 계절 배열 원소들을 하나라도 포함하면 통과
        should_conditions = []
        if season_preferences:
            for s in season_preferences:
                should_conditions.append(
                    models.FieldCondition(key="season_candidates", match=models.MatchValue(value=s))
                )

        query_filter = models.Filter(
            must=must_conditions, 
            should=should_conditions if should_conditions else None
        )
        
        # 후속 Re-ranking을 위해 넉넉하게 5배수(top_k * 5) 후보군 추출
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k * 5,
            with_vectors=True
        )

        results = []
        for hit in response.points:
            # 최소 노이즈 컷오프
            if hit.score < 0.2:
                continue
            
            p = hit.payload
            final_score = hit.score
            
            # Python 기반 Re-ranking (가중치 점수 보정 로직)
            # 1. 색상 일치 보너스
            if color and color in p.get("color_normalized", []):
                final_score += 0.05
                
            # 2. 핏 성향 일치 보너스
            if fit_type and p.get("fit_type") == fit_type:
                final_score += 0.05

            # 3. 계절 우선순위 배열 기반 가중치 차등 보정
            product_seasons = p.get("season_candidates", [])
            if season_preferences:
                best_match_idx = float('inf')
                for u_season in season_preferences:
                    if u_season in product_seasons:
                        idx = season_preferences.index(u_season)
                        if idx < best_match_idx:
                            best_match_idx = idx

                # 0번째(1순위) 일치 시 가장 큰 보너스, 후순위일수록 소폭 보너스 부여
                if best_match_idx == 0:
                    final_score += 0.06
                elif best_match_idx == 1:
                    final_score += 0.03
                elif best_match_idx > 1:
                    final_score += 0.01

            # 4. 예산 초과 감점 (1.3배 버퍼 내에 들어왔으나 원래 예산을 초과한 경우)
            if max_price:
                actual_price = p.get("price", 0)
                original_budget = max_price / 1.3
                if actual_price > original_budget:
                    over_ratio = (actual_price - original_budget) / original_budget
                    final_score -= (over_ratio * 0.1)

            results.append({
                "final_score": final_score,
                "similarity_score": round(hit.score, 4),
                "product_id": p.get("product_id"),
                "product_name": p.get("product_name"),
                "brand_name": p.get("brand_name"),
                "category": p.get("category"),
                "subcategory": p.get("subcategory"),
                "price": p.get("price"),
                "colors": p.get("color_normalized"),
                "fit_type": p.get("fit_type"),
                "season_candidates": product_seasons,
                "sizes": p.get("sizes", []),
                "image_url": p.get("image_url"),
                "product_url": p.get("product_url"),
                "vector": hit.vector
            })
        
        # 보정된 최종 점수 기준으로 재정렬 후 상위 top_k개 반환
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]