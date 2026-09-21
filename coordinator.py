from core.query_parser import QueryParser
from core.search_engine import FashionSearchEngine
from core.harmonizer import OutfitHarmonizer
from core.response_generator import ResponseGenerator

# 전체 파이프라인 중앙 제어기

class FashionPipelineCoordinator:
    def __init__(self):
        print(">> 패션 추천 파이프라인 컴포넌트 초기화 중...")
        self.parser = QueryParser()
        self.engine = FashionSearchEngine()
        self.harmonizer = OutfitHarmonizer(self.engine)
        self.responder = ResponseGenerator()
        print(">> 전체 파이프라인 준비 완료.")

    def run(self, user_query: str) -> dict:
        # [1단계] LLM 질의 파싱
        intent = self.parser.parse(user_query)

        # [단품 검색 분기]
        if intent.search_type == "single":
            slot = intent.slots[0]
            items = self.engine.search(
                query_text=slot.clip_query_en,
                category=slot.category,
                color=slot.color,
                max_price=slot.max_price,
                top_k=3
            )
            comment = self.responder.generate_single_response(user_query, intent.tpo_summary, items)
            return {
                "type": "single",
                "query": user_query,
                "tpo": intent.tpo_summary,
                "comment": comment,
                "results": items
            }

        # [코디 검색 분기]
        else:
            slot_candidates = {}
            for slot in intent.slots:
                items = self.engine.search(
                    query_text=slot.clip_query_en,
                    category=slot.category,
                    color=slot.color,
                    max_price=slot.max_price,
                    top_k=3
                )
                slot_candidates[slot.slot_name] = items

            # [3단계] 조화도 채점 및 Vision Re-ranking
            best_outfits = self.harmonizer.select_best_outfits(slot_candidates, intent.tpo_summary, top_k=2)

            # [4단계] 자연어 응답 생성
            comment = self.responder.generate_coordination_response(user_query, intent.tpo_summary, best_outfits)
            return {
                "type": "coordination",
                "query": user_query,
                "tpo": intent.tpo_summary,
                "comment": comment,
                "outfits": best_outfits
            }