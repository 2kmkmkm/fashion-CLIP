from core.query_parser import QueryParser
from core.search_engine import FashionSearchEngine
from core.harmonizer import OutfitHarmonizer
from core.response_generator import ResponseGenerator
import itertools

class FashionPipelineCoordinator:
    def __init__(self):
        print(">> 패션 추천 파이프라인 컴포넌트 초기화 중...")
        self.parser = QueryParser()
        self.engine = FashionSearchEngine()
        self.harmonizer = OutfitHarmonizer(self.engine)
        self.responder = ResponseGenerator()
        print(">> 전체 파이프라인 준비 완료.")

    def run(self, user_query: str) -> dict:
        print(f"\n[DEBUG] === 파이프라인 실행 시작 ===")
        
        # [1단계] LLM 질의 파싱 (예산, 계절, 슬롯별 1.3배 소프트 버퍼 가격 포함)
        intent = self.parser.parse(user_query)
        print(f"[DEBUG] 1. 파서 결과: 의도={intent.search_type}, 예산={intent.total_budget}, 슬롯 수={len(intent.slots)}개")

        # 의도가 single이거나 추출된 슬롯이 딱 1개뿐인 경우 무조건 단품 검색으로 처리
        if intent.search_type == "single" or len(intent.slots) == 1:
            slot = intent.slots[0]
            items = self.engine.search(
                query_text=slot.clip_query_en,
                category=slot.category,
                season=intent.season,
                max_price=slot.max_price,
                color=slot.color,
                top_k=5
            )
            print(f"[DEBUG] 2. 검색 엔진 결과: 단품 후보 {len(items)}개 추출 완료")
            
            comment = self.responder.generate_single_response(user_query, intent.tpo_summary, items)
            return {
                "type": "single",
                "query": user_query,
                "tpo": intent.tpo_summary,
                "comment": comment,
                "results": items
            }

        # [코디 검색 분기] 슬롯이 2개 이상 복합으로 들어온 경우만 진입
        else:
            slot_candidates = {}
            for slot in intent.slots:
                items = self.engine.search(
                    query_text=slot.clip_query_en,
                    category=slot.category,
                    season=intent.season,
                    max_price=slot.max_price,
                    color=slot.color,
                    top_k=5
                )
                slot_candidates[slot.slot_name] = items
                print(f"[DEBUG] 2. 검색 엔진 결과: 슬롯 '{slot.slot_name}' -> {len(items)}개 후보 찾음")

            # 단 하나의 슬롯이라도 검색 결과가 0개면 파이프라인 즉시 중단
            empty_slots = [s for s, items in slot_candidates.items() if len(items) == 0]
            if empty_slots:
                print(f"[DEBUG] 🚨 비상 상황: 다음 슬롯의 검색 결과가 0개입니다 -> {empty_slots}")
                print("[DEBUG] 조합할 상품이 부족하여 파이프라인을 중단합니다. (Qdrant DB 점검 요망)")
                return {
                    "type": "coordination",
                    "query": user_query,
                    "tpo": intent.tpo_summary,
                    "comment": f"죄송합니다. 현재 DB에서 [{', '.join(empty_slots)}]에 해당하는 적절한 상품을 찾지 못해 코디를 구성할 수 없었습니다. 검색 조건을 바꿔보시거나 DB 인덱싱 상태를 확인해주세요.",
                    "outfits": []
                }

            # 조합 단계에서 예산 Hard Cut 적용 (total_price <= user_budget)
            categories = list(slot_candidates.keys())
            item_lists = [slot_candidates[cat] for cat in categories]
            
            total_possible_combos = len(list(itertools.product(*item_lists)))
            print(f"[DEBUG] 3. 생성 가능한 전체 조합 경우의 수: {total_possible_combos}개")

            valid_combos = []
            for combo in itertools.product(*item_lists):
                total_price = sum(item["price"] for item in combo)
                # 사용자가 지정한 원래 총예산을 초과하면 즉시 배제
                if intent.total_budget and total_price > intent.total_budget:
                    continue
                valid_combos.append(combo)

            print(f"[DEBUG] 4. 예산 하드컷 통과 조합: {len(valid_combos)}개")

            # 만약 하드컷에 걸려 남은 조합이 없다면 전체 후보 조합 중 예산에 가장 근접한 것들로 Fallback
            if not valid_combos:
                print(f"[DEBUG] 🚨 예산 제한으로 모든 조합이 탈락했습니다. 강제로 5개를 추출(Fallback)합니다.")
                valid_combos = list(itertools.product(*item_lists))[:5]

            # [3단계] 조화도 채점 및 Vision Re-ranking (Top-5 압축 검수)
            print(f"[DEBUG] 5. 조화도 검수기(Harmonizer) 실행 중... (Top-5 1차 추출 -> Vision LLM 2차 심사)")
            best_outfits = self.harmonizer.select_best_outfits(
                slot_candidates, 
                intent.tpo_summary, 
                top_k=2, 
                pre_filtered_combos=valid_combos
            )
            print(f"[DEBUG] 6. 최종 추천 코디 세트 확정: {len(best_outfits)}개")

            # [4단계] 자연어 응답 생성
            comment = self.responder.generate_coordination_response(user_query, intent.tpo_summary, best_outfits)
            return {
                "type": "coordination",
                "query": user_query,
                "tpo": intent.tpo_summary,
                "comment": comment,
                "outfits": best_outfits
            }