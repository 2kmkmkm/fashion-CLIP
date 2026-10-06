class SizeProfileResolver:
    def __init__(self, db_client):
        self.db = db_client

    def resolve_and_calculate_size(self, user_id: str, subcategory: str, fit_type: str, candidate_products: list, category_type: str):
        """
        [1. 추천 진입점]
        우선순위 체인을 타서 가장 신뢰도 높은 베이스라인(기준 치수)을 확보한 뒤,
        SizeCalculator를 통해 최적의 사이즈를 도출합니다.
        """
        # 유저 상태별 우선순위 체인을 거쳐 베이스라인 실측 스펙 조회 (subcategory와 fit_type 기준)
        baseline_spec = self._get_baseline_spec(user_id, subcategory, fit_type)
        
        # 만약 유저 베이스라인 정보가 없다면 사이즈 추천 생략
        if not baseline_spec or baseline_spec["data"] is None:
            return {
                "source_type": "none",
                "baseline_spec": None,
                "recommendation": None
            }
        
        # 확보한 베이스라인의 알맹이 데이터("data")와 카테고리 타입을 SizeCalculator에 전달
        recommended_result = SizeCalculator.evaluate(candidate_products, baseline_spec["data"], category_type)
        
        return {
            "source_type": baseline_spec["source"],  # 어떤 순위의 데이터를 사용했는지
            "baseline_spec": baseline_spec["data"],  # 기준이 된 실측 스펙
            "recommendation": recommended_result  # 최종 추천된 상품 및 사이즈 결과
        }

    def _get_baseline_spec(self, user_id: str, subcategory: str, fit_type: str) -> dict:
        """
        [2. 우선순위 체인 (Fallback Chain) 라우팅]
        유저의 데이터 성숙도에 따라 1순위부터 3순위까지 차례대로 탐색합니다.
        """
        # 1순위: 본인의 "맞다"로 학습된 선호 실측 (user_fit_preferences)
        learned_pref = self._fetch_user_fit_preferences(user_id, subcategory, fit_type)
        if learned_pref:
            return {
                "source": "learned_preference",
                "data": learned_pref
            }

        # 2순위: 본인의 최근 구매 이력 상품의 실측 (카테고리 + fit_type 고려)
        recent_order_spec = self._fetch_recent_purchased_spec(user_id, subcategory, fit_type)
        if recent_order_spec:
            return {
                "source": "recent_purchase",
                "data": recent_order_spec
            }

        # 3순위: 키/몸무게 기반 비슷한 체형 타 사용자의 "맞다" 인증 데이터
        similar_user_spec = self._fetch_similar_body_spec(user_id, subcategory, fit_type)
        if similar_user_spec:
            return {
                "source": "similar_body_profile",
                "data": similar_user_spec
            }

        # 4순위 및 콜드스타트: 정보가 아예 없는 경우 None 처리
        return {
            "source": "none",
            "data": None
        }
        
    def update_fit_preferences(self, user_id: str, subcategory: str, fit_type: str, purchased_size_spec: dict):
        """
        [3. Closed-Loop 학습 루프]
        유저가 설문조사에서 "FIT(맞다)"를 선택했을 때 실행됩니다.
        기존에 쌓인 데이터와 새로운 실측값을 결합하여 누적 평균으로 갱신합니다.
        """
        existing_pref = self._fetch_user_fit_preferences(user_id, subcategory, fit_type)

        if not existing_pref:
            # 샘플 수 1개: 현재 구매한 상품의 실측을 그대로 초기값으로 등록
            self._insert_fit_preferences(user_id, subcategory, fit_type, purchased_size_spec, sample_count=1)
        else:
            # 이미 데이터가 있는 경우: 이동 평균(Moving Average) 공식을 적용하여 고도화
            n = existing_pref["sample_count"]
            updated_spec = {}
            
            for key, val in purchased_size_spec.items():
                if key in existing_pref and existing_pref[key] is not None and val is not None:
                    updated_spec[key] = round((existing_pref[key] * n + val) / (n + 1), 1)
                else:
                    updated_spec[key] = val if val is not None else existing_pref.get(key)
            
            updated_spec["sample_count"] = n + 1
            
            # 갱신된 선호 스펙을 DB에 반영
            self._update_fit_preferences_db(user_id, subcategory, fit_type, updated_spec)

    # --- 내부 DB 조회 및 갱신 헬퍼 메서드 (Stub) ---
    def _fetch_user_fit_preferences(self, user_id, subcategory, fit_type):
        pass

    def _fetch_recent_purchased_spec(self, user_id, subcategory, fit_type):
        pass

    def _fetch_similar_body_spec(self, user_id, subcategory, fit_type):
        pass

    def _insert_fit_preferences(self, user_id, subcategory, fit_type, spec, sample_count):
        pass

    def _update_fit_preferences_db(self, user_id, subcategory, fit_type, updated_spec):
        pass


class SizeCalculator:
    @staticmethod
    def evaluate(candidate_products: list, baseline_spec: dict, category_type: str) -> dict:
        """
        [4. 수학적 산출 엔진]
        분리된 상/하의 실측 항목(top_length, bottom_length, top_hem, bottom_hem 등)을 반영하고,
        기준 치수와 비교하여 오차가 가장 적은 최적의 상품과 사이즈를 결정합니다.
        """
        if not baseline_spec:
            return None
            
        best_match = None
        min_penalty = float('inf')  # 최소 오차 비교를 위해 무한대로 초기화

        # 추천 후보로 올라온 여러 개의 상품들을 순회
        for product in candidate_products:
            if "sizes" not in product or not product["sizes"]:
                continue

            for size_option in product["sizes"]:
                penalty = 0.0
                valid_fields_count = 0  # 실제로 유효하게 비교된 필드 개수

                fields = []

                # 1. 카테고리별 세부 실측 항목 매핑 (총장 및 밑단 구분 적용)
                if category_type == "top":  # 상의 / 아우터
                    fields = ["top_length", "shoulder", "chest", "sleeve", "top_hem", "cuff"]
                elif category_type == "bottom":  # 하의
                    fields = ["bottom_length", "waist", "hip", "thigh", "rise", "bottom_hem"]
                elif category_type == "shoes":  # 신발
                    fields = ["foot_length", "foot_width", "ankle_height", "heel_height"]
                elif category_type == "hat":  # 모자
                    fields = ["head_circumference", "brim_length", "cap_depth"]
                else:
                    continue  # 정의되지 않은 카테고리는 스킵

                # 2. 공통 필드 순회 및 Null 엄격 방어 로직 적용
                for field in fields:
                    if (
                        field in size_option and size_option[field] is not None and
                        field in baseline_spec and baseline_spec[field] is not None
                    ):
                        penalty += abs(size_option[field] - baseline_spec[field])
                        valid_fields_count += 1

                # 비교할 수 있는 유효한 데이터가 하나도 없다면 이 사이즈 옵션은 스킵
                if valid_fields_count == 0:
                    continue

                # 3. 최소 오차 갱신 및 15cm 마진 기준 점수화
                if penalty < min_penalty:
                    min_penalty = penalty
                    best_match = {
                        "product_id": product.get("product_id"),
                        "size_label": size_option.get("size_label"),
                        "predicted_fit_score": max(0.0, 1.0 - (min_penalty / 15.0))  # 15cm 기준 마진
                    }
                    
        return best_match