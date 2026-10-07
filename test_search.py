from coordinator import FashionPipelineCoordinator

def main():
    print("============================================================")
    print(" Fashion-CLIP 기반 AI 스타일리스트 검색 엔진을 시작합니다.")
    print(" 종료하시려면 'q', 'quit', 'exit' 중 하나를 입력하세요.")
    print("============================================================\n")
    
    coordinator = FashionPipelineCoordinator()

    while True:
        # 1. 터미널에서 사용자 입력 받기
        q = input("\n👕 질의를 입력하세요: ").strip()

        # 2. 종료 조건
        if q.lower() in ['q', 'quit', 'exit']:
            print("시스템을 종료합니다.")
            break
            
        if not q:
            continue

        print("\n" + "="*80)
        print(f"사용자 질의: \"{q}\"")
        print("="*80)

        try:
            # 3. 코디네이터 실행
            output = coordinator.run(q)
            intent = output["intent"]

            print(f">> [TPO 분석]: {output['tpo']}")
            print(f">> [추출된 슬롯 및 영문 키워드]")
            for idx, slot in enumerate(intent.slots, 1):
                print(f"  - [{slot.category.upper()}] 영문 검색어: \"{slot.clip_query_en}\" (카테고리: {slot.category}, 핏: {slot.fit_type or 'None'})")
            print(f">> [AI 스타일리스트 코멘트]:\n{output['comment']}\n")

            # 4. 결과 출력 로직
            if output["type"] == "single":
                print(">> [추천 단품 목록]")
                for idx, it in enumerate(output["results"], 1):
                    fit_val = it.get('fit_type', 'regular')
                    size_rec = it.get('size_recommendation')
                    size_str = f"추천 사이즈: {size_rec.get('size_label')} (적합도: {size_rec.get('predicted_fit_score', 0)*100:.1f}점)" if size_rec else "사이즈 정보 없음"

                    print(f"  {idx}. [유사도: {it['similarity_score']}] [ID: {it['product_id']}] [{it['brand_name']}] {it['product_name']}")
                    print(f"    - 가격: {it['price']:,}원 | 색상: {it['colors']} | 핏: {fit_val}")
                    print(f"    - {size_str} [출처: {it.get('fit_source_type', 'none')}]")
                    print(f"    - 링크: {it['product_url']}")
            else:
                print(">> [추천 코디 세트]")
                for c_idx, outfit in enumerate(output["outfits"], 1):
                    print(f"\n  ★ [코디 셋 #{c_idx}] 조화도: {outfit['harmony_score']}점 | 총액: {outfit['total_price']:,}원")
                    
                    # Vision LLM 응답 키값 방어 로직 (verdict 또는 vision_reasoning)
                    verdict = outfit.get('verdict', outfit.get('vision_reasoning', ''))
                    print(f"    - 조합 총평: {verdict}")
                    
                    if 'styling_tip' in outfit:
                        print(f"    - 스타일링 팁: {outfit['styling_tip']}")
                        
                    # items가 딕셔너리일 경우와 리스트일 경우 모두 호환되도록 처리
                    items_list = outfit["items"].values() if isinstance(outfit["items"], dict) else outfit["items"]
                    
                    for it in items_list:
                        fit_val = it.get('fit_type', 'regular')
                        size_rec = it.get('size_recommendation')
                        size_str = f"추천 사이즈: {size_rec.get('size_label')} (적합도: {size_rec.get('predicted_fit_score', 0)*100:.1f}점)" if size_rec else "사이즈 정보 없음"
                        
                        print(f"    • [{it['category'].upper()}] [ID: {it['product_id']}] {it['brand_name']} - {it['product_name']}")
                        print(f"      - 가격: {it['price']:,}원 | 색상: {it['colors']} | 핏: {fit_val}")
                        print(f"      - 📏 {size_str} [출처: {it.get('fit_source_type', 'none')}]")
                        print(f"      - 링크: {it['product_url']}")

        except Exception as e:
            print(f"\n❌ 실행 중 오류가 발생했습니다: {e}")
            print("다시 입력해주세요.")

if __name__ == "__main__":
    main()