from coordinator import FashionPipelineCoordinator

def main():
    coordinator = FashionPipelineCoordinator()

    test_queries = [
        # 1. 단품 질의
        "오늘 날씨에 딱 맞는 아우터 추천해줘",
        "여자친구랑 피크닉 갈 때 신을 신발 추천해줘",
        
        # 2. 코디 질의
        "나트랑 여행룩 추천해줘",
        "결혼식 하객룩 추천해줘"
    ]

    for q in test_queries:
        print("\n" + "="*80)
        print(f"사용자 질의: \"{q}\"")
        print("="*80)

        output = coordinator.run(q)

        print(f">> [TPO 분석]: {output['tpo']}")
        print(f">> [AI 스타일리스트 코멘트]:\n{output['comment']}\n")

        if output["type"] == "single":
            print(">> [추천 단품 목록]")
            for idx, it in enumerate(output["results"], 1):
                print(f" {idx}. [유사도: {it['similarity_score']}] [ID: {it['product_id']}] [{it['brand_name']}] {it['product_name']}")
                print(f"    - 가격: {it['price']:,}원 | 색상: {it['colors']} | 링크: {it['product_url']}")
        else:
            print(">> [추천 코디 세트]")
            for c_idx, outfit in enumerate(output["outfits"], 1):
                print(f"\n  ★ [코디 셋 #{c_idx}] 조화도: {outfit['harmony_score']}점 | 총액: {outfit['total_price']:,}원")
                print(f"     - 조합 총평: {outfit['verdict']}")
                print(f"     - 스타일링 팁: {outfit['styling_tip']}")
                for it in outfit["items"]:
                    print(f"     • [{it['category'].upper()}] [ID: {it['product_id']}] {it['brand_name']} - {it['product_name']}")
                    print(f"       - 가격: {it['price']:,}원 | 색상: {it['colors']}")
                    print(f"       - 링크: {it['product_url']}")
                    
if __name__ == "__main__":
    main()