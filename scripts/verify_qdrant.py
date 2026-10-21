import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from qdrant_client import QdrantClient

def test_qdrant():
    client = QdrantClient(url="http://localhost:6333")
    
    # 1. 컬렉션 메타데이터 확인
    info = client.get_collection("musinsa_products")
    print(f"[OK] Collection status: {info.status}")
    print(f"[OK] Total Points Count: {info.points_count}")
    
    assert info.points_count > 0, "No points in collection!"
    
    # 2. 샘플 데이터 스크롤 조회
    records, _ = client.scroll(
        collection_name="musinsa_products",
        limit=5,
        with_payload=True,
        with_vectors=False
    )
    
    print("\n[Sample Products]")
    for r in records:
        p = r.payload
        print(f" - [{p.get('category')}] {p.get('brand_name')} | {p.get('product_name')} ({p.get('price')}원)")
    
    print("\n🎉 Qdrant 컨테이너 및 사전 적재 데이터 정상 동작 확인 완료!")

if __name__ == "__main__":
    test_qdrant()
