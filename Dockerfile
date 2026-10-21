# Qdrant 공식 이미지 (v1.19.1)
FROM qdrant/qdrant:v1.19.1

LABEL maintainer="fitzza-team"
LABEL description="Pre-indexed Qdrant Vector DB with Musinsa fashion products dataset"

# 스냅샷 디렉터리 생성 및 사전 인덱싱된 스냅샷 파일 복사 (약 4.8MB)
RUN mkdir -p /qdrant/snapshots
COPY ./docker/qdrant_snapshots/musinsa_products.snapshot /qdrant/snapshots/musinsa_products.snapshot

# 컨테이너 시작 시 musinsa_products 스냅샷을 자동 복구하여 기동
ENTRYPOINT ["./entrypoint.sh", "--snapshot", "/qdrant/snapshots/musinsa_products.snapshot:musinsa_products"]
