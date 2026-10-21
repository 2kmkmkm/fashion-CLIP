# FITzza POC: 로컬 개발 및 테스트 환경 가이드

본 저장소는 무신사 상품 데이터 및 Fashion-CLIP 기반 벡터 검색을 활용한 AI 스타일링/코디 추천 POC 프로젝트입니다.

사전 인덱싱된 775건의 패션 상품 벡터 데이터가 번들링된 Docker 환경을 제공하여, 별도의 모델 다운로드 및 임베딩 연산 과정 없이 **명령어 한 번(`docker compose up -d`)으로 즉시 Qdrant Vector DB를 기동**할 수 있습니다.

---

## 🚀 빠른 시작 (Quick Start)

### 1. Qdrant Vector DB 실행 (Docker Compose)

사전 인덱싱 데이터(`musinsa_products.snapshot`)가 탑재된 컨테이너가 1초 만에 기동됩니다.

```bash
# Docker Hub(02dlrudals/fitzza-qdrant:latest)에서 이미지를 받아 백그라운드로 실행
docker compose up -d

# 또는 Docker CLI로 직접 실행 시:
# docker run -d -p 6333:6333 -p 6334:6334 --name fitzza_qdrant 02dlrudals/fitzza-qdrant:latest

# 실행 상태 및 헬스체크 확인
docker compose ps
```

- **Qdrant REST API**: `http://localhost:6333`
- **Qdrant Web Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard) 접속 후 `musinsa_products` 컬렉션 확인
- **컨테이너 종료**: `docker compose down`

---

### 2. Python 가상환경 및 라이브러리 설치

```bash
# 가상환경 생성 및 활성화 (Python 3.10+ 권장)
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate       # Windows PowerShell

# 의존성 패키지 설치
pip install -r requirements.txt
```

---

### 3. Qdrant 데이터 적재 상태 검증

아래 검증 스크립트를 실행하여 Qdrant에 775개 상품 데이터가 정상 적재되었는지 바로 확인합니다.

```bash
python scripts/verify_qdrant.py
```

출력 예시:
```text
[OK] Collection status: green
[OK] Total Points Count: 775

[Sample Products]
 - [상의] 곤니치와봉쥬르 | 씨비 너드 체크 셔츠 15종 (67000원)
 - [신발] 토앤토 | Zerovity OG 베이비블루 (59000원)
 - [상의] 피인오 | 썸머 리브드 헨리넥 숏 슬리브 (48750원)
 ...
🎉 Qdrant 컨테이너 및 사전 적재 데이터 정상 동작 확인 완료!
```

---

## 📁 주요 구성 파일 안내

```text
fitzza_poc/
├── docker/
│   └── qdrant_snapshots/
│       └── musinsa_products.snapshot  # 사전 인덱싱된 775건 상품 벡터 스냅샷 (~4.8MB)
├── Dockerfile                         # 스냅샷 자동 복원 엔트리포인트를 포함한 Qdrant 이미지
├── docker-compose.yml                 # 원클릭 Qdrant 컨테이너 실행 구성 파일
├── requirements.txt                   # 프로젝트 Python 의존성 라이브러리 목록
├── config/
│   └── color_map.json                 # 색상 정규화 매핑 데이터
├── scripts/
│   ├── verify_qdrant.py               # Qdrant 연결 및 적재 데이터 검증 스크립트
│   ├── index_to_qdrant.py             # 신규 데이터 수집 시 Qdrant 인덱싱 재실행 스크립트
│   ├── generate_color_map.py          # 색상 매핑 생성 스크립트
│   └── color_map_overrides.json       # 색상 수동 오버라이드 룰
└── core/
    └── color_palette.py               # 색상 매핑 기준 팔레트 정의
```

---

## 🔄 신규 데이터 재인덱싱이 필요한 경우 (참고)

새로운 크롤링 데이터나 상품이 추가되어 스냅샷을 갱신하고자 할 때는 기존 스크립트를 사용합니다:

1. `docker compose up -d`로 빈 Qdrant 기동 (또는 기존 볼륨 삭제)
2. `python scripts/index_to_qdrant.py` 실행 (임베딩 추출 및 Qdrant Upsert)
3. 스냅샷 내보내기:
   ```bash
   curl -X POST "http://localhost:6333/collections/musinsa_products/snapshots"
   ```
4. 생성된 스냅샷 파일을 `docker/qdrant_snapshots/musinsa_products.snapshot` 위치에 업데이트한 후 Dockerfile을 재빌드합니다.
