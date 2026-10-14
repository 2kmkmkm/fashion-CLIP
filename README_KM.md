# FITzza: Qdrant 상품 데이터 인덱싱 가이드 (`index_to_qdrant.py`)

무신사 크롤링 상품 정제 데이터(`JSONL`)와 로컬 이미지(누끼/정면컷/썸네일)를 기반으로 **Fashion-CLIP 512차원 시각 임베딩 벡터**를 추출하고, **Qdrant Vector DB**에 배치 적재(Upsert)하는 스크립트입니다.

---

## 1. 사전 필수 요구사항 (Prerequisites)

- **Python:** 3.10 이상 권장
- **Docker / Docker Desktop:** Qdrant 서버 컨테이너 구동용
- **하드웨어 환경:** CPU 또는 CUDA 지원 GPU (자동 감지)

---

## 2. 가상환경 구성 및 패키지 설치

```bash

# 1. 필수 라이브러리 설치
pip install torch torchvision
pip install transformers pillow tqdm numpy qdrant-client

# 2. Qdrant Docker 컨테이너 실행
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant

# Qdrant 웹 대시보드 확인: 브라우저에서 http://localhost:6333/dashboard 접속

# 상품 임베딩 실행
python index_to_qdrant.py

# 테스트 코드 실행
python test_search.py

## 📁 프로젝트 디렉터리 구조 (Directory Structure)

```text
fitzza_poc/
├── core/
│   ├── harmonizer.py          # 1차 벡터 조화도 스크리닝 및 2차 Vision LLM 검증
│   ├── query_parser.py        # 사용자 자연어 질의 TPO 의도 분석 및 슬롯 파싱
│   ├── response_generator.py  # 추천 코디 최종 사용자 응답 및 스타일링 팁 생성
│   └── search_engine.py       # Fashion-CLIP 임베딩 및 Qdrant HNSW 코사인 검색 엔진
├── .env                       # OpenAI API Key 및 환경 변수 설정 파일
├── coordinator.py             # 전체 추천 파이프라인 오케스트레이션 메인 모듈
├── index_to_qdrant.py         # 500개 상품 이미지 Fashion-CLIP 임베딩 및 Qdrant 적재
├── test_search.py             # 단품 및 조합 검색 기능/지연 시간 검증 단위 테스트
└── README.md                  # 프로젝트 안내 및 가이드 문서
```

---

## 3. 색상 매핑표 만들기 (Jev)

`index_to_qdrant.py`는 `config/color_map.json`으로 각 상품의 `color_image_mappings[].color_raw`를 기본 색상으로 바꿔 `color_normalized`에 저장합니다. 이 매핑표는 TypeSafe의 Jev로 만듭니다(이전 GPT-4o-mini 버전 대체).

```bash
# .env 또는 환경 변수에 TYPESAFE_API_KEY 필요 (키는 출력하지 않음)
python scripts/generate_color_map.py            # 새로 생긴 color_raw만 물어봄
python scripts/generate_color_map.py --fresh    # 색상 목록·지시문을 바꿨을 때 전부 다시
python scripts/index_to_qdrant.py
```

- 색상 목록은 `core/color_palette.py` 한 곳에서 관리합니다: black, white, ivory, beige, gray, charcoal, navy, blue, green, khaki, brown, red, pink, purple, yellow, orange, multi(여러 색), etc(판단 불가). 사용자 질문에서 뽑는 색상도 이 목록(`FILTERABLE_COLORS`)과 같아야 필터가 맞습니다.
- color_raw마다 Jev에 보기 고르기 질문을 한 번 보냅니다(같은 이름은 한 번, 기본 8개 동시 요청). 응답은 `config/color_map_jev_results.jsonl`에 남아 다시 실행할 때 재사용합니다.
- 확신이 0.6(`--threshold`) 미만이면 매핑표에는 `etc`로 넣고 `config/color_map_review.csv`에 모읍니다. 사람이 보고 정한 값은 `scripts/color_map_overrides.json`에 넣으면 Jev 결과보다 우선합니다(예: `"na": "navy"`처럼 약어).
- 파일럿 100개 상품(고유 color_raw 193개) 기준: 직접 만든 정답표와 비교해 Jev 192/193, 같은 조건의 GPT-4o-mini 158/193, 기존 GPT 프롬프트 171/193(목록 밖 값 `light gray` 등 11개). 정답표와 지시문을 Jev 결과를 보며 다듬었으므로 참고용 수치입니다.
