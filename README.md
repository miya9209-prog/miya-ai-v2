# 미야언니 GPT 중심 재설계 버전

## 적용
1. 기존 레포를 백업합니다.
2. 이 폴더의 파일 전체를 기존 레포에 덮어씁니다.
3. Streamlit secrets 또는 환경변수에 `OPENAI_API_KEY`를 넣습니다.
4. 필요하면 `OPENAI_MODEL` 환경변수로 모델명을 바꿀 수 있습니다. 기본값은 `gpt-4.1-mini`입니다.

## 포함 데이터
- misharp_miya_db.csv
- review_summary.json
- model_profiles.json
- customer_profiles_template.csv

## 고객 이름 호출
- URL query: `customer_name=홍길동`
- 또는 `customer_profiles.csv`를 추가하고 `customer_id`, `login_id`, `email` query를 넘기면 이름 매칭 가능

## 핵심 구조
- GPT가 상담 본문 생성
- 상품DB/후기/모델/고객정보는 GPT에게 근거로 전달
- 규칙은 후보 검색, 현재 상품 확인, 데이터 정리에만 사용
