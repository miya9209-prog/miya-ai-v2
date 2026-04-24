# miya-ai-v2 main full repo

전체 덮어쓰기용 패키지입니다.

포함 파일:
- app.py
- misharp_miya_db.csv
- review_summary.json
- model_profiles.json
- customer_profiles_template.csv
- build_review_summary.py
- requirements.txt
- logs/

적용 방법:
1. 기존 레포를 백업합니다.
2. 이 zip을 압축 해제합니다.
3. `miya-ai-v2-main-v10-full-repo` 폴더 안의 파일 전체를 기존 레포에 덮어씁니다.
4. Streamlit Cloud에서 재부팅/재배포합니다.

고객 이름 기능:
- URL query로 `customer_name=홍길동`을 넘기면 바로 이름으로 부를 수 있습니다.
- 또는 `customer_profiles.csv`를 같은 위치에 넣고, `customer_id`, `login_id`, `email` 중 하나를 query로 넘기면 이름 매칭이 가능합니다.