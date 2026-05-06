# 픽톡(PickTalk) 상담 챗봇 클린 배포본 - 고객DB 포함

백업 파일과 미사용 engine 폴더를 제거한 클린 운영 레포입니다.
실제 고객 이름 호출용 customer_profiles.csv를 포함합니다.

## 적용 방법

1. 기존 GitHub 레포 폴더를 백업합니다.
2. 기존 레포 폴더 안 파일을 모두 삭제합니다.
3. 이 zip 안의 pictalk-clean-release-with-customer-20260427 폴더 내부 파일 전체를 기존 레포에 복사합니다.
4. GitHub Desktop에서 commit/push 합니다.
5. Streamlit Cloud에서 재배포합니다.

## 포함 파일

- app.py
- misharp_miya_db.csv
- review_summary.json
- model_profiles.json
- customer_profiles.csv
- customer_profiles_template.csv
- requirements.txt
- logs/
- docs/

## 고객 이름 호출

URL query로 customer_id, login_id, email 중 하나가 전달되면 customer_profiles.csv에서 이름을 매칭합니다.
customer_name=홍길동을 직접 전달해도 됩니다.

## 상담 로그

상담 로그는 logs/pictalk_log_YYYY-MM.csv에 저장됩니다.
