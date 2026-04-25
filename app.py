import os
import re
import json
import csv
import time
import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="픽톡", layout="centered", initial_sidebar_state="collapsed")

# =========================================================
# 기본 설정
# =========================================================
APP_VERSION = "GPT-CENTERED-V1-20260424"
PRODUCT_DB_PATH = "misharp_miya_db.csv"
REVIEW_SUMMARY_PATH = "review_summary.json"
MODEL_PROFILES_PATH = "model_profiles.json"
CUSTOMER_PROFILES_PATH = "customer_profiles.csv"

SIZE_ORDER = {"44": 1, "55": 2, "55반": 3, "66": 4, "66반": 5, "77": 6, "77반": 7, "88": 8, "99": 9}
BODY_CHEST_ESTIMATE = {"44":82, "55":88, "55반":90, "66":94, "66반":97, "77":100, "77반":103, "88":108, "99":114}

TOP_WORDS = ["자켓", "재킷", "점퍼", "코트", "블라우스", "셔츠", "니트", "가디건", "맨투맨", "티셔츠", "후드", "조끼", "베스트"]
BOTTOM_WORDS = ["팬츠", "슬랙스", "바지", "데님", "청바지", "스커트", "치마", "레깅스"]
SHOE_WORDS = ["슈즈", "샌들", "힐", "로퍼", "부츠", "슬링백", "플랫", "스니커즈", "운동화", "신발"]
BAG_WORDS = ["가방", "백", "토트", "크로스백", "숄더백", "클러치"]
ACC_WORDS = ["머플러", "스카프"]
COLOR_WORDS = ["블랙", "아이보리", "베이지", "그레이", "네이비", "화이트", "소라", "브라운", "카키", "핑크", "민트", "레드"]

FORBIDDEN_PHRASES = [
    "상품정보상", "DB 기준", "옵션중", "사이즈 범위 안에 들어와요", "흐름대로", "한 가지만 잡아주시면", "메뉴", "제가 방금 말을 매끄럽게"
]

# =========================================================
# 유틸
# =========================================================
def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_product_no(value) -> str:
    s = clean_text(value)
    return s[:-2] if s.endswith(".0") else s

def normalize_name(name: str) -> str:
    s = clean_text(name)
    s = re.sub(r"\([^)]*color[^)]*\)", " ", s, flags=re.I)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" -|/")

def extract_product_no_from_url(url: str) -> str:
    try:
        qs = parse_qs(urlparse(url).query)
        return normalize_product_no(qs.get("product_no", [""])[0] or qs.get("pn", [""])[0])
    except Exception:
        return ""

def query_params() -> Dict[str, str]:
    try:
        qp = st.query_params
        return {k: clean_text(v[0] if isinstance(v, list) else v) for k, v in qp.items()}
    except Exception:
        return {}

def size_rank(s: str) -> Optional[int]:
    return SIZE_ORDER.get(clean_text(s))

def detect_category(text: str) -> str:
    c = clean_text(text)
    if any(w in c for w in SHOE_WORDS): return "신발"
    if any(w in c for w in BAG_WORDS): return "가방"
    if any(w in c for w in ACC_WORDS): return "악세사리"
    if any(w in c for w in ["블라우스", "블라우스/셔츠"]): return "블라우스"
    if "셔츠" in c and "셔츠 자켓" not in c: return "셔츠"
    if "맨투맨" in c: return "맨투맨"
    if "티셔츠" in c or "티" in c: return "티셔츠"
    if "가디건" in c: return "가디건"
    if "니트" in c: return "니트"
    if any(w in c for w in ["자켓", "재킷", "점퍼", "코트", "조끼", "베스트", "아우터"]): return "자켓"
    if any(w in c for w in ["팬츠", "슬랙스", "바지", "데님", "청바지"]): return "팬츠"
    if any(w in c for w in ["스커트", "치마"]): return "스커트"
    return "기타"

def tokens(text: str) -> List[str]:
    text = re.sub(r"[\[\]()/_\-,.|]+", " ", clean_text(text))
    stop = {"이", "그", "저", "상품", "옷", "비교", "추천", "해줘", "해주세요", "고민", "비슷", "다른", "더", "좋은", "랑", "이랑", "와", "과"}
    return [t for t in text.split() if len(t) >= 2 and t not in stop]

def name_score(query: str, name: str) -> float:
    qts, nts = tokens(query), tokens(name)
    if not qts or not nts: return 0
    score = 0
    ntext = clean_text(name)
    qtext = clean_text(query)
    if qtext and qtext in ntext: score += 8
    for qt in qts:
        for nt in nts:
            if qt == nt: score += 4
            elif qt in nt or nt in qt: score += 1.5
    return score

def safe_postprocess(answer: str, customer_call: str) -> str:
    ans = clean_text(answer)
    for phrase in FORBIDDEN_PHRASES:
        ans = ans.replace(phrase, "")
    ans = ans.replace("고객님", customer_call)
    ans = re.sub(r"(\S+)은\s", lambda m: (m.group(1) + "은 ") if not m.group(1).endswith(("스", "츠", "스커트", "팬츠", "슬랙스")) else (m.group(1) + "는 "), ans)
    # Remove duplicate sentences roughly
    parts = re.split(r"(?<=[.!?요])\s+", ans)
    seen, out = set(), []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            out.append(p); seen.add(p)
    return " ".join(out).strip()

# =========================================================
# 데이터 로드
# =========================================================
@st.cache_data(show_spinner=False)
def load_product_db() -> List[Dict]:
    if not os.path.exists(PRODUCT_DB_PATH):
        return []
    df = pd.read_csv(PRODUCT_DB_PATH)
    df.columns = [clean_text(c) for c in df.columns]
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str).map(clean_text)
    if "product_no" in df.columns:
        df["product_no"] = df["product_no"].map(normalize_product_no)
    return df.to_dict("records")

@st.cache_data(show_spinner=False)
def load_json(path: str):
    if not os.path.exists(path): return {} if path.endswith(".json") else []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def load_customer_profiles() -> List[Dict]:
    if not os.path.exists(CUSTOMER_PROFILES_PATH):
        return []
    try:
        df = pd.read_csv(CUSTOMER_PROFILES_PATH)
        df.columns = [clean_text(c) for c in df.columns]
        for c in df.columns:
            df[c] = df[c].fillna("").astype(str).map(clean_text)
        return df.to_dict("records")
    except Exception:
        return []

DB_ROWS = load_product_db()
REVIEW_SUMMARY = load_json(REVIEW_SUMMARY_PATH)
MODEL_PROFILES = load_json(MODEL_PROFILES_PATH)
CUSTOMER_PROFILES = load_customer_profiles()

# =========================================================
# 상태
# =========================================================
def ensure_state():
    defaults = {
        "messages": [],
        "last_context_key": "",
        "body_height": "",
        "body_weight": "",
        "body_top": "",
        "body_bottom": "",
        "shoe_size": "",
        "last_recommendations": [],
        "selected_product": {},
        "conversation_focus": "",
        "situation_context": "",
        "last_user_intent": "",
        "last_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
ensure_state()

def body_context() -> Dict:
    return {
        "height_cm": clean_text(st.session_state.body_height),
        "weight_kg": clean_text(st.session_state.body_weight),
        "top_size": clean_text(st.session_state.body_top),
        "bottom_size": clean_text(st.session_state.body_bottom),
        "shoe_size": clean_text(st.session_state.shoe_size),
    }

def body_summary() -> str:
    b = body_context()
    return f"키: {b['height_cm'] or '-'}cm, 체중: {b['weight_kg'] or '-'}kg, 상의: {b['top_size'] or '-'}, 하의: {b['bottom_size'] or '-'}, 신발: {b['shoe_size'] or '-'}"

# =========================================================
# 고객명
# =========================================================
def resolve_customer_name() -> str:
    qp = query_params()
    if qp.get("customer_name"):
        return qp["customer_name"]
    for key in ["customer_id", "login_id", "email"]:
        val = qp.get(key, "")
        if not val: continue
        for row in CUSTOMER_PROFILES:
            if clean_text(row.get(key, "")) == val and clean_text(row.get("name", "")):
                return clean_text(row.get("name", ""))
    return ""

def customer_call() -> str:
    name = resolve_customer_name()
    return f"{name}님" if name else "고객님"

# =========================================================
# 현재 상품 컨텍스트
# =========================================================
def get_db_product(product_no: str) -> Optional[Dict]:
    pno = normalize_product_no(product_no)
    if not pno: return None
    for row in DB_ROWS:
        if normalize_product_no(row.get("product_no", "")) == pno:
            return row
    return None

def get_db_product_by_name(name: str) -> Optional[Dict]:
    scored = []
    for row in DB_ROWS:
        sc = name_score(name, row.get("product_name", ""))
        if sc > 0: scored.append((sc, row))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1] if scored else None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_product_context(url: str, product_no: str = "", product_name: str = "") -> Dict:
    pno = normalize_product_no(product_no) or extract_product_no_from_url(url)
    db = get_db_product(pno)
    if db:
        name = normalize_name(db.get("product_name", ""))
        return {
            "product_no": pno,
            "product_name": name or normalize_name(product_name) or "지금 보시는 상품",
            "category": detect_category(f"{db.get('product_name','')} {db.get('category','')} {db.get('sub_category','')}"),
            "summary": clean_text(db.get("product_summary", "")),
            "fit": clean_text(db.get("fit_type", "")),
            "size_range": clean_text(db.get("size_range", "")),
            "colors": clean_text(db.get("color_options", "")),
            "db": db,
            "crawl_text": "",
        }
    name = normalize_name(product_name)
    crawl_text = ""
    if url:
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
            soup = BeautifulSoup(r.text, "html.parser")
            og = soup.select_one('meta[property="og:title"]')
            if not name and og and og.get("content"):
                name = normalize_name(og.get("content"))
            for t in soup(["script", "style", "noscript"]): t.decompose()
            crawl_text = clean_text(soup.get_text(" "))[:2500]
        except Exception:
            pass
    return {"product_no": pno, "product_name": name or "지금 보시는 상품", "category": detect_category(name + " " + crawl_text), "summary":"", "fit":"", "size_range":"", "colors":"", "db": None, "crawl_text": crawl_text}

# =========================================================
# 의도 분석: GPT 중심이지만 라우터는 상담축만 결정
# =========================================================
def detect_intent(user_text: str) -> str:
    q = clean_text(user_text)
    no_space = q.replace(" ", "")
    if no_space in {"응", "네", "넵", "좋아", "그래", "ㅇㅇ", "어"}: return "affirm"
    if re.search(r"[123]번|첫 ?번째|두 ?번째|세 ?번째", q): return "followup_selected"
    if any(k in q for k in ["일자", "부츠컷", "숏", "롱", "타입", "기본형", "와이드형"]): return "option_choice"
    if any(k in q for k in ["장점", "특징", "왜 좋아", "뭐가 좋아", "좋은 점"]): return "feature"
    if any(k in q for k in ["비교", "둘 중", "뭐가 더", "어느 게", "어느게"]): return "compare"
    if any(k in q for k in ["컬러", "색상", "무슨 색", "어떤 색", "블랙", "아이보리", "베이지", "브라운"]): return "color"
    if any(k in q for k in ["어울리는", "같이 입", "코디", "안에 입", "신발", "가방", "머플러"]): return "coordi_recommend"
    if any(k in q for k in ["비슷", "다른", "대신", "더 좋은", "더 나은", "추천"]): return "alternative_recommend"
    if any(k in q for k in ["배송", "교환", "반품", "환불", "출고"]): return "policy"
    if any(k in q for k in ["힙", "허벅지", "골반", "복부", "상체", "가슴", "다리", "짧", "키", "맞", "사이즈", "핏", "작", "클", "여유", "타이트"]): return "fit_size"
    return "general"

# =========================================================
# 후보 검색
# =========================================================
def target_category_from_text(text: str, current_category: str = "") -> str:
    q = clean_text(text)
    if any(k in q for k in ["자켓", "재킷", "아우터", "코트", "점퍼", "가디건"]):
        if "가디건" in q:
            return "니트"
        return "자켓"
    if any(k in q for k in ["블라우스"]): return "블라우스"
    if any(k in q for k in ["셔츠"]): return "셔츠"
    if any(k in q for k in ["니트", "가디건"]): return "니트"
    if any(k in q for k in ["상의", "탑", "티셔츠"]): return "블라우스"
    if any(k in q for k in ["슬랙스", "팬츠", "바지", "데님", "청바지"]): return "팬츠"
    if any(k in q for k in SHOE_WORDS): return "신발"
    if any(k in q for k in BAG_WORDS): return "가방"
    if any(k in q for k in ACC_WORDS): return "악세사리"
    return current_category or "기타"

def size_ok_for_user(row: Dict, target_cat: str) -> bool:
    # Conservative: do not filter too hard; only block clear size miss for clothing
    b = body_context()
    user_size = b["bottom_size"] if target_cat in ["팬츠", "스커트"] else b["top_size"]
    if target_cat == "신발":
        return True
    if not user_size: return True
    lo, hi = parse_size_range(row.get("size_range", ""))
    rank = size_rank(user_size)
    if not rank or not hi: return True
    return rank <= hi

def parse_size_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    text = clean_text(text).replace("~", "-")
    found = []
    for s in ["44", "55반", "55", "66반", "66", "77반", "77", "88", "99"]:
        if s in text: found.append(SIZE_ORDER[s])
    if not found and "FREE" in text.upper(): return SIZE_ORDER["55"], SIZE_ORDER["77"]
    return (min(found), max(found)) if found else (None, None)

def row_category(row: Dict) -> str:
    return detect_category(f"{row.get('category','')} {row.get('sub_category','')} {row.get('product_name','')}")

def row_category_matches(row: Dict, target_cat: str) -> bool:
    cat_blob = clean_text(f"{row.get('category','')} {row.get('sub_category','')} {row.get('product_name','')}")
    cat = row_category(row)
    if not target_cat or target_cat == "기타":
        return True
    if target_cat == "블라우스":
        return cat in ["블라우스", "셔츠"] or any(k in cat_blob for k in ["블라우스", "셔츠", "블라우스/셔츠"])
    if target_cat == "셔츠":
        return cat in ["블라우스", "셔츠"] or "셔츠" in cat_blob
    if target_cat == "니트":
        return cat in ["니트", "가디건"] or any(k in cat_blob for k in ["니트", "가디건"])
    if target_cat == "자켓":
        return cat == "자켓" or any(k in cat_blob for k in ["자켓", "재킷", "점퍼", "코트", "조끼", "베스트", "아우터"])
    if target_cat == "팬츠":
        return cat == "팬츠" or any(k in cat_blob for k in ["팬츠", "슬랙스", "바지", "데님", "청바지"])
    return cat == target_cat or target_cat in cat_blob

def product_reason_from_row(row: Dict, intent: str, user_text: str) -> str:
    name = clean_text(row.get("product_name", ""))
    blob = clean_text(f"{name} {row.get('product_summary','')} {row.get('fit_type','')} {row.get('body_cover_features','')} {row.get('style_tags','')} {row.get('fabric','')}")
    if any(k in user_text for k in ["출근", "회사", "오피스"]):
        if any(k in blob for k in ["셔츠", "블라우스"]):
            return "출근룩에 받쳐 입기 좋은 단정한 분위기예요"
        if any(k in blob for k in ["자켓", "재킷", "아우터"]):
            return "전체 코디를 깔끔하게 잡아주는 출근용 아우터로 좋아요"
    if any(k in blob for k in ["핀턱", "배기", "와이드", "여유"]):
        return "복부나 힙 라인을 너무 드러내지 않고 편하게 정리해줘요"
    if any(k in blob for k in ["셔츠", "블라우스"]):
        return "팬츠와 매치했을 때 단정하고 깔끔하게 이어져요"
    if any(k in blob for k in ["자켓", "재킷"]):
        return "출근룩이나 모임룩에 단정한 외출 느낌을 더해줘요"
    if any(k in blob for k in ["가디건", "니트"]):
        return "부드럽게 걸치기 좋아 과하지 않은 데일리 코디에 좋아요"
    return clean_text(row.get("product_summary", ""))[:60] or "데일리로 무난하게 활용하기 좋아요"

def find_candidates(intent: str, user_text: str, current: Dict, limit: int = 5) -> List[Dict]:
    current_cat = current.get("category", "기타")
    current_no = normalize_product_no(current.get("product_no", ""))
    q = clean_text(user_text)
    if intent == "alternative_recommend":
        target_cat = current_cat if current_cat != "기타" else target_category_from_text(q, current_cat)
    elif intent == "coordi_recommend":
        target_cat = target_category_from_text(q, "팬츠" if current_cat in ["자켓", "블라우스", "셔츠", "니트", "맨투맨", "티셔츠"] else "블라우스")
    else:
        target_cat = target_category_from_text(q, current_cat)

    scored = []
    for row in DB_ROWS:
        pno = normalize_product_no(row.get("product_no", ""))
        if current_no and pno == current_no: continue
        cat = row_category(row)
        if target_cat and target_cat != "기타" and not row_category_matches(row, target_cat):
            continue
        if not size_ok_for_user(row, cat): continue
        sc = 1
        blob = f"{row.get('product_name','')} {row.get('category','')} {row.get('sub_category','')} {row.get('style_tags','')} {row.get('body_cover_features','')} {row.get('product_summary','')} {row.get('fabric','')}"
        for t in tokens(q):
            if t in blob: sc += 1
        if any(k in q for k in ["출근", "학교", "상담", "단정"]):
            if any(k in blob for k in ["단정", "클래식", "슬랙스", "셔츠", "블라우스", "자켓"]): sc += 2
        if any(k in q for k in ["힙", "허벅지", "복부"]):
            if any(k in blob for k in ["커버", "와이드", "배기", "핀턱", "여유"]): sc += 2
        scored.append((sc, row))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:limit]]

def find_compare_target(user_text: str, current: Dict) -> Optional[Dict]:
    q = clean_text(user_text)
    q2 = re.sub(r"(이거|이 옷|이 바지|이 슬랙스|비교해줘|비교|둘 중|뭐가 더|어느 게|더 나아|랑|이랑|와|과)", " ", q)
    q2 = clean_text(q2)
    scored = []
    for row in DB_ROWS:
        if normalize_product_no(row.get("product_no", "")) == normalize_product_no(current.get("product_no", "")): continue
        sc = name_score(q2, row.get("product_name", ""))
        if sc > 0: scored.append((sc, row))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1] if scored else None

# =========================================================
# 리뷰/모델 요약
# =========================================================
def review_for(product_no: str) -> Dict:
    if not product_no: return {}
    return REVIEW_SUMMARY.get(str(product_no), {}) or REVIEW_SUMMARY.get(normalize_product_no(product_no), {}) or {}

def compact_review(product_no: str) -> str:
    r = review_for(product_no)
    if not r: return ""
    parts = []
    if r.get("review_count"): parts.append(f"후기 {r.get('review_count')}건")
    for key in ["positive_keywords", "top_good", "fit_keywords", "negative_keywords", "top_bad"]:
        v = r.get(key)
        if isinstance(v, list) and v:
            # may be list of strings or pairs
            vals = []
            for item in v[:3]:
                vals.append(item[0] if isinstance(item, list) else str(item))
            parts.append("/".join(vals))
            break
    if r.get("summary"): parts.append(clean_text(r.get("summary"))[:160])
    return " | ".join(parts[:3])

def model_hint() -> str:
    if isinstance(MODEL_PROFILES, dict):
        models = MODEL_PROFILES.get("models", [])
    elif isinstance(MODEL_PROFILES, list):
        models = MODEL_PROFILES
    else:
        models = []
    if not models:
        return ""
    vals = []
    for m in models[:2]:
        if isinstance(m, dict):
            h = m.get("height_cm", "")
            w = m.get("weight_kg", "")
            if h and w:
                vals.append(f"{h}cm/{w}kg")
    return "상세페이지 모델컷은 " + " 또는 ".join(vals) + " 체형 기준입니다. 모델 이름은 고객에게 말하지 않습니다." if vals else ""

def openai_client():
    if OpenAI is None: return None
    key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    key = key or os.environ.get("OPENAI_API_KEY", "")
    if not key: return None
    return OpenAI(api_key=key)

def build_system_prompt() -> str:
    return f"""
너는 미샵 쇼핑친구 '미야언니'다. 4050 여성 고객을 옆에서 같이 봐주는 믿음 가는 MD처럼 상담한다.

최우선 원칙:
1. 고객 질문에 바로 답한다. '사이즈/코디/비교/컬러 중 무엇을 볼까요?' 같은 메뉴형 질문을 하지 않는다.
2. 결론을 먼저 말하고, 이유를 짧게 붙인다.
3. 상품DB/리뷰/모델정보는 근거로만 사용한다. 'DB 기준', '상품정보상' 같은 말은 절대 쓰지 않는다.
4. 추천 상품 요청이면 allowed_candidates 안에서 2~3개를 반드시 번호 리스트로 제안한다. 없는 상품명을 만들지 않는다. allowed_candidates가 있으면 '추천 가능한 상품이 없다'고 말하지 않는다.
5. 현재 상품과 선택 상품을 구분한다. '비슷한 다른 상품'은 대체재 추천이고, '어울리는/같이 입을/코디/상의/아우터'는 코디 추천이다.
6. 66/77 같은 권장사이즈와 가슴둘레 실측을 혼동하지 않는다. 권장사이즈는 가능 여부, 실측은 핏 체감 설명에만 쓴다.
7. 고객 이름이 있으면 자연스럽게 이름+님으로 부른다. 없으면 고객님이라고 한다.
8. 모델 이름은 말하지 않는다. 필요하면 '상세페이지 모델컷 기준'으로만 말한다.
9. 반복 문장을 피하고, 3~6문장으로 답한다.
10. 금지 표현: {', '.join(FORBIDDEN_PHRASES)}

답변 톤:
- 친구 같은 MD 말투.
- '입는 건 가능해요', '다만 힙이 있는 편이면', '실패 적게 가시려면', '이쪽이 더 안전해요' 같은 자연스러운 표현을 쓴다.
""".strip()

def build_context_payload(intent: str, user_text: str, current: Dict) -> Dict:
    selected = st.session_state.selected_product or {}
    active = selected if selected.get("product_name") and any(w in user_text for w in ["그", "2번", "3번", "선택", "안에"]) else current
    candidates = []
    compare_target = None

    if intent in ["alternative_recommend", "coordi_recommend"]:
        candidates = find_candidates(intent, user_text, active, limit=5)
        st.session_state.last_recommendations = candidates[:3]
    elif intent == "compare":
        compare_target = find_compare_target(user_text, active)
        if compare_target:
            candidates = [compare_target]
        else:
            candidates = find_candidates("alternative_recommend", user_text, active, limit=3)

    product_db = active.get("db") or get_db_product(active.get("product_no", "")) or {}
    allowed = []
    for row in candidates[:5]:
        allowed.append({
            "product_no": row.get("product_no", ""),
            "product_name": row.get("product_name", ""),
            "category": row_category(row),
            "size_range": row.get("size_range", ""),
            "fit_type": row.get("fit_type", ""),
            "summary": row.get("product_summary", "")[:220],
            "body_cover": row.get("body_cover_features", "")[:160],
            "reason": product_reason_from_row(row, intent, user_text),
            "review": compact_review(row.get("product_no", "")),
        })
    return {
        "app_version": APP_VERSION,
        "intent": intent,
        "customer_call": customer_call(),
        "customer_body": body_context(),
        "current_product": {
            "product_no": current.get("product_no", ""),
            "product_name": current.get("product_name", ""),
            "category": current.get("category", ""),
            "size_range": current.get("size_range", "") or product_db.get("size_range", ""),
            "fit": current.get("fit", "") or product_db.get("fit_type", ""),
            "summary": current.get("summary", "")[:450] or product_db.get("product_summary", "")[:450],
            "colors": current.get("colors", "") or product_db.get("color_options", ""),
            "measurements": {
                "shoulder": product_db.get("shoulder", ""),
                "chest": product_db.get("chest", ""),
                "waist": product_db.get("waist", ""),
                "hip": product_db.get("hip", ""),
                "rise": product_db.get("rise", ""),
                "thigh": product_db.get("thigh", ""),
                "hem": product_db.get("hem", ""),
                "length": product_db.get("length", ""),
                "raw_measurements": product_db.get("raw_measurements", ""),
            },
            "review": compact_review(current.get("product_no", "")),
        },
        "selected_product": selected,
        "allowed_candidates": allowed,
        "model_hint": model_hint(),
        "recent_messages": st.session_state.messages[-8:],
        "situation_context": st.session_state.situation_context,
    }


# =========================================================
# 빠른 응답: 자주 나오는 사이즈/체형 질문은 GPT 호출 전에 즉시 처리
# =========================================================
def particle_eun_neun(name: str) -> str:
    name = clean_text(name) or "상품"
    last = name[-1]
    try:
        code = ord(last) - ord("가")
        has_jong = 0 <= code <= 11171 and (code % 28) != 0
        return f"{name}{'은' if has_jong else '는'}"
    except Exception:
        return f"{name}은"

def fast_size_option_answer(user_text: str, current: Dict) -> str:
    q = clean_text(user_text)
    q_low = q.lower()

    # M/L, 미디움/라지, 사이즈 선택 질문
    size_option_words = ["m", "l", "라지", "미디움", "medium", "large", "사이즈", "뭘로", "고르면", "선택", "나을"]
    if not any(w in q_low for w in size_option_words):
        return ""

    name = current.get("product_name") or "지금 보시는 상품"
    cat_text = f"{current.get('category','')} {name}"
    is_bottom = any(k in cat_text for k in ["슬랙스", "팬츠", "바지", "데님", "스커트", "치마"])
    base_size = st.session_state.get("body_bottom", "") if is_bottom else st.session_state.get("body_top", "")
    has_hip = any(k in q for k in ["힙", "골반", "허벅지", "하체"])

    # 66반 하의 고객이 M/L을 묻는 가장 빈번한 케이스
    if base_size == "66반" or "66반" in q:
        if has_hip:
            return f"{base_size or '66반'}에 힙이 있는 편이시면 {particle_eun_neun(name)} M보다는 L 쪽이 더 안전해요. 힙이나 허벅지에서 당기는 느낌이 나면 전체 실루엣이 덜 예뻐질 수 있거든요. 허리는 조금 여유 있을 수 있지만, 편하게 입고 실패 줄이려면 L을 먼저 추천드릴게요."
        return f"{base_size or '66반'} 기준이면 {particle_eun_neun(name)} M은 깔끔하게 맞는 쪽, L은 조금 더 편한 쪽으로 보시면 좋아요. 편하게 입으실 거면 L, 딱 떨어지는 핏을 원하시면 M 쪽이에요."

    if base_size in ["77", "77반", "88"]:
        return f"{base_size} 기준이면 {particle_eun_neun(name)} 가능 옵션 중에서는 큰 쪽을 먼저 보시는 게 안전해요. 특히 힙이나 허벅지가 있는 편이면 작은 사이즈는 앉거나 움직일 때 답답하게 느껴질 수 있어요."

    if base_size in ["55", "55반", "66"]:
        if has_hip:
            return f"{base_size} 기준이면 {particle_eun_neun(name)} 기본 사이즈도 가능해 보이지만, 힙이나 허벅지가 신경 쓰이면 한 사이즈 여유 있게 보시는 게 더 편해요."
        return f"{base_size} 기준이면 {particle_eun_neun(name)} 기본 사이즈 쪽부터 보셔도 괜찮아요. 다만 편한 핏을 원하시면 한 사이즈 여유 있게 보는 쪽도 좋습니다."

    return ""

def fast_body_fit_answer(user_text: str, current: Dict) -> str:
    q = clean_text(user_text)
    if not any(k in q for k in ["힙", "골반", "허벅지", "하체", "다리 짧", "다리가 짧", "키가 작"]):
        return ""

    name = current.get("product_name") or "지금 보시는 상품"
    bottom = st.session_state.get("body_bottom", "")

    if any(k in q for k in ["힙", "골반", "허벅지", "하체"]):
        if bottom in ["66반", "77", "77반", "88"]:
            return f"{bottom}에 힙이 있는 편이시면 {particle_eun_neun(name)} 사이즈 선택을 조금 여유 있게 보시는 게 좋아요. 너무 딱 맞게 고르면 힙 쪽이 먼저 당겨 보여서 핏이 덜 예쁠 수 있거든요. 편하게 예쁜 실루엣을 원하시면 가능한 옵션 중 큰 쪽을 먼저 추천드릴게요."
        return f"{particle_eun_neun(name)} 힙 라인이 신경 쓰이는 분들은 너무 딱 맞게보다 살짝 여유 있게 고르는 쪽이 예뻐요. 하의 사이즈와 옵션을 같이 보면 더 정확하게 잡아드릴 수 있어요."

    if any(k in q for k in ["다리 짧", "다리가 짧", "키가 작"]):
        return f"{particle_eun_neun(name)} 다리 비율이 걱정되시면 길이와 밑단 라인이 중요해요. 너무 애매하게 끊기는 기장보다 발등 가까이 자연스럽게 떨어지는 쪽이 다리가 더 길어 보여요. 옵션이 있다면 전체 비율이 깔끔하게 이어지는 쪽을 먼저 보시는 게 좋아요."

    return ""

def fast_answer(user_text: str, current: Dict) -> str:
    for fn in (fast_size_option_answer, fast_body_fit_answer):
        try:
            ans = fn(user_text, current)
            if ans:
                return safe_postprocess(ans, customer_call())
        except Exception:
            pass
    return ""


def call_gpt(user_text: str, current: Dict) -> Optional[str]:
    client = openai_client()
    if client is None: return None
    intent = detect_intent(user_text)
    payload = build_context_payload(intent, user_text, current)
    try:
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role":"system", "content": build_system_prompt()},
                    {"role":"user", "content": json.dumps({"user_text": user_text, "context": payload}, ensure_ascii=False)}
                ],
                temperature=0.3,
                max_tokens=520,
            )
        except Exception:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role":"system", "content": build_system_prompt()},
                    {"role":"user", "content": json.dumps({"user_text": user_text, "context": payload}, ensure_ascii=False)}
                ],
                max_tokens=520,
            )
        answer = resp.choices[0].message.content.strip()
        return safe_postprocess(answer, payload["customer_call"])
    except Exception as e:
        st.session_state.last_error = str(e)
        return None

# =========================================================
# GPT 미사용 시 안전 fallback
# =========================================================
def fallback_answer(user_text: str, current: Dict) -> str:
    intent = detect_intent(user_text)
    call = customer_call()
    name = current.get("product_name", "지금 보시는 상품")
    cat = current.get("category", "")
    b = body_context()
    if intent == "fit_size":
        if any(k in user_text for k in ["힙", "허벅지", "골반"]):
            return f"{call}, {name}은 힙이나 허벅지가 있는 편이면 너무 딱 맞게 보기보다는 여유감을 먼저 보는 게 좋아요. 현재 입력 사이즈 기준으로는 입는 것 자체보다 힙 라인이 얼마나 편하게 떨어지는지가 중요해서, 편한 핏을 원하시면 한 단계 여유 있는 슬랙스도 같이 보시는 게 안전해요."
        if any(k in user_text for k in ["다리", "짧", "키"]):
            return f"{call}, 다리 비율이 걱정되시면 {name}은 기장과 밑단 라인을 같이 보시는 게 중요해요. 실패 적게 가시려면 너무 퍼지는 핏보다 일자로 깔끔하게 떨어지는 쪽이 더 안정적이에요."
        return f"{call}, {name}은 현재 입력하신 사이즈 기준으로 먼저 가능 여부를 보고, 그다음 핏 체감을 봐야 해요. 편하게 입고 싶으시면 실측 여유와 후기 반응까지 같이 보는 쪽이 안전해요."
    if intent == "feature":
        return f"{call}, {name}의 장점은 데일리로 입기 부담 없는 안정적인 핏이에요. 너무 과하게 멋낸 느낌보다 깔끔하게 정리되는 쪽이라 출근이나 일상 코디에 활용하기 좋아요."
    if intent == "option_choice":
        if "일자" in user_text and "부츠컷" in user_text:
            return f"{call}, 다리가 짧게 느껴지는 편이면 일자 쪽이 더 안전해요. 부츠컷은 예쁘지만 기장 영향을 더 받아서 비율이 민감할 수 있어요. 실패 적게 가시려면 일자 먼저 추천드려요."
        if "숏" in user_text and "롱" in user_text:
            return f"{call}, 키가 작거나 다리 비율이 걱정되시면 롱을 무조건 고르기보다 신발과 기장을 같이 봐야 해요. 발등을 살짝 덮는 정도면 길어 보이고, 끌리면 오히려 답답해 보여요."
    if intent == "alternative_recommend":
        cands = find_candidates("alternative_recommend", user_text, current, limit=3)
        if cands:
            st.session_state.last_recommendations = cands
            lines = [f"{call}, 지금 보시는 {name}과 비슷한 무드에서 대안으로 볼 만한 상품을 골라드릴게요."]
            for i, row in enumerate(cands, 1):
                lines.append(f"{i}. {row.get('product_name')} — {row.get('size_range','')} / {product_reason_from_row(row, intent, user_text)}")
            return "\n".join(lines)
        return f"{call}, 비슷한 대안 상품을 바로 많이 잡지는 못했어요. 그래도 같은 카테고리 안에서 더 여유 있는 쪽으로 다시 골라드릴게요."
    if intent == "coordi_recommend":
        cands = find_candidates("coordi_recommend", user_text, current, limit=3)
        if cands:
            st.session_state.last_recommendations = cands
            lines = [f"{call}, {name} 기준으로 같이 입기 좋은 쪽으로 골라드릴게요."]
            for i, row in enumerate(cands, 1):
                lines.append(f"{i}. {row.get('product_name')} — {row.get('size_range','')} / {product_reason_from_row(row, intent, user_text)}")
            return "\n".join(lines)
        return f"{call}, {name}에는 너무 캐주얼한 것보다 깔끔한 슬랙스나 블라우스 계열이 안정적이에요."
    if intent == "color":
        return f"{call}, 단정하게 입으실 거면 블랙·아이보리·베이지처럼 차분한 컬러가 가장 안전해요. 상체가 도드라져 보이는 게 걱정이면 너무 강한 색보다 차분한 톤을 추천드려요."
    if intent == "compare":
        return f"{call}, 비교는 현재 보고 계신 {name}을 기준으로 사이즈 안정감, 핏, 활용도를 나눠서 보는 게 좋아요. 비교할 상품명을 조금만 더 알려주시면 어느 쪽이 더 나은지 결론까지 같이 말씀드릴게요."
    return f"{call}, 지금 질문은 {name} 기준으로 같이 볼게요. 입었을 때 느낌과 실제 활용도를 중심으로 바로 상담드릴게요."

# =========================================================
# 고객 선택 처리
# =========================================================
def maybe_update_selected(user_text: str):
    m = re.search(r"([123])번", user_text)
    if not m: return
    idx = int(m.group(1)) - 1
    recs = st.session_state.last_recommendations or []
    if 0 <= idx < len(recs):
        row = recs[idx]
        st.session_state.selected_product = {
            "product_no": row.get("product_no", ""),
            "product_name": row.get("product_name", ""),
            "category": row_category(row),
            "size_range": row.get("size_range", ""),
            "fit": row.get("fit_type", ""),
        }

# =========================================================
# UI
# =========================================================
st.markdown("""
<style>
[data-testid="stToolbar"]{visibility:hidden;height:0;position:fixed;}
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
.block-container{max-width:760px;padding-top:54px;padding-left:18px;padding-right:18px;padding-bottom:90px;}
.miya-title-wrap{display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:4px;text-align:center;}
.miya-title{font-size:30px;font-weight:900;letter-spacing:-1.2px;line-height:1.15;color:#1f2937;text-align:center;}
.miya-title .accent{color:#0f766e;}
.beta-badge{font-size:11px;background:#0f766e;color:#fff;border-radius:999px;padding:3px 8px;font-weight:700;}
.miya-sub{color:#666;font-size:14px;margin-bottom:18px;text-align:center;}
.chat-user{background:#e5f4ef;color:#12423a;border:1px solid #c6ded8;border-radius:18px 18px 4px 18px;padding:12px 14px;margin:8px 0 8px auto;max-width:82%;line-height:1.55;}
.chat-bot{background:#08245a;color:#fff;border-radius:18px 18px 18px 4px;padding:13px 15px;margin:8px auto 8px 0;max-width:86%;line-height:1.58;}
.label{font-size:12px;color:#666;margin:8px 0 3px;font-weight:700;}
.stTextInput input, .stSelectbox div[data-baseweb="select"]{border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="miya-title-wrap"><div class="miya-title">미샵 쇼핑친구 <span class="accent">픽톡</span></div><span class="beta-badge">BETA</span></div>', unsafe_allow_html=True)
st.markdown('<div class="miya-sub">24시간 쇼핑 결정에 도움드리는 스마트한 쇼핑친구</div>', unsafe_allow_html=True)

qp = query_params()
url = qp.get("url", "") or qp.get("product_url", "")
product_no = qp.get("product_no", "")
product_name = qp.get("product_name", "")
current = fetch_product_context(url, product_no, product_name)
context_key = f"{current.get('product_no','')}|{current.get('product_name','')}"
if context_key != st.session_state.last_context_key:
    st.session_state.last_context_key = context_key
    st.session_state.messages = []
    st.session_state.last_recommendations = []
    st.session_state.selected_product = {}

with st.expander("사이즈 입력 (더 구체적인 상담 가능)", expanded=True):
    size_options = ["", "44", "55", "55반", "66", "66반", "77", "77반", "88", "99"]
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.body_height = st.text_input("키", value=st.session_state.body_height, placeholder="cm")
    with c2:
        st.session_state.body_weight = st.text_input("체중", value=st.session_state.body_weight, placeholder="kg")
    c3, c4 = st.columns(2)
    with c3:
        current_top = st.session_state.body_top if st.session_state.body_top in size_options else ""
        st.session_state.body_top = st.selectbox("상의", size_options, index=size_options.index(current_top), format_func=lambda x: x or "선택")
    with c4:
        current_bottom = st.session_state.body_bottom if st.session_state.body_bottom in size_options else ""
        st.session_state.body_bottom = st.selectbox("하의", size_options, index=size_options.index(current_bottom), format_func=lambda x: x or "선택")
    st.session_state.shoe_size = st.text_input("신발사이즈(선택)", value=st.session_state.shoe_size, placeholder="예: 235")

st.caption(f"현재 입력 정보: {body_summary()}")

if not st.session_state.messages:
    call = customer_call()
    st.session_state.messages.append({"role":"assistant", "content":f"안녕하세요 :) {current.get('product_name','지금 보시는 상품')} 같이 봐드릴게요. 사이즈나 코디 고민 편하게 말씀해 주세요."})

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="label" style="text-align:right;">{customer_call()}</div><div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="label">미야언니</div><div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)

user_input = st.chat_input("궁금한 점을 입력하세요")
if user_input:
    maybe_update_selected(user_input)
    st.session_state.messages.append({"role":"user", "content":user_input})
    answer = fast_answer(user_input, current)
    if not answer:
        answer = call_gpt(user_input, current)
    if not answer or len(clean_text(answer)) < 10:
        answer = fallback_answer(user_input, current)
    answer = safe_postprocess(answer, customer_call())
    st.session_state.messages.append({"role":"assistant", "content":answer})
    st.rerun()
