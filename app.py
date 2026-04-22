
import os
import re
import csv
import json
import html
import time
import datetime
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs

import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="미야언니", layout="centered", initial_sidebar_state="collapsed")

# =========================================================
# 상태
# =========================================================
def ensure_state() -> None:
    defaults = {
        "messages": [],
        "last_context_key": "",
        "body_height": "",
        "body_weight": "",
        "body_top": "",
        "body_bottom": "",
        "shoe_size": "",
        "last_answer": "",
        "last_recommendations": [],
        "last_selected_index": None,
        "active_product_override": {},
        "pending_target_category": "",
        "pending_situation": "",
        "pending_style": "",
        "last_compare_candidates": [],
        "customer_name": "",
        "customer_id": "",
        "customer_login_id": "",
        "customer_email": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_state()

# =========================================================
# 공통 유틸
# =========================================================
SIZE_ORDER = {"44": 1, "55": 2, "55반": 3, "66": 4, "66반": 5, "77": 6, "77반": 7, "88": 8, "99": 9}
SIZE_LABELS = {v: k for k, v in SIZE_ORDER.items()}
TOP_KEYWORDS = ["자켓", "재킷", "점퍼", "코트", "블라우스", "셔츠", "니트", "가디건", "맨투맨", "티셔츠", "후드", "조끼", "베스트"]
BOTTOM_KEYWORDS = ["팬츠", "슬랙스", "바지", "데님", "청바지", "스커트", "치마", "레깅스"]
SHOE_KEYWORDS = ["슈즈", "샌들", "힐", "로퍼", "부츠", "슬링백", "플랫", "스니커즈", "운동화", "신발"]
BAG_KEYWORDS = ["가방", "백", "토트", "크로스백", "숄더백", "클러치"]
ACCESSORY_KEYWORDS = ["머플러", "스카프"]

BODY_CHEST_ESTIMATE = {
    "44": 82.0, "55": 88.0, "55반": 90.0, "66": 94.0, "66반": 97.0,
    "77": 100.0, "77반": 103.0, "88": 108.0, "99": 114.0,
}

STOP_TOKENS = set(["이", "그", "저", "상품", "옷", "랑", "이랑", "하고", "비교", "추천", "해주세요", "해줘", "고민", "되는데", "중", "뭐", "가", "더"])

def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_product_no(value) -> str:
    text = clean_text(value)
    return text[:-2] if text.endswith(".0") else text

def size_rank(token: str) -> Optional[int]:
    return SIZE_ORDER.get(clean_text(token))

def rank_to_size(rank: Optional[int]) -> str:
    return SIZE_LABELS.get(rank, "") if rank else ""

def build_body_context() -> Dict[str, str]:
    return {
        "height_cm": clean_text(st.session_state.get("body_height", "")),
        "weight_kg": clean_text(st.session_state.get("body_weight", "")),
        "top_size": clean_text(st.session_state.get("body_top", "")),
        "bottom_size": clean_text(st.session_state.get("body_bottom", "")),
        "shoe_size": clean_text(st.session_state.get("shoe_size", "")),
    }

def body_summary_text() -> str:
    vals = build_body_context()
    if not any(vals.values()):
        return "입력된 체형 정보 없음"
    return (
        f"키: {vals.get('height_cm') or '-'}cm, "
        f"체중: {vals.get('weight_kg') or '-'}kg, "
        f"상의: {vals.get('top_size') or '-'}, "
        f"하의: {vals.get('bottom_size') or '-'}, "
        f"신발: {vals.get('shoe_size') or '-'}"
    )

def extract_product_no_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        return normalize_product_no(qs.get("product_no", [""])[0] or qs.get("pn", [""])[0])
    except Exception:
        return ""

def sanitize_product_name(name: str) -> str:
    text = clean_text(name)
    for piece in ["LOGIN", "JOIN", "MY PAGE", "MYPAGE", "CART", "ABOUT", "SHOP", "COMMUNITY", "TIME SALE", "KRW", "미샵", "MISHARP", "{#item", "{#html", "기본 정보", "상품명"]:
        text = text.replace(piece, " ")
    text = re.sub(r"\([^)]*color[^)]*\)", " ", text, flags=re.I)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|/>")
    return text

def detect_category_from_name(name: str, raw_text: str = "") -> str:
    corpus = f"{clean_text(name)} {clean_text(raw_text)}"
    if "니트티" in corpus or "니트 티" in corpus:
        return "니트티"
    if "블라우스" in corpus:
        return "블라우스"
    if "셔츠" in corpus and "셔츠 자켓" not in corpus:
        return "셔츠"
    if "맨투맨" in corpus:
        return "맨투맨"
    if "티셔츠" in corpus:
        return "티셔츠"
    if "니트" in corpus or "가디건" in corpus:
        return "니트"
    if any(k in corpus for k in ["자켓", "재킷", "점퍼", "코트", "베스트", "조끼"]):
        return "자켓"
    if any(k in corpus for k in SHOE_KEYWORDS):
        return "신발"
    if any(k in corpus for k in BAG_KEYWORDS):
        return "가방"
    if any(k in corpus for k in ACCESSORY_KEYWORDS):
        return "악세사리"
    if any(k in corpus for k in ["팬츠", "슬랙스", "바지", "데님", "청바지"]):
        return "팬츠"
    if "스커트" in corpus or "치마" in corpus:
        return "스커트"
    return "기타"

def normalize_name_tokens(name: str) -> List[str]:
    text = clean_text(name)
    text = re.sub(r"[\(\)\[\]/,_\-]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    parts = [p for p in re.split(r"\s+", text) if p and p not in STOP_TOKENS]
    return parts

def token_overlap_score(query: str, name: str) -> float:
    q_tokens = normalize_name_tokens(query)
    n_tokens = normalize_name_tokens(name)
    if not q_tokens or not n_tokens:
        return 0.0
    overlap = len(set(q_tokens) & set(n_tokens))
    contains = 1 if clean_text(query) and clean_text(query) in clean_text(name) else 0
    prefix = sum(1 for qt in q_tokens if any(nt.startswith(qt) or qt.startswith(nt) for nt in n_tokens))
    return overlap * 2 + contains * 3 + prefix * 0.5

def extract_situation_tag(user_text: str) -> str:
    q = clean_text(user_text)
    mapping = {
        "학교": ["학교", "학교상담", "학교 상담", "선생님", "학부모"],
        "출근": ["출근", "오피스", "회사"],
        "모임": ["모임", "약속", "만남"],
        "하객": ["하객", "결혼식"],
    }
    for tag, words in mapping.items():
        if any(w in q for w in words):
            return tag
    return ""

def extract_style_tag(user_text: str) -> str:
    q = clean_text(user_text)
    if any(k in q for k in ["깔끔", "단정", "무난"]):
        return "단정"
    if any(k in q for k in ["편한", "편하게", "캐주얼"]):
        return "편안"
    if any(k in q for k in ["여성", "우아"]):
        return "여성스러움"
    return ""

# =========================================================
# 보조 데이터 로드
# =========================================================
@st.cache_data(show_spinner=False)
def load_model_profiles() -> List[Dict]:
    path = "model_profiles.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_review_summary() -> Dict:
    path = "review_summary.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

MODEL_PROFILES = load_model_profiles()
REVIEW_SUMMARY = load_review_summary()

@st.cache_data(show_spinner=False)
def load_customer_profiles() -> List[Dict]:
    path = "customer_profiles.csv"
    if not os.path.exists(path):
        return []
    try:
        import pandas as pd
        df = pd.read_csv(path)
        df.columns = [clean_text(c) for c in df.columns]
        for c in df.columns:
            df[c] = df[c].fillna("").astype(str).map(clean_text)
        return df.to_dict(orient="records")
    except Exception:
        return []

CUSTOMER_PROFILES = load_customer_profiles()

def resolve_customer_name(params: Dict) -> str:
    direct = clean_text(params.get("customer_name", ""))
    if direct:
        return direct
    cid = clean_text(params.get("customer_id", ""))
    login_id = clean_text(params.get("login_id", ""))
    email = clean_text(params.get("email", ""))
    if not CUSTOMER_PROFILES:
        return ""
    for row in CUSTOMER_PROFILES:
        if cid and clean_text(row.get("customer_id", "")) == cid:
            return clean_text(row.get("name", ""))
        if login_id and clean_text(row.get("login_id", "")) == login_id:
            return clean_text(row.get("name", ""))
        if email and clean_text(row.get("email", "")) == email:
            return clean_text(row.get("name", ""))
    return ""

def customer_call_name() -> str:
    name = clean_text(st.session_state.get("customer_name", ""))
    return f"{name}님" if name else "고객님"

def personalize_answer(text: str) -> str:
    t = clean_text(text)
    name = customer_call_name()
    if not t:
        return t
    return t.replace("고객님", name)

# =========================================================
# 로그
# =========================================================
def ensure_logs_dir() -> str:
    path = "logs"
    os.makedirs(path, exist_ok=True)
    return path

def write_chat_log(event_type: str, user_text: str = "", answer: str = "", response_mode: str = "", fallback_reason: str = "", error_text: str = "", latency_ms: int = 0, product_context: Optional[Dict] = None) -> None:
    try:
        log_dir = ensure_logs_dir()
        log_path = os.path.join(log_dir, f"chat_log_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            "session_id": st.session_state.get("last_context_key", ""),
            "product_no": clean_text((product_context or {}).get("product_no", "")),
            "product_name": clean_text((product_context or {}).get("product_name", "")),
            "user_text": clean_text(user_text),
            "response_mode": response_mode,
            "fallback_reason": fallback_reason,
            "is_fallback": "1" if response_mode in {"fallback", "rule_fallback"} else "0",
            "error_text": clean_text(error_text),
            "latency_ms": str(latency_ms),
            "answer": clean_text(answer),
        }
        exists = os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass

# =========================================================
# DB
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_product_db():
    path = "misharp_miya_db.csv"
    if not os.path.exists(path):
        return []
    import pandas as pd
    df = pd.read_csv(path)
    df.columns = [clean_text(c) for c in df.columns]
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str).map(clean_text)
    if "product_no" in df.columns:
        df["product_no"] = df["product_no"].map(normalize_product_no)
    return df.to_dict(orient="records")

DB_ROWS = load_product_db()

def get_db_product(product_no_value: str) -> Optional[Dict]:
    target = normalize_product_no(product_no_value)
    if not target:
        return None
    for row in DB_ROWS:
        if normalize_product_no(row.get("product_no", "")) == target:
            return row
    return None

# =========================================================
# 크롤링 컨텍스트
# =========================================================
def extract_meta_name(soup: BeautifulSoup) -> str:
    candidates = []
    for selector in ['meta[property="og:title"]', 'meta[name="og:title"]', 'meta[property="twitter:title"]', 'meta[name="title"]']:
        tag = soup.select_one(selector)
        if tag and tag.get("content"):
            candidates.append(tag.get("content"))
    if soup.title and soup.title.text:
        candidates.append(soup.title.text)
    for c in candidates:
        s = sanitize_product_name(c)
        if s:
            return s
    return ""

def split_detail_sections(text: str) -> Dict[str, str]:
    t = clean_text(text)
    if not t:
        return {"summary": "", "material": "", "fit": "", "size_tip": ""}
    material, fit, size_tip = [], [], []
    for sentence in re.split(r"(?<=[.!?])\s+|\s*/\s*", t):
        s = clean_text(sentence)
        if not s:
            continue
        if any(k in s for k in ["면", "코튼", "폴리", "레이온", "울", "아크릴", "스판", "나일론", "혼용", "%", "소재", "원단"]):
            material.append(s)
        if any(k in s for k in ["핏", "루즈", "정핏", "와이드", "커버", "복부", "허벅지", "힙", "라인", "여유", "벌룬", "루즈핏", "슬림"]):
            fit.append(s)
        if any(k in s for k in ["사이즈", "추천", "44", "55", "66", "77", "88", "FREE", "free"]):
            size_tip.append(s)
    return {"summary": t[:1400], "material": " / ".join(material)[:350], "fit": " / ".join(fit)[:350], "size_tip": " / ".join(size_tip)[:350]}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_product_context(url: str, passed_name: str = "", passed_product_no: str = "") -> Dict:
    safe_name = sanitize_product_name(passed_name)
    safe_no = normalize_product_no(passed_product_no) or extract_product_no_from_url(url)
    fallback_ctx = {"product_no": safe_no, "product_name": safe_name or "지금 보시는 상품", "category": "기타", "summary": "", "material": "", "fit": "", "size_tip": "", "raw_excerpt": ""}
    if not url:
        return fallback_ctx
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
    except Exception:
        return fallback_ctx
    soup = BeautifulSoup(r.text, "html.parser")
    product_name = safe_name or extract_meta_name(soup)
    for t in soup(["script", "style", "noscript", "header", "footer"]):
        t.decompose()
    raw_text = clean_text(re.sub(r"\n{2,}", "\n", soup.get_text("\n").replace("\r", "\n")))
    sections = split_detail_sections(raw_text)
    db_row = get_db_product(safe_no)
    if db_row and clean_text(db_row.get("product_name")):
        product_name = clean_text(db_row.get("product_name"))
    if not product_name:
        product_name = "지금 보시는 상품"
    category = detect_category_from_name(product_name, raw_text)
    return {
        "product_no": safe_no,
        "product_name": product_name,
        "category": category,
        "summary": sections["summary"],
        "material": sections["material"],
        "fit": sections["fit"],
        "size_tip": sections["size_tip"],
        "raw_excerpt": raw_text[:3000],
    }

# =========================================================
# 질문 판별
# =========================================================
def is_affirmative(user_text: str) -> bool:
    q = clean_text(user_text).replace(" ", "")
    return q in {"응", "네", "넹", "ㅇㅇ", "그래", "좋아", "웅", "어"}

def is_pure_greeting(user_text: str) -> bool:
    q = clean_text(user_text).replace(" ", "")
    return q in {"안녕", "안녕하세요", "하이", "반가워", "헬로"}

def is_size_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["사이즈", "맞을까", "맞을까요", "맞겠나", "맞나", "맞아", "핏", "작을까", "클까", "여유", "타이트", "가슴"])

def is_compare_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["비교", "뭐가 더", "어느 게", "둘 중", "고민되", "더 나아"])

def is_color_question(user_text: str) -> bool:
    q = clean_text(user_text)
    if is_compare_question(q):
        return False
    return any(k in q for k in ["컬러", "색", "무슨 색", "어떤 색", "색상"])

def is_name_question(user_text: str) -> bool:
    q = clean_text(user_text).replace(" ", "")
    return any(k in q for k in ["이옷이름", "상품명", "상품이름", "이름뭐", "이옷이뭐야", "품명"])

def is_coordi_request(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["코디", "학교방문", "학교 방문", "행사룩", "모임룩", "학부모", "뭐 입", "같이 입"])

def is_detail_request(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["전체적으로", "자세히", "설명", "얘기해줘", "좀 더", "어때", "괜찮아", "어울려"])

def is_recommendation_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["추천", "골라", "찾아", "어울리는", "같이 입", "코디", "매치", "다른 상품", "다른 자켓", "다른 바지", "다른 블라우스", "보여줘"])

def is_fit_question(user_text: str) -> bool:
    q = clean_text(user_text)
    body_terms = ["상체", "하체", "힙", "허벅지", "골반", "복부", "배", "다리", "어깨", "가슴", "키가 작", "다리가 짧"]
    return any(k in q for k in body_terms) and any(k in q for k in ["괜찮", "맞", "어울", "예쁘게", "부해", "짧아", "길어", "커버"])

def is_feature_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["장점", "특징", "뭐가 좋아", "왜 좋아", "포인트", "어떤 점이 좋아"])

def is_option_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return (("일자" in q and "부츠컷" in q) or ("숏" in q and "롱" in q) or ("와이드" in q and "부츠컷" in q) or ("타입" in q and any(n in q for n in ["1", "2", "3", "첫", "둘", "셋"])))

def build_option_choice_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    q = clean_text(user_text)
    pname = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")
    if "일자" in q and "부츠컷" in q:
        if any(k in q for k in ["다리가 짧", "키가 작", "비율"]):
            return f"{pname} 안에서 고르신다면, 다리가 짧게 느껴지는 편이면 일자 쪽이 더 안정적이에요. 부츠컷은 분위기는 예쁘지만 밑단 퍼짐과 기장 영향이 있어서 비율이 더 민감하게 보일 수 있거든요. 실패 적게 가시려면 일자, 다리 길어 보이는 분위기를 살리고 싶으면 발등 덮는 길이의 부츠컷도 가능해요."
        return f"{pname}은 일자는 깔끔하고 단정하게, 부츠컷은 다리 라인을 길어 보이게 살리는 쪽으로 보시면 돼요. 무난하게 자주 입으실 거면 일자, 여성스럽고 비율감을 조금 더 살리고 싶으면 부츠컷 쪽이 좋아요."
    if "숏" in q and "롱" in q:
        if any(k in q for k in ["키가 작", "다리가 짧", "비율"]):
            return f"{pname}은 키가 작은 편이면 숏 쪽이 더 안정적일 가능성이 커요. 롱은 분위기는 예쁘지만 기장이 길면 비율이 무거워 보일 수 있어서요. 깔끔하게 떨어지게 입으시려면 숏 쪽부터 보시는 게 안전해요."
        return f"{pname}은 숏은 산뜻하고 깔끔한 쪽, 롱은 분위기를 더 길게 살리는 쪽이에요. 평소 신발 높이랑 원하시는 분위기에 따라 고르시면 돼요."
    return f"{pname}은 타입마다 실루엣 차이가 있어서, 원하시는 분위기나 체형 기준을 한 가지만 더 알려주시면 더 정확하게 골라드릴게요 :)"

def build_feature_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    pname = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")
    cat = detect_category_from_name(pname + ' ' + clean_text((db_product or {}).get('category','')), clean_text(product_context.get('summary','')))
    corpus = ' '.join([clean_text(product_context.get('summary','')), clean_text(product_context.get('fit','')), clean_text((db_product or {}).get('product_summary','')), clean_text((db_product or {}).get('fit_type','')), clean_text((db_product or {}).get('body_cover_features',''))])
    parts = [f"{pname}의 가장 큰 장점은"]
    if cat in {'팬츠','스커트'}:
        if any(k in corpus for k in ['와이드','세미와이드']):
            parts.append('하체 라인을 너무 부각하지 않으면서 전체 실루엣을 정리해준다는 점이에요.')
        elif '부츠컷' in corpus:
            parts.append('다리 라인을 길어 보이게 정리해주는 분위기가 있다는 점이에요.')
        elif any(k in corpus for k in ['핀턱','턱']):
            parts.append('앞라인이 정리돼 보여서 상의까지 깔끔하게 살아난다는 점이에요.')
        else:
            parts.append('데일리로 입기 좋게 너무 과하지 않으면서도 실루엣이 단정하게 정리된다는 점이에요.')
    elif cat in {'자켓','블라우스','셔츠','니트','맨투맨','티셔츠'}:
        if any(k in corpus for k in ['루즈','여유']):
            parts.append('답답하게 붙지 않고 체형 부담을 덜어준다는 점이에요.')
        elif any(k in corpus for k in ['히든','카라','반오픈']):
            parts.append('단정한 무드가 살아서 출근룩이나 모임룩으로 활용하기 좋다는 점이에요.')
        else:
            parts.append('과하게 힘주지 않아도 깔끔하게 정리된다는 점이에요.')
    else:
        parts.append('코디에 무난하게 녹아들면서 활용도가 좋다는 점이에요.')
    review = build_review_note(clean_text((db_product or {}).get('product_no','') or product_context.get('product_no','')))
    if review:
        parts.append(review)
    return ' '.join(parts)

def is_selected_item_outfit_request(user_text: str) -> bool:
    q = clean_text(user_text)
    if st.session_state.get("last_selected_index") is None:
        return False
    return (
        any(k in q for k in ["같이 입", "어울리는", "코디", "안에 입"]) and
        any(k in q for k in ["바지", "슬랙스", "팬츠", "치마", "스커트", "블라우스", "셔츠", "니트", "신발", "가방", "머플러"])
    ) or any(k in q for k in ["바지 추천", "팬츠 추천", "블라우스 추천", "셔츠 추천", "신발 추천", "가방 추천"])

def is_followup_size_on_recommendations(user_text: str) -> bool:
    q = clean_text(user_text)
    return bool(st.session_state.get("last_recommendations")) and (
        ("추천해준" in q and any(k in q for k in ["사이즈", "맞아", "맞을까", "괜찮아"])) or
        (any(k in q for k in ["그거", "그 상품", "1번", "2번", "3번", "첫 번째", "두 번째", "세 번째"]) and any(k in q for k in ["사이즈", "맞아", "맞을까", "괜찮아"]))
    )

# =========================================================
# 상태/컨텍스트
# =========================================================
def extract_selected_index(user_text: str) -> Optional[int]:
    q = clean_text(user_text)
    m = re.search(r"([123])번", q)
    if m:
        return int(m.group(1)) - 1
    if "첫 번째" in q or "첫번째" in q:
        return 0
    if "두 번째" in q or "두번째" in q:
        return 1
    if "세 번째" in q or "세번째" in q:
        return 2
    return None

def update_selected_index_from_message(user_text: str) -> None:
    idx = extract_selected_index(user_text)
    if idx is not None:
        st.session_state.last_selected_index = idx

def infer_target_category_from_query(user_text: str, current_product: Dict) -> str:
    q = clean_text(user_text)
    if st.session_state.get("last_selected_index") is not None:
        if any(k in q for k in ["바지", "슬랙스", "팬츠", "데님", "청바지"]):
            return "팬츠"
        if any(k in q for k in ["스커트", "치마"]):
            return "스커트"
        if any(k in q for k in ["블라우스"]):
            return "블라우스"
        if any(k in q for k in ["셔츠"]):
            return "셔츠"
        if any(k in q for k in ["니트", "가디건"]):
            return "니트"
        if any(k in q for k in ["신발", "슈즈", "로퍼", "힐", "샌들", "부츠"]):
            return "신발"
        if any(k in q for k in ["가방", "백", "토트", "크로스"]):
            return "가방"
        if any(k in q for k in ["머플러", "스카프"]):
            return "악세사리"

    if "니트티" in q or "니트 티" in q:
        return "니트티"
    if "맨투맨" in q:
        return "맨투맨"
    if "블라우스" in q:
        return "블라우스"
    if "셔츠" in q:
        return "셔츠"
    if "니트" in q or "가디건" in q:
        return "니트"
    if "자켓" in q or "재킷" in q or "아우터" in q:
        return "자켓"
    if any(k in q for k in ["바지", "슬랙스", "팬츠", "데님", "청바지"]):
        return "팬츠"
    if any(k in q for k in ["스커트", "치마"]):
        return "스커트"
    if any(k in q for k in ["신발", "슈즈", "로퍼", "힐", "샌들", "부츠"]):
        return "신발"
    if any(k in q for k in ["가방", "백", "토트", "크로스"]):
        return "가방"
    if any(k in q for k in ["머플러", "스카프"]):
        return "악세사리"
    current_cat = clean_text(current_product.get("category", ""))
    if current_cat in ["자켓", "블라우스", "셔츠", "니트", "니트티", "맨투맨", "티셔츠"]:
        return "팬츠"
    return ""

def update_conversation_context(user_text: str) -> None:
    situation = extract_situation_tag(user_text)
    style = extract_style_tag(user_text)
    if situation:
        st.session_state.pending_situation = situation
    if style:
        st.session_state.pending_style = style
    target = infer_target_category_from_query(user_text, {})
    if target:
        st.session_state.pending_target_category = target

def continue_previous_flow(product_context: Dict, db_product: Optional[Dict]) -> str:
    pending_target = clean_text(st.session_state.get("pending_target_category", ""))
    pending_situation = clean_text(st.session_state.get("pending_situation", ""))
    pending_style = clean_text(st.session_state.get("pending_style", ""))
    if pending_target:
        prompt = "추천"
        if pending_situation:
            prompt += f" {pending_situation}"
        if pending_style:
            prompt += f" {pending_style}"
        prompt += f" {pending_target}"
        return recommend_products(prompt, product_context, db_product)
    if st.session_state.get("last_recommendations"):
        return "좋아요 :) 방금 고른 후보 기준으로 더 볼게요. 번호나 보고 싶은 포인트를 바로 말씀 주세요."
    return "좋아요 :) 지금 보시는 상품 기준으로 바로 이어서 같이 볼게요. 궁금한 걸 자연스럽게 말씀해주시면 그 흐름대로 봐드릴게요."

def get_active_base_product(product_context: Dict, db_product: Optional[Dict]) -> Dict:
    override = st.session_state.get("active_product_override", {}) or {}
    if override:
        return override
    return {
        "product_name": clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "")),
        "category": clean_text((db_product or {}).get("category", "") or product_context.get("category", "")),
        "product_no": clean_text((db_product or {}).get("product_no", "") or product_context.get("product_no", "")),
    }

# =========================================================
# 리뷰/모델 활용
# =========================================================
def get_review_summary(product_no: str) -> Dict:
    if not product_no:
        return {}
    return REVIEW_SUMMARY.get(str(product_no), {}) or REVIEW_SUMMARY.get(normalize_product_no(product_no), {}) or {}

def build_review_note(product_no: str, user_size: str = "") -> str:
    summary = get_review_summary(product_no)
    if not summary:
        return ""
    parts = []
    if summary.get("review_count", 0) >= 3:
        parts.append(f"후기 {summary.get('review_count')}건 기준")
    if summary.get("top_size_mentions") and user_size:
        size_hits = [s for s, _ in summary.get("top_size_mentions", []) if user_size in str(s)]
        if size_hits:
            parts.append(f"{user_size} 언급 후기도 보여요")
    if summary.get("top_good"):
        kw = summary["top_good"][0][0]
        mapping = {"편하":"편하다는 반응", "부드럽":"부드럽다는 반응", "잘 맞":"잘 맞는다는 반응", "날씬":"날씬해 보인다는 반응", "깔끔":"깔끔하다는 반응", "무난":"무난하다는 반응"}
        parts.append(mapping.get(kw, "만족 반응이 있는 편이에요"))
    if summary.get("top_bad"):
        bad = summary["top_bad"][0][0]
        mapping = {"작":"작게 느꼈다는 반응도 있고요", "타이트":"타이트하다는 반응도 있고요", "길":"길게 느꼈다는 반응도 있고요", "부해":"부해 보인다는 반응도 있고요"}
        parts.append(mapping.get(bad, "아쉬운 반응도 조금 있어요"))
    if not parts:
        return ""
    return " ".join(parts[:2]) + "."

def build_model_note() -> str:
    if not MODEL_PROFILES:
        return ""
    usable = []
    for m in MODEL_PROFILES[:2]:
        usable.append(f"{m.get('height_cm')}cm/{m.get('weight_kg')}kg")
    if usable:
        return f"상세페이지 모델컷은 대체로 {' 또는 '.join(usable)} 체형 기준 느낌이에요."
    return ""

# =========================================================
# 사이즈 엔진
# =========================================================
def parse_range_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    text = clean_text(text).replace("~", "-")
    tokens = []
    ordered = ["44", "55반", "55", "66반", "66", "77반", "77", "88", "99"]
    for token in ordered:
        if token in text:
            r = size_rank(token)
            if r:
                tokens.append(r)
    if not tokens:
        if "FREE" in text.upper():
            return size_rank("55"), size_rank("77")
        return None, None
    return min(tokens), max(tokens)

def get_size_range_ranks(text: str) -> Tuple[Optional[int], Optional[int]]:
    return parse_range_from_text(text)

def context_uses_top_size(product_context: Dict, db_product: Optional[Dict]) -> bool:
    corpus = " ".join([
        clean_text((db_product or {}).get("category", "")),
        clean_text((db_product or {}).get("sub_category", "")),
        clean_text((db_product or {}).get("product_name", "")),
        clean_text((product_context or {}).get("category", "")),
        clean_text((product_context or {}).get("product_name", "")),
    ])
    cat = detect_category_from_name(corpus, corpus)
    return cat not in {"팬츠", "스커트", "신발"}

def get_active_user_size(product_context: Dict, db_product: Optional[Dict]) -> Tuple[str, str]:
    body = build_body_context()
    cat = detect_category_from_name(
        clean_text((db_product or {}).get("product_name", "")) + " " + clean_text((db_product or {}).get("category", "")) + " " + clean_text(product_context.get("product_name", "")),
        ""
    )
    if cat in {"팬츠", "스커트"}:
        return clean_text(body.get("bottom_size", "")), "하의"
    if cat == "신발":
        return clean_text(body.get("shoe_size", "")), "신발"
    return clean_text(body.get("top_size", "")), "상의"

def normalize_garment_chest(db_product: Optional[Dict]) -> Optional[float]:
    if not db_product:
        return None
    raw_measurements = clean_text(db_product.get("raw_measurements", ""))
    m2 = re.search(r'가슴둘레["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)', raw_measurements)
    if m2:
        return float(m2.group(1))
    raw = clean_text(db_product.get("chest", ""))
    if not raw:
        return None
    m = re.search(r"\d+(?:\.\d+)?", raw)
    if not m:
        return None
    chest = float(m.group())
    measure_type = clean_text(db_product.get("chest_measure_type", "")).lower()
    if chest < 70:
        chest = chest * 2
    elif measure_type in {"flat", "half", "half_width"}:
        chest = chest * 2
    return chest

def body_chest_estimate(user_size: str) -> Optional[float]:
    return BODY_CHEST_ESTIMATE.get(clean_text(user_size))

def classify_fit_text(corpus: str) -> str:
    c = clean_text(corpus)
    if any(k in c for k in ["루즈", "여유", "오버", "벌룬"]):
        return "loose"
    if any(k in c for k in ["슬림", "정핏", "기본핏"]):
        return "regular"
    return "unknown"

def evaluate_size_support(user_size: str, body_label: str, product_context: Dict, db_product: Optional[Dict]) -> Dict:
    rank = size_rank(user_size) if body_label != "신발" else None
    if body_label == "신발":
        # simple shoe handling from options text
        opts = " ".join([
            clean_text((db_product or {}).get("size_range", "")),
            clean_text((db_product or {}).get("product_summary", "")),
            clean_text(product_context.get("summary", "")),
            clean_text(product_context.get("raw_excerpt", ""))[:1200],
        ])
        if not user_size:
            return {"supported": None, "reason": "신발 사이즈를 먼저 알려주시면 더 정확하게 볼 수 있어요.", "confidence": "unknown"}
        sizes = re.findall(r"\b(225|230|235|240|245|250|255|260)\b", opts)
        if sizes and user_size not in sizes:
            return {"supported": False, "reason": f"현재 보이는 옵션 기준으로는 {user_size} 사이즈가 바로 확인되진 않아요.", "confidence": "range"}
        return {"supported": True, "reason": f"지금 보이는 옵션 기준으로는 {user_size} 사이즈를 같이 볼 수 있는 쪽이에요.", "confidence": "range"}

    if not rank:
        return {"supported": None, "reason": "", "confidence": "unknown"}
    lo, hi = get_size_range_ranks(clean_text((db_product or {}).get("size_range", "")) or clean_text(product_context.get("size_tip", "")))
    range_ok = lo is not None and hi is not None and lo <= rank <= hi
    garment_chest = normalize_garment_chest(db_product)
    body_chest = body_chest_estimate(user_size)
    ease = None
    if garment_chest and body_chest:
        ease = garment_chest - body_chest

    if not range_ok:
        return {"supported": False, "reason": f"{clean_text((db_product or {}).get('size_range', '')) or '사이즈 범위'} 기준으로는 고객님 {body_label} {user_size}보다 작게 나오는 편이에요.", "confidence": "range", "ease": ease}

    if ease is None:
        return {"supported": True, "reason": f"고객님 {body_label} {user_size} 기준으로는 사이즈 범위 안에 들어와요.", "confidence": "range", "ease": None}

    if ease >= 10:
        return {"supported": True, "reason": "사이즈 범위 안이고 실측 기준으로도 여유가 있는 편이에요.", "confidence": "chest", "ease": ease}
    if ease >= 5:
        return {"supported": True, "reason": "사이즈 범위 안이고 깔끔하게 맞는 느낌에 가까울 가능성이 커요.", "confidence": "chest", "ease": ease}
    if ease >= 0:
        return {"supported": "edge", "reason": "사이즈 범위 안이긴 한데 아주 여유 있는 느낌보다는 경계선에 가까운 쪽이에요.", "confidence": "chest", "ease": ease}
    if ease >= -2:
        return {"supported": "edge", "reason": "사이즈 범위에는 들어오지만 실측 기준으로는 여유가 아주 많진 않은 편이에요.", "confidence": "chest", "ease": ease}
    return {"supported": False, "reason": f"사이즈 범위에는 들어오지만 실측 기준으로 여유가 부족해 보여요(의류 가슴둘레 약 {round(garment_chest)}cm).", "confidence": "chest", "ease": ease}

def is_size_pushback_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["추천이던데", "77까지", "66까지", "타이트하다고", "근데 왜", "그런데 왜", "추천인데", "맞다며"])

def build_size_pushback_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    user_size, body_label = get_active_user_size(product_context, db_product)
    product_name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")
    result = evaluate_size_support(user_size, body_label, product_context, db_product)
    fit_corpus = " ".join([
        clean_text((db_product or {}).get("fit_type", "")),
        clean_text(product_context.get("fit", "")),
        clean_text(product_context.get("summary", "")),
        clean_text(product_context.get("size_tip", "")),
    ])
    fit_type = classify_fit_text(fit_corpus)
    if result.get("supported") is False:
        return f"{product_name}은 추천 범위 문구상으로는 넓어 보여도, 고객님 {body_label} {user_size} 기준으로 보면 편하게 입는 느낌까지는 아니어서 제가 보수적으로 말씀드린 거예요. 입는 것 자체보다 핏이 어떻게 떨어질지가 더 중요해서, 지금 기준으로는 다른 쪽을 같이 보시는 게 더 안전해요."
    if result.get("supported") == "edge":
        return f"{product_name}은 추천 범위 안쪽이긴 한데 고객님 {body_label} {user_size}가 딱 경계에 가까운 편이라서 그래요. 못 입는다는 뜻은 아니고, 여유 있게 기대하시면 조금 아쉬울 수 있다는 쪽으로 이해하시면 제일 정확해요."
    if fit_type == "regular":
        return f"{product_name}은 {body_label} {user_size} 기준으로 입으실 수 있는 쪽이 맞아요. 다만 추천 문구가 넉넉하게 느껴지더라도 실제 핏은 정돈된 쪽일 수 있어서, 루즈하게 떨어진다기보다 깔끔하게 맞는 느낌으로 보시면 더 맞아요."
    return f"{product_name}은 고객님 {body_label} {user_size} 기준으로 무리 없는 쪽이 맞아요. 제가 드린 설명은 못 입는다는 뜻이 아니라, 실제로는 여유감보다 핏 체감이 더 중요하다는 뜻으로 봐주시면 돼요."

def build_size_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    user_size, body_label = get_active_user_size(product_context, db_product)
    product_name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")
    if not user_size:
        if body_label == "신발":
            return "신발은 채팅창에 평소 신는 사이즈를 적어주시면 더 정확하게 봐드릴게요 :)"
        return "사이즈 같이 봐드릴게요 :) 상의랑 하의 사이즈 먼저 알려주시면 더 정확하게 말씀드릴 수 있어요."
    if is_size_pushback_question(user_text):
        return build_size_pushback_answer(user_text, product_context, db_product)

    result = evaluate_size_support(user_size, body_label, product_context, db_product)
    q = clean_text(user_text)
    fit_corpus = " ".join([clean_text((db_product or {}).get("fit_type", "")), clean_text(product_context.get("fit", "")), clean_text(product_context.get("summary", ""))])
    fit_type = classify_fit_text(fit_corpus)
    upper_heavy = "상체" in q and any(k in q for k in ["큰", "크고", "있는", "가슴"])
    looks_short = "키가 작" in q or "키가 작은" in q

    parts = []
    if body_label == "신발":
        if result.get("supported") is False:
            return f"고객님 신발 {user_size} 기준이면 {product_name}은 {result.get('reason','지금 보이는 옵션 기준으로는 어려워 보여요.')}"
        return f"고객님 신발 {user_size} 기준이면 {product_name}은 {result.get('reason','같이 볼 수 있는 쪽이에요.')}"

    if result.get("supported") is False:
        parts.append(f"고객님 {body_label} {user_size} 기준이면 {product_name}은 편하게 입는 기준으로는 추천을 강하게 드리기 어려워요.")
        parts.append(result.get("reason", ""))
        if upper_heavy and context_uses_top_size(product_context, db_product):
            parts.append("상체 쪽은 조금 더 또렷하게 느껴질 수 있어서 여유 있는 쪽을 같이 보시는 게 더 안전해요.")
    elif result.get("supported") == "edge":
        parts.append(f"고객님 {body_label} {user_size} 기준이면 {product_name}은 경계선에 가까운 쪽이에요.")
        parts.append(result.get("reason", ""))
        if upper_heavy:
            parts.append("가슴이나 어깨 쪽이 있는 편이면 앞모습이 조금 더 또렷하게 느껴질 수 있어요.")
    else:
        parts.append(f"고객님 {body_label} {user_size} 기준이면 {product_name}은 무리 없는 쪽이에요 :)")
        parts.append(result.get("reason", ""))
        if fit_type == "regular":
            parts.append("다만 루즈핏보다는 단정하게 떨어지는 느낌에 더 가까울 수 있어요.")
        elif fit_type == "loose":
            parts.append("전체적으로 답답하게 붙는 타입은 아닐 가능성이 커요.")
        if upper_heavy and fit_type == "regular":
            parts.append("상체가 있는 편이면 앞쪽은 조금 더 또렷하게 느껴질 수는 있어요.")
        if looks_short and fit_type == "loose":
            parts.append("키가 작은 편이면 기장은 살짝 크게 느껴질 수도 있어요.")
    review_note = build_review_note(clean_text((db_product or {}).get("product_no", "") or product_context.get("product_no", "")), user_size)
    if review_note:
        parts.append(review_note)
    model_note = build_model_note()
    if model_note and context_uses_top_size(product_context, db_product):
        parts.append(model_note)
    return " ".join([p for p in parts if p])

# =========================================================
# 추천/비교 엔진
# =========================================================
def normalized_row_category(rowd: Dict) -> str:
    combined = " ".join([clean_text(rowd.get("product_name", "")), clean_text(rowd.get("category", "")), clean_text(rowd.get("sub_category", ""))])
    return detect_category_from_name(combined, combined)

def row_blob(rowd: Dict) -> str:
    cols = ["product_name", "category", "sub_category", "style_tags", "coordination_items", "body_cover_features", "recommended_body_type", "product_summary", "fabric", "fit_type", "color_options"]
    return " ".join(clean_text(rowd.get(c, "")) for c in cols)

def row_matches_target(rowd: Dict, target_cat: str) -> bool:
    row_cat = normalized_row_category(rowd)
    if target_cat == "니트":
        return row_cat in {"니트", "니트티"}
    if target_cat == "니트티":
        return row_cat == "니트티"
    return row_cat == target_cat

def item_supports_user(rowd: Dict, target_cat: str) -> bool:
    temp_ctx = {
        "product_name": clean_text(rowd.get("product_name", "")),
        "category": normalized_row_category(rowd),
        "summary": clean_text(rowd.get("product_summary", "")),
        "fit": clean_text(rowd.get("fit_type", "")),
        "size_tip": clean_text(rowd.get("size_range", "")),
        "product_no": clean_text(rowd.get("product_no", "")),
    }
    user_size, body_label = get_active_user_size(temp_ctx, rowd)
    if not user_size:
        return True
    result = evaluate_size_support(user_size, body_label, temp_ctx, rowd)
    return result.get("supported") in [True, "edge", None]

def get_base_selected_product() -> Optional[Dict]:
    recos = st.session_state.get("last_recommendations", [])
    idx = st.session_state.get("last_selected_index", None)
    if idx is None or idx >= len(recos):
        return None
    row = recos[idx]
    return {
        "product_name": clean_text(row.get("product_name", "")),
        "category": normalized_row_category(row),
        "summary": clean_text(row.get("product_summary", "")),
        "fit": clean_text(row.get("fit_type", "")),
        "size_tip": clean_text(row.get("size_range", "")),
        "product_no": clean_text(row.get("product_no", "")),
    }

def build_style_reason(rowd: Dict, user_text: str, target_cat: str) -> str:
    name = clean_text(rowd.get("product_name", ""))
    corpus = " ".join([name, clean_text(rowd.get("product_summary", "")), clean_text(rowd.get("fit_type", "")), clean_text(rowd.get("style_tags", "")), clean_text(rowd.get("body_cover_features", ""))])
    q = clean_text(user_text)
    reasons = []
    selected_base = get_base_selected_product()
    if target_cat in ["팬츠", "스커트"]:
        if any(k in corpus for k in ["일자", "세미와이드", "와이드", "앵클"]):
            reasons.append("라인이 정돈돼 보여서 상의를 깔끔하게 받쳐주는 쪽이에요")
        elif any(k in corpus for k in ["배기", "턱", "핀턱"]):
            reasons.append("허벅지나 복부 라인을 조금 더 편하게 커버해주는 쪽이에요")
        elif any(k in corpus for k in ["논페이드", "데님"]):
            reasons.append("너무 힘주지 않으면서도 단정하게 연결하기 좋은 쪽이에요")
        else:
            if selected_base:
                reasons.append(f"{selected_base.get('product_name','지금 고른 상의')}랑 붙였을 때 전체 라인이 깔끔하게 정리되는 쪽이에요")
            else:
                reasons.append("전체 실루엣이 과하게 무겁지 않게 정리되는 쪽이에요")
    elif target_cat in ["블라우스", "셔츠"]:
        if any(k in corpus for k in ["히든", "카라", "반오픈"]):
            reasons.append("자켓 안에 받쳐 입었을 때 단정한 느낌이 잘 살아나는 쪽이에요")
        elif any(k in corpus for k in ["스트랩", "랩"]):
            reasons.append("여성스럽지만 과하지 않게 라인을 정리해주는 쪽이에요")
        else:
            reasons.append("출근룩이나 모임룩에 무난하게 받쳐 입기 좋은 쪽이에요")
    elif target_cat == "자켓":
        if any(k in corpus for k in ["셔츠 자켓", "가벼운", "경량"]):
            reasons.append("부담 없이 걸치기 좋고 실루엣이 답답하지 않은 편이에요")
        elif any(k in corpus for k in ["백버튼", "클래식", "정장"]):
            reasons.append("전체 실루엣을 단정하게 잡아주는 편이에요")
        else:
            reasons.append("핏이 과하게 크지 않아 깔끔하게 정리돼 보이기 좋아요")
    elif target_cat == "신발":
        reasons.append("코디를 너무 무겁지 않게 마무리해주기 좋은 쪽이에요")
    elif target_cat == "가방":
        reasons.append("전체 스타일을 단정하게 정리해주기 좋은 쪽이에요")
    else:
        if any(k in corpus for k in ["루즈", "여유"]):
            reasons.append("상체가 있는 편이어도 답답한 느낌이 덜한 편이에요")
        elif any(k in corpus for k in ["슬림", "정핏"]):
            reasons.append("너무 부해 보이지 않고 깔끔하게 잡히는 쪽이에요")
        else:
            reasons.append("무난하게 손이 가면서 실루엣이 정리되는 쪽이에요")
    if any(k in q for k in ["출근", "학교", "상담", "방문", "모임"]):
        reasons.append("지금처럼 단정하게 보여야 하는 자리에도 잘 맞는 쪽이에요")
    review_note = build_review_note(clean_text(rowd.get("product_no", "")))
    if review_note:
        reasons.append(review_note.replace("후기", "").strip())
    return " ".join(reasons[:2]).strip()

def pick_recommendation_rows(target_cat: str, user_text: str, product_context: Dict, db_product: Optional[Dict], limit: int = 3) -> List[Dict]:
    current_no = clean_text((db_product or {}).get("product_no", "") or product_context.get("product_no", ""))
    seen_names = set()
    candidates = []
    q = clean_text(user_text)

    # allow base selected product context for outfit
    for row in DB_ROWS:
        name = clean_text(row.get("product_name", ""))
        if not name:
            continue
        if current_no and normalize_product_no(row.get("product_no", "")) == normalize_product_no(current_no):
            continue
        if not row_matches_target(row, target_cat):
            continue
        if name in seen_names:
            continue
        if any(k in q for k in ["학교", "상담", "출근"]) and any(k in name for k in ["후드", "쭈리", "트레이닝"]):
            continue
        if not item_supports_user(row, target_cat):
            continue
        seen_names.add(name)
        candidates.append(row)
        if len(candidates) >= limit:
            break
    return candidates[:limit]

def recommend_products(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    base_product = get_base_selected_product() or get_active_base_product(product_context, db_product)
    target_cat = infer_target_category_from_query(user_text, base_product)
    if not target_cat:
        target_cat = "팬츠"
    picked = pick_recommendation_rows(target_cat, user_text, product_context, db_product, limit=3)
    if not picked:
        return f"지금 조건에 딱 맞는 {target_cat}가 바로 많이 잡히진 않아서요. 원하시면 조금 더 단정하게 볼지, 편하게 볼지 기준을 맞춰서 다시 골라드릴게요 :)"

    st.session_state.last_recommendations = picked
    st.session_state.last_selected_index = None
    st.session_state.pending_target_category = target_cat

    prefix = {
        "니트티": "네, 고객님께 잘 맞을 만한 니트티로 먼저 골라드릴게요.",
        "맨투맨": "네, 고객님께 잘 맞을 만한 맨투맨으로 먼저 골라드릴게요.",
        "블라우스": "네, 고객님께 잘 맞을 만한 블라우스로 먼저 골라드릴게요.",
        "셔츠": "네, 고객님께 잘 맞을 만한 셔츠로 먼저 골라드릴게요.",
        "니트": "네, 고객님께 잘 맞을 만한 니트로 먼저 골라드릴게요.",
        "자켓": "네, 고객님께 잘 맞을 만한 자켓 쪽으로 먼저 골라드릴게요.",
        "팬츠": f"네, {base_product.get('product_name','지금 보시는 상품')}이랑 잘 어울리는 바지 쪽으로 먼저 골라드릴게요.",
        "스커트": f"네, {base_product.get('product_name','지금 보시는 상품')}이랑 잘 어울리는 스커트 쪽으로 먼저 골라드릴게요.",
        "신발": f"네, {base_product.get('product_name','지금 보시는 상품')}이랑 잘 어울리는 신발 쪽으로 먼저 골라드릴게요.",
        "가방": f"네, {base_product.get('product_name','지금 보시는 상품')}이랑 잘 어울리는 가방 쪽으로 먼저 골라드릴게요.",
        "악세사리": f"네, {base_product.get('product_name','지금 보시는 상품')}이랑 같이 보기 좋은 소품으로 먼저 골라드릴게요.",
    }.get(target_cat, "네, 같이 보기 좋은 상품으로 먼저 골라드릴게요.")

    lines = [prefix]
    for i, row in enumerate(picked, start=1):
        lines.append(f"{i}. {clean_text(row.get('product_name',''))} ({clean_text(row.get('size_range',''))}) 🔗 — {build_style_reason(row, user_text, target_cat)}")
    lines.append("번호 말씀해주시면 사이즈감이나 코디까지 바로 이어서 봐드릴게요 :)")
    return "\n".join(lines)

def build_selected_item_detail_answer(user_text: str) -> str:
    recos = st.session_state.get("last_recommendations", [])
    idx = extract_selected_index(user_text)
    if idx is None:
        idx = st.session_state.get("last_selected_index", None)
    if idx is None or idx >= len(recos):
        return "지금 보고 있는 상품 번호를 한 번만 더 말씀해주시면 바로 이어서 자세히 봐드릴게요 :)"
    st.session_state.last_selected_index = idx
    row = recos[idx]
    st.session_state.active_product_override = {
        "product_name": clean_text(row.get("product_name", "")),
        "category": normalized_row_category(row),
        "product_no": clean_text(row.get("product_no", "")),
    }
    temp_ctx = {
        "product_name": clean_text(row.get("product_name", "")),
        "category": normalized_row_category(row),
        "summary": clean_text(row.get("product_summary", "")),
        "fit": clean_text(row.get("fit_type", "")),
        "size_tip": clean_text(row.get("size_range", "")),
        "product_no": clean_text(row.get("product_no", "")),
    }
    user_size, body_label = get_active_user_size(temp_ctx, row)
    size_result = evaluate_size_support(user_size, body_label, temp_ctx, row)
    q = clean_text(user_text)
    want_size = "사이즈" in q
    want_coordi = any(k in q for k in ["코디", "바지", "같이 입", "어울", "신발", "가방"])
    want_all = any(k in q for k in ["전체적으로", "다 같이", "다같이", "설명", "얘기해줘", "어때"]) or (not want_size and not want_coordi)
    parts = [f"{idx+1}번 {clean_text(row.get('product_name',''))} 기준으로 보면,"]
    if want_all or want_size:
        parts.append(f"고객님 {body_label} {user_size} 기준으로는 {size_result.get('reason','무리 없는 쪽이에요.')}")
    if want_all:
        parts.append(build_style_reason(row, user_text, normalized_row_category(row)))
    if want_all or want_coordi:
        cat = normalized_row_category(row)
        if cat in ["자켓", "블라우스", "셔츠", "니트", "니트티", "맨투맨", "티셔츠"]:
            parts.append("슬랙스나 일자 팬츠 쪽이랑 같이 입으시면 전체가 단정하게 정리돼 보여요.")
        elif cat in ["팬츠", "스커트"]:
            parts.append("상의는 너무 부한 것보다 깔끔한 셔츠나 니트 쪽이 더 잘 어울려요.")
    review_note = build_review_note(clean_text(row.get("product_no", "")), user_size)
    if review_note:
        parts.append(review_note)
    return " ".join(parts)

def build_reco_followup_size_answer(user_text: str) -> str:
    recos = st.session_state.get("last_recommendations", [])
    if not recos:
        return "지금 바로 이어서 볼 추천 상품이 없어서요 :) 먼저 보고 싶은 상품 하나 골라주시면 그 기준으로 바로 봐드릴게요."
    idx = extract_selected_index(user_text)
    if idx is None:
        idx = st.session_state.get("last_selected_index", None)
    if idx is None or idx >= len(recos):
        return "몇 번 상품 기준으로 볼지 알려주시면 바로 이어서 봐드릴게요 :)"
    row = recos[idx]
    st.session_state.last_selected_index = idx
    temp_ctx = {
        "product_name": clean_text(row.get("product_name", "")),
        "category": normalized_row_category(row),
        "summary": clean_text(row.get("product_summary", "")),
        "fit": clean_text(row.get("fit_type", "")),
        "size_tip": clean_text(row.get("size_range", "")),
        "product_no": clean_text(row.get("product_no", "")),
    }
    user_size, body_label = get_active_user_size(temp_ctx, row)
    result = evaluate_size_support(user_size, body_label, temp_ctx, row)
    review_note = build_review_note(clean_text(row.get("product_no", "")), user_size)
    answer = f"{idx+1}번으로 추천드린 {clean_text(row.get('product_name',''))}은 고객님 {body_label} {user_size} 기준으로 보면 {result.get('reason','무리 없는 쪽으로 보여요.')}"
    if review_note:
        answer += " " + review_note
    return answer

# =========================================================
# 비교 엔진
# =========================================================
def extract_compare_target_phrase(user_text: str) -> str:
    q = clean_text(user_text)
    q = re.sub(r"(비교해줘|비교해 줘|비교해봐|고민되는데|뭐가 더.*|어느 게.*|더 나아.*)$", "", q).strip()
    m = re.search(r"(?:이|그|저)?\s*[^ ]+\s*(?:이랑|랑|와|과)\s*(.+)", q)
    if m:
        return clean_text(m.group(1))
    return ""

def find_product_candidates_by_name(query: str, current_product_no: str = "") -> List[Dict]:
    query = clean_text(query)
    scored = []
    for row in DB_ROWS:
        if current_product_no and normalize_product_no(row.get("product_no", "")) == normalize_product_no(current_product_no):
            continue
        score = token_overlap_score(query, clean_text(row.get("product_name", "")))
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: (-x[0], clean_text(x[1].get("product_name", ""))))
    return [r for _, r in scored[:3]]

def build_comparison_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    base = get_active_base_product(product_context, db_product)
    current_name = clean_text(base.get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")
    target_phrase = extract_compare_target_phrase(user_text)
    current_no = clean_text(base.get("product_no", ""))
    candidates = []

    if st.session_state.get("last_recommendations"):
        for row in st.session_state.get("last_recommendations", []):
            score = token_overlap_score(target_phrase, clean_text(row.get("product_name", "")))
            if score > 0:
                candidates.append((score, row))
        candidates.sort(key=lambda x: -x[0])
        candidates = [r for _, r in candidates[:2]]

    if not candidates and target_phrase:
        candidates = find_product_candidates_by_name(target_phrase, current_no)

    if not target_phrase or (target_phrase in ["다른 슬랙스", "다른 바지", "다른 팬츠", "다른 자켓", "다른 블라우스", "다른 셔츠"] and not candidates):
        current_cat = normalized_row_category(base_row or db_product or {}) if (base_row or db_product) else detect_category_from_name(current_name, current_name)
        fallback_cat = current_cat if current_cat in ["팬츠","자켓","블라우스","셔츠","니트","스커트"] else "팬츠"
        compare_rows = pick_recommendation_rows(fallback_cat, user_text, product_context, db_product, limit=2)
        if compare_rows:
            st.session_state.last_compare_candidates = compare_rows
            names = [clean_text(r.get("product_name","")) for r in compare_rows[:2]]
            return f"지금 보고 계신 {current_name} 기준으로 비교할 만한 비슷한 상품을 먼저 골라드리면 {names[0]} / {names[1]} 쪽이에요. 둘 중 하나를 말씀해주시면 바로 비교해드릴게요 :)"
        return f"지금 보고 계신 {current_name} 기준으로 비교할 다른 상품을 바로 많이 잡지는 못했어요. 같은 카테고리 안에서 하나 골라주시면 바로 같이 비교해드릴게요 :)"
    if not target_phrase or not candidates:
        return "비교할 다른 상품명을 제가 조금 더 정확히 잡아야 해서요 :) 비교하고 싶은 상품명을 한 번만 더 적어주시면 바로 같이 봐드릴게요."

    if len(candidates) > 1 and token_overlap_score(target_phrase, clean_text(candidates[0].get("product_name",""))) == token_overlap_score(target_phrase, clean_text(candidates[1].get("product_name",""))):
        names = [clean_text(c.get("product_name","")) for c in candidates[:2]]
        return f"말씀하신 상품이 {names[0]} 쪽인지, {names[1]} 쪽인지 제가 한 번만 확인할게요 :)"

    target = candidates[0]
    base_row = db_product or get_db_product(base.get("product_no", ""))
    base_ctx = {
        "product_name": clean_text(base.get("product_name", "")),
        "category": clean_text(base.get("category", "")),
        "summary": clean_text((base_row or {}).get("product_summary", "")),
        "fit": clean_text((base_row or {}).get("fit_type", "")),
        "size_tip": clean_text((base_row or {}).get("size_range", "")),
        "product_no": clean_text((base_row or {}).get("product_no", "")),
    }
    user_size, body_label = get_active_user_size(base_ctx, base_row)
    base_eval = evaluate_size_support(user_size, body_label, base_ctx, base_row)
    target_ctx = {
        "product_name": clean_text(target.get("product_name", "")),
        "category": normalized_row_category(target),
        "summary": clean_text(target.get("product_summary", "")),
        "fit": clean_text(target.get("fit_type", "")),
        "size_tip": clean_text(target.get("size_range", "")),
        "product_no": clean_text(target.get("product_no", "")),
    }
    target_eval = evaluate_size_support(user_size, body_label, target_ctx, target)
    situation = clean_text(st.session_state.get("pending_situation", ""))

    def score_result(ev):
        if ev.get("supported") is True:
            return 3
        if ev.get("supported") == "edge":
            return 2
        if ev.get("supported") is None:
            return 1
        return 0

    base_score = score_result(base_eval)
    target_score = score_result(target_eval)

    if target_score > base_score:
        conclusion = f"고객님 {body_label} {user_size} 기준이면 {clean_text(target.get('product_name',''))} 쪽이 조금 더 안정적이에요."
    elif target_score < base_score:
        conclusion = f"고객님 {body_label} {user_size} 기준이면 지금 보고 계신 {clean_text(base.get('product_name',''))} 쪽이 조금 더 안정적이에요."
    else:
        conclusion = f"두 상품 다 가능은 한데 고객님 {body_label} {user_size} 기준으로는 핏 취향에 따라 갈릴 수 있어요."

    parts = [conclusion]
    parts.append(f"지금 보고 계신 상품은 {base_eval.get('reason','무리 없는 쪽이에요.')}")
    parts.append(f"{clean_text(target.get('product_name',''))}은 {target_eval.get('reason','무리 없는 쪽이에요.')}")
    if situation in {"학교", "출근"}:
        parts.append("단정하게 보여야 하는 자리 기준이면 너무 캐주얼한 쪽보다 셔츠/블라우스나 슬랙스 무드에 가까운 쪽이 더 안전해요.")
    base_review = build_review_note(clean_text(base.get("product_no","")), user_size)
    target_review = build_review_note(clean_text(target.get("product_no","")), user_size)
    if target_review and target_score >= base_score:
        parts.append(f"후기 쪽도 {clean_text(target.get('product_name',''))}이 {target_review}")
    st.session_state.last_compare_candidates = [base_row or {}, target]
    return " ".join([p for p in parts if p])

# =========================================================
# 컬러/상황 답변
# =========================================================
def build_color_style_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    q = clean_text(user_text)
    base = get_base_selected_product() or get_active_base_product(product_context, db_product)
    active_row = get_db_product(base.get("product_no","")) or db_product or {}
    product_name = clean_text(base.get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")
    text_pool = " ".join([clean_text((active_row or {}).get("color_options", "")), clean_text(product_context.get("raw_excerpt", ""))[:1200], clean_text(product_context.get("summary", ""))])
    colors = []
    for c in ["블랙", "아이보리", "베이지", "그레이", "네이비", "화이트", "소라", "브라운", "카키", "핑크"]:
        if c in text_pool and c not in colors:
            colors.append(c)
    if not colors:
        return f"{product_name}은 컬러를 딱 잘라 말씀드리기보다는 지금 보이는 옵션 기준으로 같이 보는 게 좋아요. 원하시면 차분한 쪽이 나은지, 얼굴이 덜 답답해 보이는 쪽이 나은지 기준으로 골라드릴게요 :)"
    formal = any(k in q for k in ["출근", "학교", "상담", "면접", "모임"]) or clean_text(st.session_state.get("pending_situation","")) in {"학교","출근","모임"}
    upper_heavy = any(k in q for k in ["상체", "가슴"]) or clean_text(st.session_state.get("body_top","")) in {"77","77반","88"}
    if formal:
        preferred = [c for c in colors if c in ["아이보리", "베이지", "블랙", "네이비", "그레이"]]
        picked = preferred[:2] if preferred else colors[:2]
        msg = f"{product_name}은 {', '.join(colors)} 쪽으로 보이고요. 지금처럼 단정하게 입으실 거면 {' / '.join(picked)} 쪽이 제일 무난하고 깔끔해요."
        if upper_heavy:
            msg += " 상체가 더 도드라져 보이는 걸 피하고 싶으시면 너무 강하게 튀는 색보다는 차분한 톤이 더 안전해요."
        return msg
    return f"{product_name}은 {', '.join(colors)} 쪽으로 보이고요, 고객님 체형 기준으로는 너무 강하게 튀는 색보다 차분한 톤이 더 손이 잘 가실 가능성이 커요."

def build_school_visit_coordi_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    outer_candidates = pick_recommendation_rows("자켓", user_text, product_context, db_product, limit=2)
    bottom_candidates = pick_recommendation_rows("팬츠", user_text, product_context, db_product, limit=2)
    lines = ["네, 학교 방문이면 너무 캐주얼한 것보다는 단정하게 정리되는 쪽으로 같이 골라드릴게요."]
    combined = []
    for row in outer_candidates[:2]:
        combined.append((row, "자켓"))
    for row in bottom_candidates[:2]:
        if len(combined) >= 3:
            break
        combined.append((row, "팬츠"))
    if not combined:
        return "학교 방문에 맞는 단정한 코디를 바로 많이 잡지는 못했어요. 자켓 쪽으로 볼지, 팬츠 쪽으로 볼지 먼저 정해서 같이 골라드릴게요 :)"
    for i, (row, cat) in enumerate(combined, start=1):
        lines.append(f"{i}. {clean_text(row.get('product_name',''))} ({clean_text(row.get('size_range',''))}) 🔗 — {build_style_reason(row, user_text, cat)}")
    lines.append("마음 가는 번호 말씀해주시면 그 기준으로 더 자세히 봐드릴게요 :)")
    st.session_state.last_recommendations = [row for row, _ in combined]
    st.session_state.last_selected_index = None
    st.session_state.pending_target_category = "코디"
    return "\n".join(lines)

# =========================================================
# 기타
# =========================================================
def get_fast_policy_answer(user_text: str) -> Optional[str]:
    q = clean_text(user_text).replace(" ", "")
    if any(k in q for k in ["배송비", "무료배송"]):
        return "배송비는 3,000원이고요 :) 7만원 이상이면 무료배송으로 보시면 돼요."
    if any(k in q for k in ["출고", "당일출고", "언제와", "언제와요", "배송언제"]):
        return "보통 결제 완료 후 2~4영업일 정도로 봐주시면 되고요 :) 오후 2시 이전 주문은 당일 출고 기준으로 안내드리고 있어요."
    if "교환" in q:
        return "교환은 가능해요 :) 상품 수령 후 7일 이내 접수해주시면 되고, 단순 변심 교환은 왕복 배송비 기준으로 안내드리고 있어요."
    if any(k in q for k in ["반품", "환불"]):
        return "반품도 가능해요 :) 상품 수령 후 7일 이내 접수해주시면 되고, 단순 변심 반품은 주문금액 기준에 따라 배송비가 달라질 수 있어요."
    return None

# =========================================================
# 메인 처리
# =========================================================
def process_user_message(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    started = time.time()
    try:
        q = clean_text(user_text)
        if not q:
            return ""
        update_selected_index_from_message(q)
        update_conversation_context(q)

        if is_affirmative(q):
            answer = continue_previous_flow(product_context, db_product)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_continue", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        if is_pure_greeting(q):
            return "안녕하세요 :) 지금 보시는 상품 같이 봐드릴게요. 사이즈, 코디, 비교, 컬러 중 편한 쪽부터 말씀 주세요."
        if is_followup_size_on_recommendations(q):
            answer = build_reco_followup_size_answer(q)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_reco_followup", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        if is_name_question(q):
            name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")
            answer = f"지금 보시는 상품은 {name}이에요 :)"
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_name", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        if is_compare_question(q):
            answer = build_comparison_answer(q, product_context, db_product)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_compare", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        if is_color_question(q):
            answer = build_color_style_answer(q, product_context, db_product)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_color", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        if is_selected_item_outfit_request(q):
            answer = recommend_products(q, product_context, db_product)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_selected_outfit", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        if is_coordi_request(q):
            answer = build_school_visit_coordi_answer(q, product_context, db_product)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_coordi", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        if is_recommendation_question(q):
            answer = recommend_products(q, product_context, db_product)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_recommendation", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        if is_detail_request(q) and st.session_state.get("last_recommendations"):
            answer = build_selected_item_detail_answer(q)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_selected_detail", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        policy = get_fast_policy_answer(q)
        if policy:
            write_chat_log("assistant_response", user_text=q, answer=policy, response_mode="rule_policy", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return policy
        if is_size_question(q):
            answer = build_size_answer(q, product_context, db_product)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_size", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        answer = "지금 기준으로는 사이즈, 코디, 비교, 컬러 중 어떤 쪽을 먼저 보고 싶으신지 한 가지만 잡아주시면 바로 이어서 봐드릴게요 :)"
        write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="fallback", fallback_reason="generic", latency_ms=int((time.time()-started)*1000), product_context=product_context)
        return answer
    except Exception as e:
        answer = "앗, 제가 방금 말을 매끄럽게 못 이었어요. 한 번만 더 보내주시면 바로 이어서 봐드릴게요 :)"
        write_chat_log("error", user_text=user_text, answer=answer, response_mode="error", error_text=str(e), latency_ms=int((time.time()-started)*1000), product_context=product_context)
        return answer

# =========================================================
# 컨텍스트 로드
# =========================================================
params = st.query_params
resolved_customer_name = resolve_customer_name(params)
if resolved_customer_name:
    st.session_state.customer_name = resolved_customer_name
current_url = clean_text(params.get("url", ""))
passed_product_name = clean_text(params.get("pname", ""))
passed_product_no = clean_text(params.get("pn", "")) or extract_product_no_from_url(current_url)
product_context = fetch_product_context(current_url, passed_product_name, passed_product_no)
db_product = get_db_product(product_context.get("product_no", ""))

context_key = f"{product_context.get('product_no','')}|{product_context.get('product_name','')}"
if context_key != st.session_state.get("last_context_key", ""):
    st.session_state.last_context_key = context_key
    st.session_state.messages = []
    st.session_state.last_recommendations = []
    st.session_state.last_selected_index = None
    st.session_state.active_product_override = {}
    st.session_state.pending_target_category = ""
    st.session_state.pending_situation = ""
    st.session_state.pending_style = ""

# =========================================================
# UI
# =========================================================
st.markdown("""
<style>
header[data-testid="stHeader"] {display:none;}
div[data-testid="stToolbar"] {display:none;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
.block-container{max-width:760px;padding-top:0.02rem !important;padding-bottom:6.2rem !important;}
:root{
  --miya-accent:#0f6a63;--miya-title:#303443;--miya-sub:#5f6471;--miya-muted:#8f94a3;
  --miya-divider:#ccccd2;--miya-bot-bg:#071b4e;--miya-user-bg:#dff0ec;--miya-user-text:#1f3b36;
  --miya-page-bg:#f6f7fb;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {color: var(--miya-title);background: var(--miya-page-bg) !important;}
[data-testid="stAppViewContainer"] > .main {background: var(--miya-page-bg) !important;}
.block-container{background: var(--miya-page-bg) !important;}
div[data-testid="stTextInput"] label,div[data-testid="stSelectbox"] label{color:var(--miya-title)!important;font-weight:700!important;font-size:11.5px!important;}
div[data-testid="stTextInput"] input,div[data-baseweb="select"] > div{border-radius:12px!important;}
hr{margin-top:0 !important;margin-bottom:0 !important;border-color:var(--miya-divider)!important;}
div[data-testid="stChatInput"]{position:fixed!important;left:50%!important;transform:translateX(-50%)!important;bottom:68px!important;width:min(720px, calc(100% - 24px))!important;z-index:9999!important;background:transparent!important;}
div[data-testid="stChatInput"] > div{background:transparent!important;border-radius:0!important;padding:0!important;box-shadow:none!important;border:none!important;}
div[data-testid="stChatInput"] textarea {background:#1f2740!important;color:#ffffff!important;caret-color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;font-size:16px!important;line-height:1.35!important;padding-top:12px!important;padding-bottom:12px!important;}
div[data-testid="stChatInput"] textarea::placeholder {color:#cfd6e6!important;opacity:1!important;-webkit-text-fill-color:#cfd6e6!important;}
div[data-testid="stChatInput"] [data-baseweb="textarea"] {background:#1f2740!important;border-radius:999px!important;border:1px solid rgba(255,255,255,0.08)!important;min-height:52px!important;padding:0 10px!important;display:flex!important;align-items:center!important;}
div[data-testid="stChatInput"] [data-baseweb="textarea"] > div {background:transparent!important;display:flex!important;align-items:center!important;}
div[data-testid="stChatInput"] button {background:#2f3a5f!important;color:#ffffff!important;border-radius:14px!important;}
div[data-testid="stChatInput"] button svg {fill:#ffffff!important;}
.miya-chat-wrap{padding-top:0;margin-top:-10px;padding-bottom:62px;}
.miya-row{display:flex; margin:0 0 10px 0; width:100%;}
.miya-row.assistant{justify-content:flex-start;}
.miya-row.user{justify-content:flex-end;}
.miya-msgbox{max-width:82%;}
.miya-label{font-size:12px; color:#6d7383; font-weight:700; margin-bottom:4px;}
.miya-row.user .miya-label{text-align:right;}
.miya-bubble{padding:10px 13px; border-radius:16px; line-height:1.55; font-size:14.5px; word-break:keep-all; box-shadow:none; white-space:pre-wrap;}
.miya-row.assistant .miya-bubble{background:var(--miya-bot-bg); color:#ffffff; border-top-left-radius:8px;}
.miya-row.user .miya-bubble{background:var(--miya-user-bg); color:var(--miya-user-text); border-top-right-radius:8px;}
@media (max-width: 768px){
  .block-container{max-width:100%;padding-top:0.02rem!important;padding-bottom:6.6rem!important;}
  div[data-testid="stHorizontalBlock"]{gap:6px!important;}
  div[data-testid="stHorizontalBlock"] > div{flex:1 1 0!important;min-width:0!important;}
  div[data-testid="stChatInput"]{bottom:64px!important;width:calc(100% - 16px)!important;}
  .miya-msgbox{max-width:88%;}
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center; margin:0 0 6px 0;">
      <div style="font-size:31px; font-weight:800; line-height:1.1; letter-spacing:-0.02em; color:#303443;">
        미샵 쇼핑친구 <span style="color:#0f6a63;">미야언니</span>
      </div>
      <div style="margin-top:4px; font-size:13.5px; line-height:1.35; color:#5f6471;">
        24시간 쇼핑 결정에 도움드리는 스마트한 쇼핑친구
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-top:0; margin-bottom:0;">
      <div style="font-size:13px; font-weight:700; line-height:1.2; color:#303443; margin-bottom:3px;">
        사이즈 입력<span style="font-size:11px; font-weight:500; color:#7a7f8c;">(더 구체적인 상담 가능)</span>
      </div>
      <div style="padding:4px 8px 0 8px; border:1px solid rgba(0,0,0,0.04); border-radius:14px; background:transparent;">
    """,
    unsafe_allow_html=True,
)

row1 = st.columns(2, gap="small")
with row1[0]:
    st.session_state.body_height = st.text_input("키", value=st.session_state.body_height, placeholder="cm", key="body_height_input")
with row1[1]:
    st.session_state.body_weight = st.text_input("체중", value=st.session_state.body_weight, placeholder="kg", key="body_weight_input")

size_options = ["", "44", "55", "55반", "66", "66반", "77", "77반", "88", "99"]
row2 = st.columns(2, gap="small")
with row2[0]:
    current_top = st.session_state.body_top if st.session_state.body_top in size_options else ""
    st.session_state.body_top = st.selectbox("상의", options=size_options, index=size_options.index(current_top), key="body_top_input")
with row2[1]:
    current_bottom = st.session_state.body_bottom if st.session_state.body_bottom in size_options else ""
    st.session_state.body_bottom = st.selectbox("하의", options=size_options, index=size_options.index(current_bottom), key="body_bottom_input")

st.session_state.shoe_size = st.text_input("신발사이즈(선택)", value=st.session_state.shoe_size, placeholder="예: 240", key="shoe_size_input")

st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown(f'<div style="margin-top:0; margin-bottom:0; font-size:10.8px; color:#7a7f8c;">현재 입력 정보: {html.escape(body_summary_text())}</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if not st.session_state.messages:
    if product_context.get("product_name"):
        welcome = personalize_answer(f"안녕하세요 :) {product_context.get('product_name','지금 보시는 상품')} 같이 봐드릴게요. 궁금한 걸 편하게 말씀해주시면 그 흐름대로 바로 봐드릴게요.")
    else:
        welcome = personalize_answer("안녕하세요 :) 지금은 일반 상담 모드예요. 상품 상세페이지에서 열면 그 상품 기준으로 더 정확하게 상담해드릴 수 있어요.")
    st.session_state.messages.append({"role": "assistant", "content": welcome})

def render_message(role: str, content: str):
    role_class = "assistant" if role == "assistant" else "user"
    label = "미야언니" if role == "assistant" else customer_call_name()
    safe_content = html.escape(personalize_answer(content) if role == "assistant" else content).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="miya-row {role_class}">
          <div class="miya-msgbox">
            <div class="miya-label">{label}</div>
            <div class="miya-bubble">{safe_content}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="miya-chat-wrap">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    render_message(msg.get("role", "assistant"), msg.get("content", ""))
st.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input("메시지를 입력하세요...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    write_chat_log("user_message", user_text=user_input, response_mode="user_message", product_context=product_context)
    answer = process_user_message(user_input, product_context, db_product)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_answer = answer
    st.rerun()
