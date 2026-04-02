
import os
import re
import json
import html
import time
import csv
import datetime
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from openai import OpenAI, RateLimitError, APIError, APITimeoutError
except Exception:
    OpenAI = None
    class RateLimitError(Exception): ...
    class APIError(Exception): ...
    class APITimeoutError(Exception): ...

st.set_page_config(page_title="미야언니", layout="centered", initial_sidebar_state="collapsed")

# =========================
# 기본 상수
# =========================
POLICY_DB = {
    "shipping": {
        "courier": "CJ 대한통운",
        "shipping_fee": 3000,
        "free_shipping_over": 70000,
        "delivery_time": "결제 완료 후 2~4영업일 정도",
        "same_day_dispatch_rule": "오후 2시 이전 주문은 당일 출고",
    },
    "exchange_return": {
        "period": "상품 수령 후 7일 이내",
        "exchange_fee": 6000,
        "return_fee_rule": "단순 변심 반품은 반품 후 주문금액이 7만원 이상이면 편도 3,000원 / 7만원 미만이면 왕복 6,000원",
    },
}

SIZE_ORDER = {"44": 1, "55": 2, "55반": 3, "66": 4, "66반": 5, "77": 6, "77반": 7, "88": 8, "99": 9}
SIZE_LABELS = {v: k for k, v in SIZE_ORDER.items()}
TOP_KEYWORDS = ["자켓", "재킷", "점퍼", "코트", "셔츠", "블라우스", "니트", "가디건", "맨투맨", "티셔츠", "후드", "베스트", "조끼"]
BOTTOM_KEYWORDS = ["팬츠", "슬랙스", "바지", "데님", "청바지", "스커트", "치마"]
SUBCATEGORY_KEYWORDS = {
    "맨투맨": ["맨투맨"],
    "티셔츠": ["티셔츠", "티 ", "tee"],
    "블라우스": ["블라우스"],
    "셔츠": ["셔츠"],
    "니트": ["니트"],
    "가디건": ["가디건"],
    "자켓": ["자켓", "재킷", "아우터", "점퍼", "코트"],
    "팬츠": ["팬츠", "바지", "슬랙스", "데님", "청바지"],
    "스커트": ["스커트", "치마"],
}
COLOR_CANDIDATES = ["블랙", "화이트", "아이보리", "그레이", "베이지", "브라운", "네이비", "핑크", "소라", "블루", "카키", "민트", "레드", "옐로우"]
SAFE_PRODUCT_FALLBACK = "지금 보시는 상품"

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")) if OpenAI else ""
client = OpenAI(api_key=OPENAI_API_KEY, timeout=20.0, max_retries=1) if (OpenAI and OPENAI_API_KEY) else None

SYSTEM_PROMPT = """
너는 미샵 쇼핑친구 미야언니다.
4050 여성 고객을 옆에서 같이 봐주는 믿음 가는 MD처럼 상담한다.

반드시 지켜야 할 규칙:
1. 현재 상품명은 current_product_name만 사용한다. 모르면 '지금 보시는 상품'이라고 말한다.
2. 추천 상품명은 allowed_recommendation_candidates 안에 있는 이름만 사용한다.
3. 답변은 2~5문장, 자연스럽고 따뜻한 MD 상담체로 말한다.
4. 내부 표현(DB 기준, 상품정보상, 현재 페이지 기준)은 절대 쓰지 않는다.
5. 상의 상품이면 어깨/가슴/팔통 중심으로, 하의 상품이면 허리/힙/허벅지 중심으로만 말한다.
6. 추천 요청이면 현재 상품 설명을 반복하지 말고 바로 추천으로 들어간다.
""".strip()

# =========================
# 공통 유틸
# =========================
def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def trim_text(text: str, n: int) -> str:
    text = clean_text(text)
    return text[:n]

def normalize_product_no(value) -> str:
    text = clean_text(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text

def size_rank(token: str) -> Optional[int]:
    return SIZE_ORDER.get(clean_text(token))

def rank_to_size(rank: Optional[int]) -> str:
    if not rank:
        return ""
    return SIZE_LABELS.get(rank, "")

def expand_size_text(size_text: str) -> List[int]:
    text = clean_text(size_text)
    if not text:
        return []
    text = text.replace("~", "-")
    found: List[int] = []
    order = ["44", "55반", "55", "66반", "66", "77반", "77", "88", "99"]
    for token in order:
        if token in text:
            r = size_rank(token)
            if r:
                found.append(r)
    for a, b in re.findall(r"(44|55반|55|66반|66|77반|77|88|99)\s*-\s*(44|55반|55|66반|66|77반|77|88|99)", text):
        ra, rb = size_rank(a), size_rank(b)
        if ra and rb:
            found.extend(list(range(min(ra, rb), max(ra, rb) + 1)))
    m = re.search(r"(44|55반|55|66반|66|77반|77|88|99)\s*까지", text)
    if m:
        rb = size_rank(m.group(1))
        if rb:
            found.extend(list(range(2, rb + 1)))
    if "free" in text.lower() or text.upper() == "FREE":
        found.extend([2, 3, 4, 5, 6])
    return sorted(set(found))

def ensure_state() -> None:
    defaults = {
        "messages": [],
        "last_context_key": "",
        "body_height": "156",
        "body_weight": "68",
        "body_top": "77반",
        "body_bottom": "77반",
        "is_processing": False,
        "last_user_hash": "",
        "last_user_ts": 0.0,
        "last_answer": "",
        "last_recommendations": [],
        "reco_cursor": 0,
        "last_reco_target": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_state()

def ensure_logs_dir() -> str:
    path = "logs"
    os.makedirs(path, exist_ok=True)
    return path

def write_chat_log(event_type: str, user_text: str = "", answer: str = "", response_mode: str = "",
                   fallback_reason: str = "", error_text: str = "", latency_ms: int = 0,
                   product_context: Optional[Dict] = None) -> None:
    try:
        log_dir = ensure_logs_dir()
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        log_path = os.path.join(log_dir, f"chat_log_{date_str}.csv")
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

# =========================
# DB / 페이지 컨텍스트
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def load_product_db() -> pd.DataFrame:
    path = "misharp_miya_db.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    df.columns = [clean_text(c) for c in df.columns]
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str).map(clean_text)
    if "product_no" in df.columns:
        df["product_no"] = df["product_no"].map(normalize_product_no)
    return df

DB = load_product_db()

def get_db_product(product_no_value: str) -> Optional[Dict]:
    if DB.empty or not product_no_value or "product_no" not in DB.columns:
        return None
    rows = DB[DB["product_no"] == normalize_product_no(product_no_value)]
    if len(rows) == 0:
        return None
    return rows.iloc[0].to_dict()

def extract_product_no_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        return normalize_product_no(qs.get("product_no", [""])[0] or qs.get("pn", [""])[0])
    except Exception:
        return ""

def sanitize_product_name(name: str) -> str:
    text = clean_text(name)
    if not text:
        return ""
    bad_pieces = ["LOGIN", "JOIN", "MY PAGE", "MYPAGE", "CART", "ABOUT", "SHOP", "COMMUNITY",
                  "TIME SALE", "KRW", "미샵", "MISHARP", "{#item", "{#html", "기본 정보", "상품명"]
    for piece in bad_pieces:
        text = text.replace(piece, " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"★+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|/>")
    return text if len(text) >= 2 else ""

def extract_meta_name(soup: BeautifulSoup) -> str:
    candidates = []
    for selector in [
        'meta[property="og:title"]',
        'meta[name="og:title"]',
        'meta[property="twitter:title"]',
        'meta[name="title"]',
    ]:
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

def detect_product_subcategory(name: str, category_text: str = "", raw_text: str = "") -> str:
    corpus = f"{clean_text(name)} {clean_text(category_text)} {clean_text(raw_text)}"
    for subcat, kws in SUBCATEGORY_KEYWORDS.items():
        if any(k in corpus for k in kws):
            return subcat
    return ""

def detect_main_category(subcat: str) -> str:
    if subcat in ["팬츠", "스커트"]:
        return "하의"
    return "상의"

def extract_colors_from_text(text: str) -> List[str]:
    out = []
    for color in COLOR_CANDIDATES:
        if color in text and color not in out:
            out.append(color)
    return out

def split_detail_sections(text: str) -> Dict[str, str]:
    t = clean_text(text)
    if not t:
        return {"summary": "", "material": "", "fit": "", "size_tip": ""}
    material, fit, size_tip = [], [], []
    for s in re.split(r"(?<=[.!?])\s+|\s*/\s*", t):
        s = clean_text(s)
        if not s:
            continue
        if any(k in s for k in ["면", "코튼", "폴리", "레이온", "울", "아크릴", "스판", "나일론", "소재", "원단", "%"]):
            material.append(s)
        if any(k in s for k in ["핏", "루즈", "정핏", "와이드", "커버", "라인", "여유"]):
            fit.append(s)
        if any(k in s for k in ["사이즈", "추천", "44", "55", "66", "77", "88", "FREE", "free"]):
            size_tip.append(s)
    return {
        "summary": t[:1400],
        "material": " / ".join(material)[:300],
        "fit": " / ".join(fit)[:300],
        "size_tip": " / ".join(size_tip)[:300],
    }

@st.cache_data(ttl=300, show_spinner=False)
def fetch_product_context(url: str, passed_name: str = "", passed_product_no: str = "") -> Dict:
    safe_name = sanitize_product_name(passed_name)
    safe_no = normalize_product_no(passed_product_no) or extract_product_no_from_url(url)
    fallback_ctx = {
        "product_no": safe_no,
        "product_name": safe_name or SAFE_PRODUCT_FALLBACK,
        "category": "",
        "sub_category": "",
        "summary": "",
        "material": "",
        "fit": "",
        "size_tip": "",
        "raw_excerpt": "",
        "colors": [],
    }
    if not url:
        return fallback_ctx
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
    except Exception:
        return fallback_ctx
    soup = BeautifulSoup(r.text, "html.parser")
    product_name = safe_name or extract_meta_name(soup)
    db_row = get_db_product(safe_no)
    if db_row and clean_text(db_row.get("product_name")):
        product_name = clean_text(db_row.get("product_name"))
    for t in soup(["script", "style", "noscript", "header", "footer"]):
        t.decompose()
    raw_text = clean_text(soup.get_text("\n"))
    sections = split_detail_sections(raw_text)
    subcat = detect_product_subcategory(product_name, clean_text((db_row or {}).get("sub_category", "")), raw_text)
    return {
        "product_no": safe_no,
        "product_name": product_name or SAFE_PRODUCT_FALLBACK,
        "category": detect_main_category(subcat),
        "sub_category": subcat,
        "summary": sections["summary"],
        "material": sections["material"],
        "fit": sections["fit"],
        "size_tip": sections["size_tip"],
        "raw_excerpt": raw_text[:4000],
        "colors": extract_colors_from_text(raw_text),
    }

# =========================
# 체형 / 사이즈 판단
# =========================
def build_body_context() -> Dict[str, str]:
    return {
        "height_cm": clean_text(st.session_state.body_height),
        "weight_kg": clean_text(st.session_state.body_weight),
        "top_size": clean_text(st.session_state.body_top),
        "bottom_size": clean_text(st.session_state.body_bottom),
    }

def context_uses_top_size(product_context: Dict, db_product: Optional[Dict]) -> bool:
    name = clean_text((db_product or {}).get("product_name", "")) or clean_text(product_context.get("product_name", ""))
    category = clean_text((db_product or {}).get("category", "")) + " " + clean_text((db_product or {}).get("sub_category", "")) + " " + clean_text(product_context.get("sub_category", ""))
    corpus = f"{name} {category}"
    if any(k in corpus for k in TOP_KEYWORDS):
        return True
    if any(k in corpus for k in BOTTOM_KEYWORDS):
        return False
    return True

def get_active_user_size(product_context: Dict, db_product: Optional[Dict]) -> Tuple[str, str]:
    body = build_body_context()
    if context_uses_top_size(product_context, db_product):
        return clean_text(body.get("top_size", "")), "상의"
    return clean_text(body.get("bottom_size", "")), "하의"

def evaluate_size_support(product_context: Dict, db_product: Optional[Dict]) -> Dict:
    user_size, body_label = get_active_user_size(product_context, db_product)
    user_rank = size_rank(user_size)
    if not user_rank:
        return {"supported": None, "reason": "입력된 사이즈 정보가 아직 없어요.", "label": body_label, "size": user_size}
    raw_size = clean_text((db_product or {}).get("size_range", "")) or clean_text(product_context.get("size_tip", ""))
    ranks = expand_size_text(raw_size)
    if ranks:
        max_rank = max(ranks)
        if user_rank not in ranks:
            return {"supported": False, "reason": f"지금 느낌으로는 최대 {rank_to_size(max_rank)}까지로 보여요.", "label": body_label, "size": user_size}
        if user_rank >= max_rank:
            return {"supported": "edge", "reason": f"{body_label} {user_size} 기준으로는 경계선에 가까운 편이에요.", "label": body_label, "size": user_size}
        return {"supported": True, "reason": f"{body_label} {user_size} 기준으로는 무리 없는 쪽으로 보여요.", "label": body_label, "size": user_size}
    fit_text = clean_text((db_product or {}).get("fit_type", "")) + " " + clean_text(product_context.get("fit", ""))
    if any(k in fit_text for k in ["루즈", "여유", "오버"]):
        return {"supported": True, "reason": f"{body_label} {user_size} 기준으로 편하게 가는 쪽에 가까워요.", "label": body_label, "size": user_size}
    if any(k in fit_text for k in ["정핏", "슬림"]):
        return {"supported": "edge", "reason": f"{body_label} {user_size} 기준으로는 살짝 또렷하게 느껴질 수 있어요.", "label": body_label, "size": user_size}
    return {"supported": None, "reason": "페이지 정보만으로는 딱 잘라 말씀드리기 어려워요.", "label": body_label, "size": user_size}

# =========================
# 질문 분류
# =========================
def is_name_question(text: str) -> bool:
    q = clean_text(text).replace(" ", "")
    return any(k in q for k in ["이름", "상품명", "뭐야", "품명"])

def is_color_question(text: str) -> bool:
    q = clean_text(text)
    return any(k in q for k in ["컬러", "색상", "무슨 색", "어떤 색"] + COLOR_CANDIDATES)

def is_size_question(text: str) -> bool:
    q = clean_text(text).replace(" ", "")
    return any(k in q for k in ["사이즈", "맞을까", "맞을까요", "맞아", "핏", "타이트", "여유", "작을까", "클까", "상체", "하체"])

def is_recommendation_question(text: str) -> bool:
    q = clean_text(text)
    return any(k in q for k in ["추천", "어울리는", "같이 입", "코디", "매치", "다른", "비슷한", "학교", "행사", "입고 갈", "입고갈"])

def is_policy_question(text: str) -> bool:
    q = clean_text(text)
    return any(k in q for k in ["배송", "출고", "반품", "환불", "교환", "배송비", "무료배송"])

def get_fast_policy_answer(user_text: str) -> Optional[str]:
    q = clean_text(user_text).replace(" ", "")
    if any(k in q for k in ["배송비", "무료배송"]):
        return f"배송비는 {POLICY_DB['shipping']['shipping_fee']:,}원이고, {POLICY_DB['shipping']['free_shipping_over']:,}원 이상이면 무료배송이에요 :)"
    if any(k in q for k in ["출고", "배송", "언제와", "배송언제"]):
        return f"{POLICY_DB['shipping']['same_day_dispatch_rule']} 기준이고, 보통 {POLICY_DB['shipping']['delivery_time']} 정도로 봐주시면 돼요 :)"
    if "교환" in q:
        return f"교환은 {POLICY_DB['exchange_return']['period']} 안에 가능하고, 단순 변심은 왕복 {POLICY_DB['exchange_return']['exchange_fee']:,}원으로 안내드리고 있어요 :)"
    if any(k in q for k in ["반품", "환불"]):
        return f"반품은 {POLICY_DB['exchange_return']['period']} 안에 가능하고, {POLICY_DB['exchange_return']['return_fee_rule']} 기준으로 진행돼요 :)"
    return None

# =========================
# 직접 답변
# =========================
def build_name_answer(product_context: Dict, db_product: Optional[Dict]) -> str:
    name = clean_text((db_product or {}).get("product_name", "")) or clean_text(product_context.get("product_name", "")) or SAFE_PRODUCT_FALLBACK
    return f"지금 보시는 상품은 {name}이에요 :)"

def build_color_answer(product_context: Dict, db_product: Optional[Dict]) -> str:
    colors = product_context.get("colors", []) or []
    if not colors and db_product:
        raw = clean_text((db_product or {}).get("color_options", ""))
        colors = [c for c in COLOR_CANDIDATES if c in raw]
    if not colors:
        return "지금 보이는 정보만으로는 컬러를 딱 잘라 말씀드리기 어려워요. 옵션 쪽 같이 보면 더 정확하게 이어서 봐드릴게요 :)"
    return f"지금 확인되는 컬러는 {', '.join(colors)} 쪽으로 보여요 :)"

def build_size_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    support = evaluate_size_support(product_context, db_product)
    name = clean_text((db_product or {}).get("product_name", "")) or clean_text(product_context.get("product_name", "")) or SAFE_PRODUCT_FALLBACK
    body_label, user_size = support.get("label", "상의"), support.get("size", "")
    body_desc = clean_text(user_text)
    large_upper = any(k in body_desc for k in ["상체가 크", "상체가 좀 있", "어깨가 넓", "가슴이 있"])
    short_height = any(k in body_desc for k in ["키가 작", "키가 좀 작", "키가 작은"])
    if support["supported"] is True:
        lines = [
            f"고객님 {body_label} {user_size} 기준으로 보면 {name}은 무리 없는 쪽으로 보여요 :)",
        ]
        if context_uses_top_size(product_context, db_product):
            if large_upper:
                lines.append("상체가 있는 편이어도 너무 답답하게 붙는 타입은 아닌 쪽에 가까워요.")
            else:
                lines.append("전체적으로 편하게 입기 좋은 쪽에 가까워 보여요.")
            if short_height:
                lines.append("키가 작으신 편이면 기장은 살짝 여유 있게 느껴질 수 있지만 과하게 부담 가는 느낌은 아니에요.")
        else:
            lines.append("허리나 힙 라인도 과하게 불편하게 잡히는 느낌은 덜한 쪽이에요.")
        return " ".join(lines[:3])
    if support["supported"] == "edge":
        lines = [
            f"고객님 {body_label} {user_size} 기준으로 보면 {name}은 살짝 또렷하게 느껴질 수 있어요.",
            support["reason"],
        ]
        if context_uses_top_size(product_context, db_product):
            if large_upper:
                lines.append("상체가 있는 편이라고 하셔서 어깨나 가슴 쪽은 조금 더 또렷하게 느껴지실 수 있어요.")
            lines.append("평소 딱 맞게 입는 것보다 조금 편한 핏을 좋아하시면, 한 단계 더 여유 있는 쪽을 같이 보는 게 나아요.")
        else:
            lines.append("편하게 입으시는 기준이면 조금 더 여유 있는 쪽을 같이 보는 게 나아요.")
        return " ".join(lines[:4])
    if support["supported"] is False:
        lines = [
            f"고객님 {body_label} {user_size} 기준이면 {name}은 편하게 맞는 쪽보다는 살짝 타이트하게 느껴질 수 있어요.",
            support["reason"],
        ]
        if context_uses_top_size(product_context, db_product) and large_upper:
            lines.append("상체가 있는 편이라고 하셔서 어깨나 가슴 쪽은 조금 더 또렷하게 느껴지실 수 있어요.")
        lines.append("평소 딱 맞게 입는 것보다 조금 편한 핏을 좋아하시면, 한 단계 더 여유 있는 쪽을 같이 보는 게 나아요.")
        return " ".join(lines[:4])
    return f"지금 정보만으로는 {name} 사이즈를 딱 잘라 말씀드리기보다, 비슷한 핏의 다른 상품까지 같이 보는 쪽이 더 정확해요 :)"

# =========================
# 추천 엔진
# =========================
def row_blob(rowd: Dict) -> str:
    cols = ["product_name", "category", "sub_category", "style_tags", "coordination_items", "body_cover_features", "recommended_body_type", "product_summary", "fabric", "fit_type"]
    return " ".join(clean_text(rowd.get(c, "")) for c in cols)

def infer_target_category_from_query(user_text: str, product_context: Dict, current_product: Dict) -> str:
    q = clean_text(user_text)
    for subcat, kws in SUBCATEGORY_KEYWORDS.items():
        if any(k in q for k in kws):
            return subcat
    if any(k in q for k in ["무슨 바지", "어울리는 바지", "바지 추천", "팬츠 추천", "슬랙스 추천", "데님 추천"]):
        return "팬츠"
    if any(k in q for k in ["자켓과 바지", "자켓이랑 바지"]):
        return "팬츠"
    current_sub = clean_text(current_product.get("sub_category", "")) or clean_text(product_context.get("sub_category", ""))
    current_main = clean_text(current_product.get("category", "")) or clean_text(product_context.get("category", ""))
    if current_main == "상의":
        return "팬츠"
    if current_main == "하의":
        return "블라우스"
    return ""

def match_target_category(rowd: Dict, target: str) -> bool:
    if not target:
        return True
    corpus = row_blob(rowd)
    kws = SUBCATEGORY_KEYWORDS.get(target, [target])
    return any(k in corpus for k in kws)

def build_product_reason(rowd: Dict, user_text: str, current_product: Dict) -> List[str]:
    reasons = []
    q = clean_text(user_text)
    corpus = row_blob(rowd)
    style = clean_text(rowd.get("style_tags", ""))
    fit = clean_text(rowd.get("fit_type", ""))
    cover = clean_text(rowd.get("body_cover_features", ""))

    if any(k in q for k in ["학교", "행사", "학부모", "모임", "상담"]):
        if any(k in style for k in ["클래식", "오피스", "단정"]):
            reasons.append("학교 행사에 입기에도 너무 캐주얼하지 않고 깔끔한 쪽이에요")
        else:
            reasons.append("과하게 힘준 느낌 없이 단정하게 입기 좋은 쪽이에요")

    if any(k in corpus for k in ["자켓", "재킷"]):
        if any(k in fit for k in ["여유", "루즈", "오버"]):
            reasons.append("상체가 있는 편이어도 답답한 느낌이 덜한 편이에요")
        else:
            reasons.append("핏이 과하게 크지 않아서 단정하게 보이기 좋아요")
    elif any(k in corpus for k in ["블라우스", "셔츠"]):
        reasons.append("얼굴 쪽이 답답해 보이지 않고 깔끔하게 받쳐주기 좋아요")
    elif any(k in corpus for k in ["팬츠", "슬랙스", "데님", "스커트"]):
        reasons.append("지금 보시는 상의랑 붙였을 때 전체 라인이 깔끔하게 정리돼요")
    elif any(k in corpus for k in ["맨투맨", "티셔츠", "니트", "가디건"]):
        reasons.append("편하게 입으면서도 전체 실루엣이 둔해 보이지 않는 쪽이에요")

    if any(k in cover for k in ["상체", "가슴", "팔뚝"]):
        reasons.append("상체라인 부담을 조금 덜어주는 쪽이에요")
    elif any(k in cover for k in ["허리", "복부", "힙"]):
        reasons.append("전체 실루엣이 부해 보이지 않게 잡아주는 편이에요")

    if not reasons:
        reasons.append("지금 찾으시는 느낌으로 무난하게 손이 갈 만한 쪽이에요")
    out = []
    for r in reasons:
        if r not in out:
            out.append(r)
    return out[:2]

def recommendation_size_ok(rowd: Dict, target: str) -> bool:
    size_text = clean_text(rowd.get("size_range", ""))
    ranks = expand_size_text(size_text)
    user_top = clean_text(st.session_state.body_top)
    user_bottom = clean_text(st.session_state.body_bottom)
    if target in ["팬츠", "스커트"]:
        user_rank = size_rank(user_bottom)
    else:
        user_rank = size_rank(user_top)
    if not user_rank or not ranks:
        return True
    return user_rank in ranks or user_rank <= max(ranks)

def candidate_rows(target: str, user_text: str, current_product: Dict) -> List[Dict]:
    if DB.empty:
        return []
    rows = []
    seen = set(st.session_state.get("reco_seen_names", []))
    current_name = clean_text(current_product.get("product_name", ""))
    for _, row in DB.iterrows():
        rowd = row.to_dict()
        name = clean_text(rowd.get("product_name", ""))
        if not name or name == current_name or name in seen:
            continue
        if not match_target_category(rowd, target):
            continue
        if any(k in clean_text(user_text) for k in ["학교", "행사", "학부모"]) and any(k in row_blob(rowd) for k in ["후드", "후드집업", "트레이닝"]):
            continue
        if not recommendation_size_ok(rowd, target):
            continue
        rowd["_reasons"] = build_product_reason(rowd, user_text, current_product)
        rows.append(rowd)
    # fallback: same main category if too few
    if len(rows) < 3:
        for _, row in DB.iterrows():
            rowd = row.to_dict()
            name = clean_text(rowd.get("product_name", ""))
            if not name or name == current_name or name in seen:
                continue
            if target in ["팬츠", "스커트"]:
                if not any(k in row_blob(rowd) for k in BOTTOM_KEYWORDS):
                    continue
            else:
                if not any(k in row_blob(rowd) for k in TOP_KEYWORDS):
                    continue
            if name in [clean_text(r.get("product_name", "")) for r in rows]:
                continue
            if any(k in clean_text(user_text) for k in ["학교", "행사", "학부모"]) and any(k in row_blob(rowd) for k in ["후드", "후드집업", "트레이닝"]):
                continue
            rowd["_reasons"] = build_product_reason(rowd, user_text, current_product)
            rows.append(rowd)
            if len(rows) >= 6:
                break
    return rows

def save_recommendations(recos: List[Dict], target: str) -> None:
    cleaned = []
    for rowd in recos:
        cleaned.append({
            "product_name": clean_text(rowd.get("product_name", "")),
            "product_no": normalize_product_no(clean_text(rowd.get("product_no", ""))),
            "category": clean_text(rowd.get("category", "")),
            "sub_category": clean_text(rowd.get("sub_category", "")),
            "size_range": clean_text(rowd.get("size_range", "")),
            "reasons": rowd.get("_reasons", [])[:2],
            "_full_row": rowd,
        })
    st.session_state.last_recommendations = cleaned
    st.session_state.last_reco_target = target
    st.session_state.reco_cursor = min(3, len(cleaned))
    st.session_state.reco_seen_names = [r["product_name"] for r in cleaned[:3]]

def build_recommendation_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    current_product = {
        "product_name": clean_text((db_product or {}).get("product_name", "")) or clean_text(product_context.get("product_name", "")),
        "category": clean_text(product_context.get("category", "")),
        "sub_category": clean_text(product_context.get("sub_category", "")),
    }
    target = infer_target_category_from_query(user_text, product_context, current_product)
    rows = candidate_rows(target, user_text, current_product)
    if not rows:
        target_label = target or "상품"
        return f"지금 조건에 딱 맞는 {target_label}이 바로 많이 잡히진 않아서요. 원하시면 조금 더 단정하게 볼지, 편하게 볼지 기준 맞춰서 다시 골라드릴게요 :)"
    save_recommendations(rows, target)
    shown = st.session_state.last_recommendations[:3]
    target_label = target or "상품"
    opener = f"네, 고객님 쪽에 잘 맞을 만한 {target_label}으로 먼저 골라드릴게요."
    body_lines = []
    for i, reco in enumerate(shown, start=1):
        reasons = reco.get("reasons", [])
        body_lines.append(f"{i}. {reco['product_name']} — {' '.join(reasons)}")
    return opener + "\n" + "\n".join(body_lines) + "\n마음 가는 번호 말씀해주시면 그 상품 기준으로 사이즈감까지 바로 이어서 봐드릴게요 :)"

def get_recommendation_reference_index(user_text: str) -> Optional[int]:
    q = clean_text(user_text)
    m = re.search(r"([1-3])번", q)
    if m:
        return int(m.group(1)) - 1
    if "첫 번째" in q or "첫번째" in q:
        return 0
    if "두 번째" in q or "두번째" in q:
        return 1
    if "세 번째" in q or "세번째" in q:
        return 2
    return None

def build_followup_recommendation_answer(user_text: str) -> Optional[str]:
    recos = st.session_state.get("last_recommendations", [])
    if not recos:
        return None
    q = clean_text(user_text)
    if any(k in q for k in ["다른", "더 없어", "더 보여", "다른 건", "다른거"]):
        cursor = int(st.session_state.get("reco_cursor", 0))
        next_batch = recos[cursor:cursor+3]
        if not next_batch:
            target = st.session_state.get("last_reco_target", "") or "상품"
            return f"지금 조건에 맞춰서는 방금 보여드린 쪽이 제일 먼저 잡히는 편이었어요. 원하시면 {target} 느낌을 조금 더 단정하게 볼지, 편하게 볼지 기준 바꿔서 다시 골라드릴게요 :)"
        st.session_state.reco_cursor = cursor + len(next_batch)
        st.session_state.reco_seen_names.extend([r["product_name"] for r in next_batch])
        lines = [f"{i+1}. {r['product_name']} — {' '.join(r.get('reasons', []))}" for i, r in enumerate(next_batch)]
        return "이어서 보면 이런 쪽도 괜찮아요 :)\n" + "\n".join(lines)
    idx = get_recommendation_reference_index(user_text)
    if idx is not None and idx < len(recos):
        reco = recos[idx]
        rowd = reco.get("_full_row", {})
        target_pc = {
            "product_name": reco.get("product_name", ""),
            "category": "하의" if any(k in row_blob(rowd) for k in BOTTOM_KEYWORDS) else "상의",
            "sub_category": detect_product_subcategory(reco.get("product_name", ""), reco.get("sub_category", ""), row_blob(rowd)),
            "summary": clean_text(rowd.get("product_summary", "")),
            "material": clean_text(rowd.get("fabric", "")),
            "fit": clean_text(rowd.get("fit_type", "")),
            "size_tip": clean_text(rowd.get("size_range", "")),
            "raw_excerpt": row_blob(rowd),
            "colors": [],
        }
        ans = build_size_answer(user_text, target_pc, rowd)
        return f"{idx+1}번으로 말씀드린 {reco['product_name']}은 {ans}"
    if any(k in q for k in ["내 사이즈에 맞", "사이즈에는 맞", "사이즈는 어때", "사이즈감"]) and recos:
        lines = []
        for i, reco in enumerate(recos[:3], start=1):
            rowd = reco.get("_full_row", {})
            target_pc = {
                "product_name": reco.get("product_name", ""),
                "category": "하의" if any(k in row_blob(rowd) for k in BOTTOM_KEYWORDS) else "상의",
                "sub_category": detect_product_subcategory(reco.get("product_name", ""), reco.get("sub_category", ""), row_blob(rowd)),
                "summary": clean_text(rowd.get("product_summary", "")),
                "material": clean_text(rowd.get("fabric", "")),
                "fit": clean_text(rowd.get("fit_type", "")),
                "size_tip": clean_text(rowd.get("size_range", "")),
                "raw_excerpt": row_blob(rowd),
                "colors": [],
            }
            support = evaluate_size_support(target_pc, rowd)
            if support["supported"] is False:
                verdict = "살짝 타이트하게 느껴질 수 있어요"
            elif support["supported"] == "edge":
                verdict = "경계선에 가까운 편이에요"
            else:
                verdict = "무리 없는 쪽으로 보여요"
            lines.append(f"{i}번 {reco['product_name']}은 고객님 {support['label']} {support['size']} 기준으로 {verdict}.")
        return "\n".join(lines)
    return None

# =========================
# LLM 보조
# =========================
def slim_current_context(product_context: Dict, db_product: Optional[Dict], user_text: str) -> Dict:
    active_size, body_label = get_active_user_size(product_context, db_product)
    allowed = [r["product_name"] for r in st.session_state.get("last_recommendations", [])[:3]]
    return {
        "current_product_name": clean_text((db_product or {}).get("product_name", "")) or clean_text(product_context.get("product_name", "")) or SAFE_PRODUCT_FALLBACK,
        "current_product_category": clean_text(product_context.get("sub_category", "")) or clean_text(product_context.get("category", "")),
        "body_label": body_label,
        "body_size": active_size,
        "body_height": clean_text(st.session_state.body_height),
        "body_weight": clean_text(st.session_state.body_weight),
        "current_product_fit": trim_text(product_context.get("fit", ""), 220),
        "current_product_material": trim_text(product_context.get("material", ""), 180),
        "current_product_summary": trim_text(product_context.get("summary", ""), 250),
        "confirmed_colors": product_context.get("colors", [])[:5],
        "allowed_recommendation_candidates": allowed,
        "confirmed_size_support": evaluate_size_support(product_context, db_product).get("supported") not in [False, None],
    }

def llm_can_help(user_text: str) -> bool:
    if is_name_question(user_text) or is_size_question(user_text) or is_color_question(user_text) or is_recommendation_question(user_text) or is_policy_question(user_text):
        return False
    return client is not None

def call_llm(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    if client is None:
        return None
    pack = slim_current_context(product_context, db_product, user_text)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": "참고 데이터(JSON):\n" + json.dumps(pack, ensure_ascii=False)}]
    for m in st.session_state.messages[-4:]:
        messages.append({"role": m["role"], "content": trim_text(m["content"], 280)})
    messages.append({"role": "user", "content": trim_text(user_text, 300)})
    last_error = None
    for wait in (0, 1.0):
        if wait:
            time.sleep(wait)
        try:
            resp = client.chat.completions.create(model="gpt-4.1-mini", messages=messages, temperature=0.35, max_tokens=220)
            return clean_text(resp.choices[0].message.content or "")
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break
    print(f"[MIYA LLM ERROR] {type(last_error).__name__}: {last_error}")
    return None

def safe_llm_fallback(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    for builder in [
        lambda: build_name_answer(product_context, db_product) if is_name_question(user_text) else None,
        lambda: get_fast_policy_answer(user_text),
        lambda: build_followup_recommendation_answer(user_text),
        lambda: build_recommendation_answer(user_text, product_context, db_product) if is_recommendation_question(user_text) else None,
        lambda: build_size_answer(user_text, product_context, db_product) if is_size_question(user_text) else None,
        lambda: build_color_answer(product_context, db_product) if is_color_question(user_text) else None,
    ]:
        ans = builder()
        if ans:
            return ans
    return "지금 문의가 잠시 몰려서 답변 연결이 늦어지고 있어요. 같은 내용을 잠깐 뒤 한 번만 다시 보내주시면 바로 이어서 도와드릴게요 :)"

# =========================
# 메인 처리
# =========================
def process_user_message(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> None:
    user_hash = str(hash(clean_text(user_text)))
    now = time.time()
    if st.session_state.is_processing:
        return
    if user_hash == st.session_state.last_user_hash and now - st.session_state.last_user_ts < 4 and st.session_state.last_answer:
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.last_answer})
        return

    st.session_state.last_user_hash = user_hash
    st.session_state.last_user_ts = now
    st.session_state.is_processing = True
    st.session_state.messages.append({"role": "user", "content": user_text})
    write_chat_log("user_message", user_text=user_text, product_context=product_context)
    started = time.time()
    try:
        direct_answers = [
            (build_followup_recommendation_answer(user_text), "followup"),
            (build_recommendation_answer(user_text, product_context, db_product) if is_recommendation_question(user_text) else None, "rule"),
            (build_name_answer(product_context, db_product) if is_name_question(user_text) else None, "rule"),
            (get_fast_policy_answer(user_text), "rule"),
            (build_size_answer(user_text, product_context, db_product) if is_size_question(user_text) else None, "rule"),
            (build_color_answer(product_context, db_product) if is_color_question(user_text) else None, "rule"),
        ]
        answer = None
        response_mode = ""
        for candidate, mode in direct_answers:
            if candidate:
                answer = candidate
                response_mode = mode
                break
        if not answer and llm_can_help(user_text):
            answer = call_llm(user_text, product_context, db_product)
            if answer:
                response_mode = "llm"
        if not answer:
            answer = safe_llm_fallback(user_text, product_context, db_product)
            response_mode = "fallback"
        st.session_state.last_answer = answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        write_chat_log("assistant_response", user_text=user_text, answer=answer, response_mode=response_mode,
                       latency_ms=int((time.time() - started) * 1000), product_context=product_context)
    except Exception as e:
        answer = "앗, 지금 연결이 잠깐 흔들렸어요. 같은 내용을 한 번만 더 보내주시면 바로 이어서 봐드릴게요 :)"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        write_chat_log("error", user_text=user_text, answer=answer, response_mode="error",
                       error_text=repr(e), latency_ms=int((time.time() - started) * 1000), product_context=product_context)
    finally:
        st.session_state.is_processing = False

# =========================
# 화면 렌더링
# =========================
st.markdown("""
<style>
.block-container{padding-top:0.7rem;padding-bottom:0.4rem;max-width:760px;}
div[data-testid="stChatInput"] {position: fixed; bottom: 20px; left: calc(50% - 310px); width: 620px; background: white; z-index:999;}
@media (max-width:900px){
  div[data-testid="stChatInput"] {left: 1rem; right: 1rem; width: auto;}
}
</style>
""", unsafe_allow_html=True)

qp = st.query_params
current_url = clean_text(qp.get("url", "")) if hasattr(qp, "get") else ""
passed_pname = clean_text(qp.get("pname", "")) if hasattr(qp, "get") else ""
passed_pn = normalize_product_no(clean_text(qp.get("pn", ""))) if hasattr(qp, "get") else ""
product_context = fetch_product_context(current_url, passed_pname, passed_pn)
db_product = get_db_product(product_context.get("product_no", ""))

ctx_key = json.dumps({"url": current_url, "pn": product_context.get("product_no", ""), "name": product_context.get("product_name", "")}, ensure_ascii=False)
if st.session_state.last_context_key != ctx_key:
    st.session_state.last_context_key = ctx_key
    st.session_state.messages = []
    st.session_state.last_recommendations = []
    st.session_state.reco_cursor = 0
    st.session_state.reco_seen_names = []

col1, col2 = st.columns(2)
with col1:
    st.text_input("키", key="body_height")
with col2:
    st.text_input("체중", key="body_weight")
col3, col4 = st.columns(2)
with col3:
    st.selectbox("상의", options=list(SIZE_ORDER.keys()), index=list(SIZE_ORDER.keys()).index(st.session_state.body_top) if st.session_state.body_top in SIZE_ORDER else 6, key="body_top")
with col4:
    st.selectbox("하의", options=list(SIZE_ORDER.keys()), index=list(SIZE_ORDER.keys()).index(st.session_state.body_bottom) if st.session_state.body_bottom in SIZE_ORDER else 6, key="body_bottom")

st.caption(f"현재 입력 정보: 키: {clean_text(st.session_state.body_height)}cm, 체중: {clean_text(st.session_state.body_weight)}kg, 상의: {clean_text(st.session_state.body_top)}, 하의: {clean_text(st.session_state.body_bottom)}")
st.divider()

if not st.session_state.messages:
    if current_url and product_context.get("product_name"):
        welcome = f"안녕하세요? 옷 같이 봐드리는 미야언니예요 :) 지금 보시는 상품 기준으로 제가 같이 봐드릴게요. 사이즈, 코디, 배송, 교환 중 뭐부터 이야기해볼까요?"
    else:
        welcome = ("안녕하세요? 옷 같이 봐드리는 미야언니예요 :)\n"
                   "지금은 일반 상담 모드예요.\n"
                   "상품 상세페이지에서 채팅창을 열면 그 상품 기준으로 더 정확하게 상담해드릴 수 있어요 :)\n"
                   "이 창을 닫고 해당 상품 상세페이지에서 채팅창을 다시 클릭해주세요^^")
    st.session_state.messages.append({"role": "assistant", "content": welcome})

for msg in st.session_state.messages:
    safe_text = html.escape(msg["content"]).replace("\n", "<br>")
    if msg["role"] == "user":
        st.markdown(
            '<div style="display:flex; justify-content:flex-end; width:100%; margin:2px 0 4px 0;">'
            '<div style="max-width:92%;">'
            '<div style="display:block; font-size:12px; font-weight:700; line-height:1.15; color:#0f6a63; text-align:right; margin:0 6px 1px 0;">고객님</div>'
            f'<div style="padding:10px 14px 10px 10px; border-radius:18px; border-bottom-right-radius:6px; font-size:15px; line-height:1.5; white-space:pre-wrap; word-break:keep-all; background:#dff0ec; color:#1f3b36; border:1px solid rgba(15,106,99,0.14);">{safe_text}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:flex; justify-content:flex-start; width:100%; margin:2px 0 4px 0;">'
            '<div style="max-width:92%;">'
            '<div style="display:block; font-size:12px; font-weight:700; line-height:1.15; color:#505767; text-align:left; margin:0 0 1px 6px;">미야언니</div>'
            f'<div style="padding:10px 10px 10px 14px; border-radius:18px; border-bottom-left-radius:6px; font-size:15px; line-height:1.6; white-space:pre-wrap; word-break:keep-all; background:#071b4e; color:#ffffff;">{safe_text}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
st.write("")
st.write("")

user_input = st.chat_input("메시지를 입력하세요.")
if user_input:
    process_user_message(user_input, product_context, db_product)
    st.rerun()
