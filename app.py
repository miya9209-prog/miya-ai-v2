
import os
import re
import csv
import html
import time
import datetime
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs

import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="미야언니", layout="centered", initial_sidebar_state="collapsed")

def ensure_state() -> None:
    defaults = {
        "messages": [],
        "last_context_key": "",
        "body_height": "",
        "body_weight": "",
        "body_top": "",
        "body_bottom": "",
        "last_answer": "",
        "last_recommendations": [],
        "reco_seen_names": [],
        "last_reco_target": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_state()

SIZE_ORDER = {"44": 1, "55": 2, "55반": 3, "66": 4, "66반": 5, "77": 6, "77반": 7, "88": 8, "99": 9}
SIZE_LABELS = {v: k for k, v in SIZE_ORDER.items()}
TOP_KEYWORDS = ["자켓", "재킷", "점퍼", "코트", "블라우스", "셔츠", "니트", "가디건", "맨투맨", "티셔츠", "후드", "조끼", "베스트"]
BOTTOM_KEYWORDS = ["팬츠", "슬랙스", "바지", "데님", "청바지", "스커트", "치마", "레깅스"]

def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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

def build_body_context() -> Dict[str, str]:
    return {
        "height_cm": clean_text(st.session_state.get("body_height", "")),
        "weight_kg": clean_text(st.session_state.get("body_weight", "")),
        "top_size": clean_text(st.session_state.get("body_top", "")),
        "bottom_size": clean_text(st.session_state.get("body_bottom", "")),
    }

def body_summary_text() -> str:
    vals = build_body_context()
    if not any(vals.values()):
        return "입력된 체형 정보 없음"
    return (
        f"키: {vals.get('height_cm') or '-'}cm, "
        f"체중: {vals.get('weight_kg') or '-'}kg, "
        f"상의: {vals.get('top_size') or '-'}, "
        f"하의: {vals.get('bottom_size') or '-'}"
    )

def extract_product_no_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        no = qs.get("product_no", [""])[0] or qs.get("pn", [""])[0]
        return normalize_product_no(no)
    except Exception:
        return ""

def sanitize_product_name(name: str) -> str:
    text = clean_text(name)
    bad = ["LOGIN","JOIN","MY PAGE","MYPAGE","CART","ABOUT","SHOP","COMMUNITY","TIME SALE","KRW","미샵","MISHARP","{#item","{#html","기본 정보","상품명"]
    for piece in bad:
        text = text.replace(piece, " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|/>")
    return text

def detect_category_from_name(name: str, raw_text: str = "") -> str:
    corpus = f"{clean_text(name)} {clean_text(raw_text)}"
    if "블라우스" in corpus:
        return "블라우스"
    if "셔츠" in corpus:
        return "셔츠"
    if "맨투맨" in corpus:
        return "맨투맨"
    if "니트티" in corpus or "니트 티" in corpus:
        return "니트티"
    if "티셔츠" in corpus or "티 " in corpus:
        return "티셔츠"
    if "니트" in corpus or "가디건" in corpus:
        return "니트"
    if any(k in corpus for k in ["자켓", "재킷", "점퍼", "코트", "베스트", "조끼"]):
        return "자켓"
    if any(k in corpus for k in ["팬츠", "슬랙스", "바지", "데님", "청바지"]):
        return "팬츠"
    if "스커트" in corpus or "치마" in corpus:
        return "스커트"
    return "기타"

def context_uses_top_size(product_context: Dict, db_product: Optional[Dict]) -> bool:
    corpus = " ".join([
        clean_text((db_product or {}).get("category", "")),
        clean_text((db_product or {}).get("sub_category", "")),
        clean_text((db_product or {}).get("product_name", "")),
        clean_text((product_context or {}).get("category", "")),
        clean_text((product_context or {}).get("product_name", "")),
    ])
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

def ensure_logs_dir() -> str:
    path = "logs"
    os.makedirs(path, exist_ok=True)
    return path

def write_chat_log(event_type: str, user_text: str = "", answer: str = "", response_mode: str = "", fallback_reason: str = "", error_text: str = "", latency_ms: int = 0, product_context: Optional[Dict] = None) -> None:
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

@st.cache_data(ttl=600, show_spinner=False)
def load_product_db():
    path = "misharp_miya_db.csv"
    if not os.path.exists(path):
        return []
    try:
        import pandas as pd
        df = pd.read_csv(path)
        df.columns = [clean_text(c) for c in df.columns]
        for c in df.columns:
            df[c] = df[c].fillna("").astype(str).map(clean_text)
        if "product_no" in df.columns:
            df["product_no"] = df["product_no"].map(normalize_product_no)
        return df.to_dict(orient="records")
    except Exception:
        return []

DB_ROWS = load_product_db()

def get_db_product(product_no_value: str) -> Optional[Dict]:
    if not product_no_value:
        return None
    target = normalize_product_no(product_no_value)
    for row in DB_ROWS:
        if normalize_product_no(row.get("product_no", "")) == target:
            return row
    return None

def extract_meta_name(soup: BeautifulSoup) -> str:
    candidates = []
    for selector in ['meta[property="og:title"]','meta[name="og:title"]','meta[property="twitter:title"]','meta[name="title"]']:
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
        if any(k in s for k in ["핏", "루즈", "정핏", "와이드", "커버", "복부", "허벅지", "힙", "라인", "여유", "벌룬", "루즈핏"]):
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
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
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
    return {"product_no": safe_no, "product_name": product_name, "category": category, "summary": sections["summary"], "material": sections["material"], "fit": sections["fit"], "size_tip": sections["size_tip"], "raw_excerpt": raw_text[:3000]}

def is_pure_greeting(user_text: str) -> bool:
    q = clean_text(user_text).replace(" ", "")
    return q in {"안녕", "안녕하세요", "하이", "반가워", "헬로"}

def is_size_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["사이즈", "맞을까", "맞을까요", "맞겠나", "맞나", "맞아", "핏", "작을까", "클까", "여유", "타이트"])

def is_recommendation_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["추천", "어울리는", "같이 입", "코디", "매치", "무슨 바지", "어떤 바지", "무슨 치마", "잘 어울리는", "다른", "비슷한", "학교에", "행사", "입고갈"])

def is_name_question(user_text: str) -> bool:
    q = clean_text(user_text).replace(" ", "")
    return any(k in q for k in ["이옷이름", "상품명", "상품이름", "이름뭐", "이옷이뭐야", "품명"])

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

def evaluate_size_support(user_size: str, body_label: str, product_context: Dict, db_product: Optional[Dict]) -> Dict:
    rank = size_rank(user_size)
    if not rank:
        return {"supported": None, "reason": "", "confidence": "unknown"}
    text_sources = [clean_text((db_product or {}).get("size_range", "")), clean_text((product_context or {}).get("size_tip", "")), clean_text((product_context or {}).get("summary", "")), clean_text((product_context or {}).get("fit", "")), clean_text((db_product or {}).get("fit_type", ""))]
    max_rank = None
    for src in text_sources:
        _, hi = parse_range_from_text(src)
        if hi:
            max_rank = hi if max_rank is None else max(max_rank, hi)
    fit_corpus = " ".join(text_sources)
    has_loose = any(k in fit_corpus for k in ["루즈", "여유", "오버", "벌룬"])
    has_regular = any(k in fit_corpus for k in ["정핏", "기본", "슬림"])
    if max_rank:
        if rank > max_rank:
            return {"supported": False, "reason": f"최대 {rank_to_size(max_rank)} 정도까지로 보여서 고객님 {body_label} {user_size} 기준으로는 살짝 타이트하게 느껴질 수 있어요.", "confidence": "range"}
        if rank == max_rank:
            return {"supported": "edge", "reason": f"고객님 {body_label} {user_size} 기준이면 경계선에 가까운 쪽이에요.", "confidence": "range"}
        return {"supported": True, "reason": f"고객님 {body_label} {user_size} 기준으로는 기본 가능권 안쪽으로 보여요.", "confidence": "range"}
    if has_regular:
        return {"supported": "edge", "reason": f"지금 보이는 정보로는 슬림하거나 정핏에 가까워서 고객님 {body_label} {user_size} 기준으로는 또렷하게 느껴질 수 있어요.", "confidence": "fit"}
    if has_loose:
        return {"supported": True, "reason": f"지금 보이는 정보로는 루즈한 쪽이라 답답하게 붙는 타입은 아닐 것 같아요.", "confidence": "fit"}
    return {"supported": None, "reason": "지금 정보만으로는 딱 잘라 말씀드리기보다는 실측까지 같이 보는 쪽이 더 정확해요.", "confidence": "unknown"}

def build_size_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    user_size, body_label = get_active_user_size(product_context, db_product)
    product_name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")
    if not user_size:
        return "사이즈 같이 봐드릴게요 :) 상의랑 하의 사이즈 먼저 알려주시면 더 정확하게 말씀드릴 수 있어요."
    result = evaluate_size_support(user_size, body_label, product_context, db_product)
    q = clean_text(user_text)
    fit_corpus = " ".join([clean_text((db_product or {}).get("fit_type", "")), clean_text(product_context.get("fit", "")), clean_text(product_context.get("summary", ""))])
    is_loose = any(k in fit_corpus for k in ["루즈", "여유", "오버", "벌룬"])
    looks_short = "키가 작" in q or "키가 작은" in q
    upper_heavy = "상체" in q and any(k in q for k in ["큰", "크고", "있는"])
    if context_uses_top_size(product_context, db_product) and is_loose and (upper_heavy or user_size in ["77", "77반", "88"]):
        parts = [f"고객님 {body_label} {user_size} 기준으로 보면 {product_name}은 입는 것 자체는 가능해도, 핏이 예쁘게 떨어지는 쪽은 아닐 수 있어요."]
        if upper_heavy:
            parts.append("루즈핏이라 상체가 있는 편이면 오히려 더 부해 보일 수 있고요.")
        if looks_short:
            parts.append("키가 작은 편이면 기장도 조금 더 크게 느껴질 수 있어요.")
        parts.append("편하게 툭 입는 느낌은 가능하지만, 깔끔하게 정리돼 보이는 쪽을 찾으시면 조금 더 짧거나 정돈된 핏이 더 나아요.")
        return " ".join(parts)
    parts = [f"고객님 {body_label} {user_size} 기준으로 보면 {product_name}은 {result.get('reason','무리 없는 쪽으로 보여요.')}"]
    if upper_heavy and context_uses_top_size(product_context, db_product):
        parts.append("상체가 있는 편이라고 하셔서 어깨나 가슴 쪽은 조금 더 또렷하게 느껴지실 수 있어요.")
    if looks_short and is_loose:
        parts.append("키가 작은 편이면 기장이 살짝 크게 느껴질 수도 있어요.")
    if result.get("supported") is False:
        parts.append("편하게 입으시는 기준이면 한 단계 더 여유 있는 쪽이나, 핏이 정돈된 다른 타입 같이 보시는 게 더 좋아요.")
    elif result.get("supported") == "edge":
        parts.append("입으실 수는 있는데, 여유 있게 입는 기준이면 조금 더 편한 쪽이 더 안정적이에요.")
    else:
        parts.append("전체적으로는 가능권 안쪽으로 볼 수 있어요.")
    return " ".join(parts)

def infer_target_category_from_query(user_text: str, current_product: Dict) -> str:
    q = clean_text(user_text)
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
    if "어울리는 바지" in q:
        return "팬츠"
    current_cat = clean_text(current_product.get("category", ""))
    if current_cat in ["자켓", "블라우스", "셔츠", "니트", "니트티", "맨투맨", "티셔츠"]:
        return "팬츠"
    return ""

def row_blob(rowd: Dict) -> str:
    cols = ["product_name", "category", "sub_category", "style_tags", "coordination_items", "body_cover_features", "recommended_body_type", "product_summary", "fabric", "fit_type"]
    return " ".join(clean_text(rowd.get(c, "")) for c in cols)

def row_matches_target(rowd: Dict, target_cat: str) -> bool:
    corpus = row_blob(rowd)
    name = clean_text(rowd.get("product_name", ""))
    if target_cat == "니트티":
        return ("니트" in corpus and ("티" in corpus or "티셔츠" in corpus or "탑" in corpus)) or ("니트티" in name or "니트 티" in name)
    if target_cat == "맨투맨":
        return "맨투맨" in corpus
    if target_cat == "블라우스":
        return "블라우스" in corpus
    if target_cat == "셔츠":
        return "셔츠" in corpus and "블라우스" not in corpus
    if target_cat == "니트":
        return any(k in corpus for k in ["니트", "가디건"]) and not any(k in corpus for k in ["플랫", "슈즈", "슬링백", "로퍼", "샌들", "힐"])
    if target_cat == "자켓":
        return any(k in corpus for k in ["자켓", "재킷", "점퍼", "코트", "베스트", "조끼"])
    if target_cat == "팬츠":
        return any(k in corpus for k in ["팬츠", "슬랙스", "바지", "데님", "청바지"])
    if target_cat == "스커트":
        return any(k in corpus for k in ["스커트", "치마"])
    return False

def item_supports_user(rowd: Dict, target_cat: str) -> bool:
    temp_ctx = {"product_name": clean_text(rowd.get("product_name", "")), "category": clean_text(rowd.get("sub_category", "") or rowd.get("category", "") or target_cat), "summary": clean_text(rowd.get("product_summary", "")), "fit": clean_text(rowd.get("fit_type", "")), "size_tip": clean_text(rowd.get("size_range", ""))}
    user_size, body_label = get_active_user_size(temp_ctx, rowd)
    if not user_size:
        return True
    result = evaluate_size_support(user_size, body_label, temp_ctx, rowd)
    return result.get("supported") in [True, "edge", None]

def build_reason(rowd: Dict, user_text: str, target_cat: str) -> str:
    corpus = row_blob(rowd)
    q = clean_text(user_text)
    reasons = []
    if target_cat in ["팬츠", "스커트"]:
        reasons.append("지금 보시는 상의랑 붙였을 때 전체 라인이 깔끔하게 정리되는 쪽이에요")
    else:
        if any(k in corpus for k in ["루즈", "여유", "오버"]):
            reasons.append("상체가 있는 편이어도 답답한 느낌이 덜한 편이에요")
        elif any(k in corpus for k in ["정핏", "기본", "슬림"]):
            reasons.append("너무 부해 보이지 않고 깔끔하게 잡히는 쪽이에요")
        else:
            reasons.append("전체 핏이 과하게 크지 않아 정리돼 보이기 좋아요")
    if any(k in q for k in ["학교", "행사", "학부모", "모임", "상담"]):
        reasons.append("학교에 입고 가기에도 너무 캐주얼하지 않고 단정한 쪽이에요")
    if any(k in corpus for k in ["커버", "부해", "실루엣"]):
        reasons.append("전체 실루엣이 부해 보이지 않게 잡아주는 편이에요")
    return " ".join(reasons[:2]).strip()

def recommend_products(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    current_product = {"product_name": clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "")), "category": clean_text((db_product or {}).get("category", "") or product_context.get("category", "")), "product_no": clean_text((db_product or {}).get("product_no", "") or product_context.get("product_no", ""))}
    target_cat = infer_target_category_from_query(user_text, current_product)
    if not target_cat:
        target_cat = "자켓" if current_product["category"] in ["블라우스", "셔츠", "니트", "니트티", "맨투맨", "티셔츠"] else "팬츠"
    candidates = []
    seen_names = set(st.session_state.get("reco_seen_names", []))
    q = clean_text(user_text)
    for row in DB_ROWS:
        name = clean_text(row.get("product_name", ""))
        blob = row_blob(row)
        if not name:
            continue
        if any(k in blob for k in ["플랫", "슈즈", "슬링백", "샌들", "힐", "로퍼", "백", "가방"]):
            continue
        if current_product["product_no"] and normalize_product_no(row.get("product_no", "")) == normalize_product_no(current_product["product_no"]):
            continue
        if not row_matches_target(row, target_cat):
            continue
        if name in seen_names:
            continue
        if any(k in q for k in ["학교", "행사", "학부모", "모임", "상담"]) and any(k in name for k in ["후드", "쭈리", "트레이닝"]):
            continue
        if not item_supports_user(row, target_cat):
            continue
        candidates.append(row)
    if len(candidates) < 3:
        for row in DB_ROWS:
            name = clean_text(row.get("product_name", ""))
            blob = row_blob(row)
            if not name:
                continue
            if any(k in blob for k in ["플랫", "슈즈", "슬링백", "샌들", "힐", "로퍼", "백", "가방"]):
                continue
            if current_product["product_no"] and normalize_product_no(row.get("product_no", "")) == normalize_product_no(current_product["product_no"]):
                continue
            if not row_matches_target(row, target_cat):
                continue
            if row in candidates:
                continue
            candidates.append(row)
            if len(candidates) >= 3:
                break
    if not candidates:
        return f"지금 조건에 딱 맞는 {target_cat}가 바로 많이 잡히진 않아서요. 원하시면 조금 더 단정하게 볼지, 편하게 볼지 기준을 맞춰서 다시 골라드릴게요 :)"
    picked = candidates[:3]
    st.session_state.last_recommendations = picked
    st.session_state.reco_seen_names.extend([clean_text(x.get("product_name", "")) for x in picked])
    st.session_state.last_reco_target = target_cat
    prefix = {"니트티":"네, 고객님 쪽에 잘 맞을 만한 니트티로 먼저 골라드릴게요.","맨투맨":"네, 고객님 쪽에 잘 맞을 만한 맨투맨으로 먼저 골라드릴게요.","블라우스":"네, 고객님 쪽에 잘 맞을 만한 블라우스로 먼저 골라드릴게요.","셔츠":"네, 고객님 쪽에 잘 맞을 만한 셔츠로 먼저 골라드릴게요.","니트":"네, 고객님 쪽에 잘 맞을 만한 니트 쪽으로 먼저 골라드릴게요.","자켓":"네, 고객님 쪽에 잘 맞을 만한 자켓으로 먼저 골라드릴게요.","팬츠":"네, 지금 옷이랑 잘 이어입기 좋은 바지로 먼저 골라드릴게요.","스커트":"네, 지금 옷이랑 잘 이어입기 좋은 스커트로 먼저 골라드릴게요."}.get(target_cat,"네, 같이 보기 좋은 상품으로 먼저 골라드릴게요.")
    lines = [prefix]
    for i, row in enumerate(picked, start=1):
        lines.append(f"{i}. {clean_text(row.get('product_name',''))} — {build_reason(row, user_text, target_cat)}")
    lines.append("마음 가는 번호 말씀해주시면 그 상품 기준으로 사이즈감까지 바로 이어서 봐드릴게요 :)")
    return "\\n".join(lines)

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

def is_followup_size_on_recommendations(user_text: str) -> bool:
    q = clean_text(user_text)
    return bool(st.session_state.get("last_recommendations")) and ((("추천해준" in q) and any(k in q for k in ["사이즈", "맞아", "맞을까", "괜찮아"])) or (any(k in q for k in ["그거", "그 상품", "1번", "2번", "3번", "첫 번째", "두 번째", "세 번째"]) and any(k in q for k in ["사이즈", "맞아", "맞을까", "괜찮아"])))

def build_reco_followup_size_answer(user_text: str) -> str:
    recos = st.session_state.get("last_recommendations", [])
    if not recos:
        return "지금 바로 이어서 볼 추천 상품이 없어서요 :) 먼저 보고 싶은 상품 하나 골라주시면 그 기준으로 바로 봐드릴게요."
    idx = extract_selected_index(user_text)
    if idx is not None and 0 <= idx < len(recos):
        row = recos[idx]
        temp_ctx = {"product_name": clean_text(row.get("product_name", "")), "category": clean_text(row.get("sub_category", "") or row.get("category", "") or st.session_state.get("last_reco_target", "")), "summary": clean_text(row.get("product_summary", "")), "fit": clean_text(row.get("fit_type", "")), "size_tip": clean_text(row.get("size_range", ""))}
        user_size, body_label = get_active_user_size(temp_ctx, row)
        result = evaluate_size_support(user_size, body_label, temp_ctx, row)
        return f"{idx+1}번으로 추천드린 {clean_text(row.get('product_name',''))}은 고객님 {body_label} {user_size} 기준으로 보면 {result.get('reason','무리 없는 쪽으로 보여요.')} 지금 기준으로는 무리 없이 보셔도 되는 쪽이에요."
    lines = []
    for i, row in enumerate(recos[:3], start=1):
        temp_ctx = {"product_name": clean_text(row.get("product_name", "")), "category": clean_text(row.get("sub_category", "") or row.get("category", "") or st.session_state.get("last_reco_target", "")), "summary": clean_text(row.get("product_summary", "")), "fit": clean_text(row.get("fit_type", "")), "size_tip": clean_text(row.get("size_range", ""))}
        user_size, body_label = get_active_user_size(temp_ctx, row)
        result = evaluate_size_support(user_size, body_label, temp_ctx, row)
        lines.append(f"{i}번 {clean_text(row.get('product_name',''))}은 고객님 {body_label} {user_size} 기준으로 {result.get('reason','무리 없는 쪽으로 보여요.')}")
    return "\\n".join(lines)

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

def process_user_message(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    started = time.time()
    try:
        q = clean_text(user_text)
        if not q:
            return ""
        if is_pure_greeting(q):
            return "안녕하세요 :) 지금 보시는 상품 같이 봐드릴게요. 사이즈가 궁금하신지, 코디가 궁금하신지 편하게 말씀 주세요."
        if is_followup_size_on_recommendations(q):
            answer = build_reco_followup_size_answer(q)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_reco_followup", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        if is_name_question(q):
            name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")
            answer = f"지금 보시는 상품은 {name}이에요 :)"
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_name", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        if is_recommendation_question(q):
            answer = recommend_products(q, product_context, db_product)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_recommendation", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        policy = get_fast_policy_answer(q)
        if policy:
            write_chat_log("assistant_response", user_text=q, answer=policy, response_mode="rule_policy", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return policy
        if is_size_question(q):
            answer = build_size_answer(q, product_context, db_product)
            write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="rule_size", latency_ms=int((time.time()-started)*1000), product_context=product_context)
            return answer
        answer = "같이 봐드릴게요 :) 사이즈부터 볼지, 코디부터 볼지 편하게 말씀 주세요."
        write_chat_log("assistant_response", user_text=q, answer=answer, response_mode="fallback", fallback_reason="generic", latency_ms=int((time.time()-started)*1000), product_context=product_context)
        return answer
    except Exception as e:
        answer = "앗, 제가 방금 말을 매끄럽게 못 이었어요. 한 번만 더 보내주시면 바로 이어서 봐드릴게요 :)"
        write_chat_log("error", user_text=user_text, answer=answer, response_mode="error", error_text=str(e), latency_ms=int((time.time()-started)*1000), product_context=product_context)
        return answer

params = st.query_params
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
    st.session_state.reco_seen_names = []
    st.session_state.last_reco_target = ""

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

st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown(f'<div style="margin-top:0; margin-bottom:0; font-size:10.8px; color:#7a7f8c;">현재 입력 정보: {html.escape(body_summary_text())}</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if not st.session_state.messages:
    if product_context.get("product_name"):
        welcome = "안녕하세요? 옷 같이 봐드리는 미야언니예요 :) 지금 보시는 상품 기준으로 제가 같이 봐드릴게요. 사이즈, 코디, 배송, 교환 중 뭐부터 이야기해볼까요?"
    else:
        welcome = "안녕하세요? 옷 같이 봐드리는 미야언니예요 :) 지금은 일반 상담 모드예요. 상품 상세페이지에서 채팅창을 열면 그 상품 기준으로 더 정확하게 상담해드릴 수 있어요 :) 이 창을 닫고 해당 상품 상세페이지에서 채팅창을 다시 클릭해주세요^^"
    st.session_state.messages.append({"role": "assistant", "content": welcome})

def render_message(role: str, content: str):
    role_class = "assistant" if role == "assistant" else "user"
    label = "미야언니" if role == "assistant" else "고객님"
    safe_content = html.escape(content).replace("\n", "<br>")
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
