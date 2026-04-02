
# -*- coding: utf-8 -*-
import os
import re
import csv
import json
import time
import datetime
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="미야언니", layout="centered", initial_sidebar_state="collapsed")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")) if hasattr(st, "secrets") else os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY, timeout=20.0, max_retries=1) if (OPENAI_API_KEY and OpenAI) else None

POLICY_DB = {
    "shipping": {
        "courier": "CJ 대한통운",
        "shipping_fee": 3000,
        "free_shipping_over": 70000,
        "delivery_time": "결제 후 2~4영업일 정도",
        "same_day_dispatch_rule": "오후 2시 이전 주문은 당일 출고",
    },
    "exchange_return": {
        "period": "상품 수령 후 7일 이내",
        "exchange_fee": 6000,
        "defect_wrong": "불량/오배송은 미샵 부담"
    }
}

SIZE_ORDER = {"44": 1, "55": 2, "55반": 3, "66": 4, "66반": 5, "77": 6, "77반": 7, "88": 8, "99": 9}
SIZE_LABELS = {v: k for k, v in SIZE_ORDER.items()}
APPROX_BODY_BUST = {2: 85, 3: 88, 4: 91, 5: 95, 6: 100, 7: 104, 8: 109, 9: 114}
COLOR_CANDIDATES = ["블랙", "화이트", "아이보리", "그레이", "베이지", "브라운", "네이비", "핑크", "소라", "블루", "카키", "민트", "레드", "옐로우", "크림"]

SYSTEM_PROMPT = """
너는 미샵 쇼핑친구 미야언니다.
4050 여성 고객을 옆에서 같이 봐주는 믿음 가는 MD처럼 상담한다.
내부 표현인 'DB 기준', '상품정보상', '현재 페이지 기준' 같은 말은 절대 쓰지 않는다.
""".strip()

def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def trim_text(value: str, max_len: int) -> str:
    return clean_text(value)[:max_len]

def normalize_product_no(value) -> str:
    text = clean_text(value)
    return text[:-2] if text.endswith(".0") else text

def size_rank(token: str) -> Optional[int]:
    return SIZE_ORDER.get(clean_text(token))

def rank_to_size(rank: Optional[int]) -> str:
    return SIZE_LABELS.get(rank, "") if rank else ""

def ensure_logs_dir() -> str:
    path = "logs"
    os.makedirs(path, exist_ok=True)
    return path

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
        "reco_seen_names": [],
        "last_reco_target": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
ensure_state()

def write_chat_log(event_type: str, user_text: str = "", answer: str = "", response_mode: str = "", fallback_reason: str = "", error_text: str = "", latency_ms: int = 0, product_context: Optional[Dict] = None, extra: Optional[Dict] = None) -> None:
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
            "extra": json.dumps(extra or {}, ensure_ascii=False),
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
    target = normalize_product_no(product_no_value)
    rows = DB[DB["product_no"] == target]
    if len(rows) == 0:
        return None
    return rows.iloc[0].to_dict()

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
    bad_pieces = ["LOGIN", "JOIN", "MY PAGE", "MYPAGE", "CART", "ABOUT", "SHOP", "COMMUNITY", "TIME SALE", "KRW", "미샵", "MISHARP", "{#item", "{#html", "기본 정보", "상품명"]
    for piece in bad_pieces:
        text = text.replace(piece, " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"★+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|/>")
    return text

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

def detect_category_from_name(name: str, raw_text: str = "") -> str:
    corpus = f"{clean_text(name)} {clean_text(raw_text)}"
    order = [
        ("블라우스", ["블라우스"]),
        ("맨투맨", ["맨투맨"]),
        ("셔츠", ["셔츠"]),
        ("니트", ["니트"]),
        ("가디건", ["가디건"]),
        ("자켓", ["자켓", "재킷", "점퍼", "코트", "베스트", "조끼", "아우터"]),
        ("티셔츠", ["티셔츠", "반팔", "긴팔", "탑"]),
        ("스커트", ["스커트", "치마"]),
        ("슬랙스", ["슬랙스"]),
        ("데님", ["데님", "청바지"]),
        ("팬츠", ["팬츠", "바지"]),
    ]
    for cat, words in order:
        if any(w in corpus for w in words):
            return cat
    return "기타"

def context_uses_top_size(product_context: Dict, db_product: Optional[Dict]) -> bool:
    corpus = " ".join([
        clean_text((db_product or {}).get("category", "")),
        clean_text((db_product or {}).get("sub_category", "")),
        clean_text((db_product or {}).get("product_name", "")),
        clean_text(product_context.get("category", "")),
        clean_text(product_context.get("product_name", "")),
    ])
    if any(k in corpus for k in ["자켓", "재킷", "점퍼", "코트", "셔츠", "블라우스", "니트", "가디건", "맨투맨", "티셔츠", "조끼", "베스트", "아우터"]):
        return True
    if any(k in corpus for k in ["팬츠", "슬랙스", "바지", "데님", "청바지", "스커트", "치마"]):
        return False
    return True

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
    for sentence in re.split(r"(?<=[.!?])\s+|\s*/\s*", t):
        s = clean_text(sentence)
        if not s:
            continue
        if any(k in s for k in ["면", "코튼", "폴리", "레이온", "울", "아크릴", "스판", "나일론", "혼용", "%", "소재", "원단"]):
            material.append(s)
        if any(k in s for k in ["핏", "루즈", "정핏", "와이드", "세미", "커버", "라인", "여유"]):
            fit.append(s)
        if any(k in s for k in ["사이즈", "추천", "44", "55", "66", "77", "88", "FREE", "free", "L(", "M(", "S("]):
            size_tip.append(s)
    return {"summary": t[:1400], "material": " / ".join(material)[:350], "fit": " / ".join(fit)[:350], "size_tip": " / ".join(size_tip)[:350]}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_product_context(url: str, passed_name: str = "", passed_product_no: str = "") -> Dict:
    safe_name = sanitize_product_name(passed_name)
    safe_no = normalize_product_no(passed_product_no) or extract_product_no_from_url(url)
    ctx = {"product_no": safe_no, "product_name": safe_name or "지금 보시는 상품", "category": detect_category_from_name(safe_name, ""), "summary": "", "material": "", "fit": "", "size_tip": "", "raw_excerpt": "", "colors": []}
    db_row = get_db_product(safe_no)
    if db_row:
        ctx["product_name"] = clean_text(db_row.get("product_name", "")) or ctx["product_name"]
        ctx["category"] = detect_category_from_name(ctx["product_name"], " ".join([clean_text(db_row.get("category","")), clean_text(db_row.get("sub_category",""))]))
        ctx["summary"] = clean_text(db_row.get("product_summary", ""))
        ctx["material"] = clean_text(db_row.get("fabric", ""))
        ctx["fit"] = " ".join([clean_text(db_row.get("fit_type", "")), clean_text(db_row.get("body_cover_features", ""))]).strip()
        ctx["size_tip"] = clean_text(db_row.get("size_range", ""))
        ctx["colors"] = extract_colors_from_text(" ".join([clean_text(db_row.get("color_options","")), clean_text(db_row.get("product_name",""))]))
    if not url:
        return ctx
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        if ctx["product_name"] == "지금 보시는 상품":
            ctx["product_name"] = extract_meta_name(soup) or ctx["product_name"]
        for t in soup(["script", "style", "noscript", "header", "footer"]):
            t.decompose()
        raw_text = clean_text(re.sub(r"\n{2,}", "\n", soup.get_text("\n")))
        sections = split_detail_sections(raw_text)
        if not ctx["category"] or ctx["category"] == "기타":
            ctx["category"] = detect_category_from_name(ctx["product_name"], raw_text)
        for k in ["summary", "material", "fit", "size_tip"]:
            if not ctx[k]:
                ctx[k] = sections[k]
        ctx["raw_excerpt"] = raw_text[:4000]
        if not ctx["colors"]:
            ctx["colors"] = extract_colors_from_text(raw_text)
    except Exception:
        pass
    return ctx

def build_body_context() -> Dict[str, str]:
    return {"height_cm": clean_text(st.session_state.body_height), "weight_kg": clean_text(st.session_state.body_weight), "top_size": clean_text(st.session_state.body_top), "bottom_size": clean_text(st.session_state.body_bottom)}

def build_body_context_text(body_ctx: Dict[str, str]) -> str:
    return f"키: {body_ctx.get('height_cm') or '-'}cm, 체중: {body_ctx.get('weight_kg') or '-'}kg, 상의: {body_ctx.get('top_size') or '-'}, 하의: {body_ctx.get('bottom_size') or '-'}"

def expand_size_text(size_text: str) -> List[int]:
    text = clean_text(size_text)
    if not text:
        return []
    text = text.replace("~", "-")
    found = []
    ordered = ["44", "55반", "55", "66반", "66", "77반", "77", "88", "99"]
    for token in ordered:
        if token in text:
            rank = size_rank(token)
            if rank:
                found.append(rank)
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

def parse_float_value(value) -> Optional[float]:
    text = clean_text(value).replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def infer_fit_need_cm(product_context: Dict, db_product: Optional[Dict]) -> int:
    corpus = " ".join([
        clean_text((db_product or {}).get("category", "")),
        clean_text((db_product or {}).get("sub_category", "")),
        clean_text((db_product or {}).get("product_name", "")),
        clean_text(product_context.get("category", "")),
        clean_text(product_context.get("product_name", "")),
    ])
    if any(k in corpus for k in ["코트", "자켓", "재킷", "점퍼", "패딩", "베스트", "조끼"]):
        return 14
    if any(k in corpus for k in ["맨투맨", "니트", "가디건", "후드", "티셔츠"]):
        return 10
    if any(k in corpus for k in ["블라우스", "셔츠"]):
        return 8
    return 9

def chest_support_signal(user_rank: Optional[int], product_context: Dict, db_product: Optional[Dict]) -> Dict:
    if not user_rank:
        return {"status": None, "reason": ""}
    chest = parse_float_value((db_product or {}).get("chest", ""))
    chest_type = clean_text((db_product or {}).get("chest_measure_type", "")).lower()
    if chest is None:
        raw = clean_text((db_product or {}).get("raw_measurements", ""))
        if raw:
            m = re.search(r"가슴둘레[^0-9]*(\d+(?:\.\d+)?)", raw)
            if m:
                chest = float(m.group(1)); chest_type = "circumference"
            else:
                m = re.search(r"가슴단면[^0-9]*(\d+(?:\.\d+)?)", raw)
                if m:
                    chest = float(m.group(1)); chest_type = "flat"
    if chest is None:
        return {"status": None, "reason": ""}
    garment_chest = chest * 2 if chest_type.startswith("flat") else chest
    body_bust = APPROX_BODY_BUST.get(user_rank)
    if not garment_chest or not body_bust:
        return {"status": None, "reason": ""}
    ease = garment_chest - body_bust
    needed = infer_fit_need_cm(product_context, db_product)
    if ease < max(2, needed - 5):
        return {"status": False, "reason": "가슴쪽은 여유가 크지 않아 보여요."}
    if ease < needed:
        return {"status": "edge", "reason": "가슴쪽은 딱 맞는 쪽에 가까워 보여요."}
    return {"status": True, "reason": "가슴쪽은 답답한 느낌이 덜한 편이에요."}

def get_active_user_size(product_context: Dict, db_product: Optional[Dict]) -> Tuple[str, str]:
    body = build_body_context()
    if context_uses_top_size(product_context, db_product):
        return clean_text(body.get("top_size", "")), "상의"
    return clean_text(body.get("bottom_size", "")), "하의"

def evaluate_size_support(user_size: str, product_context: Dict, db_product: Optional[Dict]) -> Dict:
    user_rank = size_rank(user_size)
    if not user_rank:
        return {"supported": None, "reason": "", "confidence": "unknown"}
    size_text = clean_text((db_product or {}).get("size_range", "")) or clean_text(product_context.get("size_tip", ""))
    ranks = expand_size_text(size_text)
    chest_signal = chest_support_signal(user_rank, product_context, db_product)
    if ranks:
        max_rank = max(ranks)
        if user_rank not in ranks:
            return {"supported": False, "reason": f"지금 느낌으로는 최대 {rank_to_size(max_rank)}까지로 보여요.", "confidence": "range"}
        if chest_signal["status"] is False:
            return {"supported": False, "reason": chest_signal["reason"], "confidence": "range+measure"}
        if user_rank >= max_rank or chest_signal["status"] == "edge":
            return {"supported": "edge", "reason": chest_signal["reason"] or "딱 맞는 쪽에 가까워 보여요.", "confidence": "range+measure"}
        return {"supported": True, "reason": chest_signal["reason"] or "편하게 입기 좋은 쪽에 가까워요.", "confidence": "range+measure"}
    if chest_signal["status"] in (True, False, "edge"):
        return {"supported": chest_signal["status"], "reason": chest_signal["reason"], "confidence": "measure"}
    return {"supported": None, "reason": "", "confidence": "unknown"}

def normalize_q(q: str) -> str:
    return clean_text(q).replace(" ", "")

def is_name_question(user_text: str) -> bool:
    q = normalize_q(user_text)
    return any(k in q for k in ["이름뭐", "상품명", "상품이름", "이옷이름", "이옷이뭐야"])

def is_color_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["컬러", "색상", "무슨 색", "어떤 색"])

def is_size_question(user_text: str) -> bool:
    q = normalize_q(user_text)
    return any(k in q for k in ["맞을까", "맞을까요", "사이즈", "핏", "작을까", "클까", "여유", "타이트", "입을수있", "입어도되", "내사이즈", "상체", "하체"])

def is_recommendation_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["추천", "어울리는", "같이 입", "코디", "매치", "다른", "비슷한", "학교", "행사", "입고 갈", "입고갈"])

def is_followup_reco_size_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return ("추천" in q and "사이즈" in q) or ("추천해준" in q and "맞" in q) or ("지금 추천해준" in q and "사이즈" in q)

def build_name_answer(product_context: Dict, db_product: Optional[Dict]) -> str:
    name = clean_text((db_product or {}).get("product_name", "")) or clean_text(product_context.get("product_name", "")) or "지금 보시는 상품"
    return f"지금 같이 보고 있는 상품은 {name}이에요 :) 궁금한 부분 있으면 바로 이어서 같이 봐드릴게요."

def build_color_answer(product_context: Dict, db_product: Optional[Dict]) -> str:
    colors = extract_colors_from_text(" ".join([clean_text((db_product or {}).get("color_options", "")), clean_text(product_context.get("raw_excerpt", ""))]))
    if not colors:
        return "지금 보이는 정보에서는 컬러가 또렷하게 정리되진 않아요. 원하시면 상세페이지 기준으로 같이 한 번 더 체크해드릴게요 :)"
    return f"이 상품은 {' / '.join(colors)} 쪽으로 같이 보시면 돼요 :)"

def get_fast_policy_answer(user_text: str) -> Optional[str]:
    q = normalize_q(user_text)
    if any(k in q for k in ["배송비", "무료배송"]):
        return f"배송비는 {POLICY_DB['shipping']['shipping_fee']:,}원이고, {POLICY_DB['shipping']['free_shipping_over']:,}원 이상이면 무료배송이에요 :)"
    if any(k in q for k in ["출고", "당일출고", "언제와", "배송언제"]):
        return f"{POLICY_DB['shipping']['same_day_dispatch_rule']} 기준이고, 보통 {POLICY_DB['shipping']['delivery_time']} 정도로 봐주시면 돼요 :)"
    if "교환" in q:
        return f"교환은 가능하고요 :) 상품 수령 후 {POLICY_DB['exchange_return']['period']} 안에 접수해주시면 되고, 왕복 {POLICY_DB['exchange_return']['exchange_fee']:,}원 기준으로 안내드리고 있어요."
    if any(k in q for k in ["반품", "환불"]):
        return f"반품도 가능해요 :) 상품 수령 후 {POLICY_DB['exchange_return']['period']} 안에 접수해주시면 되고, 불량이나 오배송은 미샵 쪽에서 부담해드려요."
    return None

def build_size_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    product_name = clean_text((db_product or {}).get("product_name", "")) or clean_text(product_context.get("product_name", "")) or "지금 보시는 상품"
    user_size, body_label = get_active_user_size(product_context, db_product)
    support = evaluate_size_support(user_size, product_context, db_product)
    fit_text = clean_text((db_product or {}).get("fit_type", "")) or clean_text(product_context.get("fit", ""))
    is_short = False
    try:
        is_short = float(clean_text(st.session_state.body_height) or 0) <= 158
    except Exception:
        pass
    if support["supported"] is True:
        lines = [f"고객님 {body_label} {user_size} 기준으로 보면 {product_name}은 무리 없이 입기 좋은 쪽이에요 :)"]
        if support["reason"]:
            lines.append(support["reason"])
        if any(k in fit_text for k in ["루즈", "여유", "오버"]):
            lines.append("상체가 있는 편이어도 너무 답답하게 붙는 타입은 아니에요.")
        if is_short and context_uses_top_size(product_context, db_product):
            lines.append("키가 작으신 편이면 기장만 살짝 여유 있게 느껴질 수는 있어요.")
        return " ".join(lines[:4])
    if support["supported"] == "edge":
        lines = [f"고객님 {body_label} {user_size} 기준이면 {product_name}은 딱 맞는 쪽에 조금 더 가까워요."]
        if support["reason"]:
            lines.append(support["reason"])
        if context_uses_top_size(product_context, db_product):
            lines.append("상체가 있는 편이라고 하셔서 어깨나 가슴 쪽은 조금 더 또렷하게 느껴지실 수 있어요.")
        lines.append("평소 편하게 입으시는 쪽이면 조금 더 여유 있는 타입을 같이 보는 게 좋아요 :)")
        return " ".join(lines[:4])
    if support["supported"] is False:
        lines = [f"고객님 {body_label} {user_size} 기준이면 {product_name}은 편하게 맞는 쪽보다는 살짝 타이트하게 느껴질 수 있어요."]
        if support["reason"]:
            lines.append(support["reason"])
        if context_uses_top_size(product_context, db_product):
            lines.append("상체가 있는 편이면 어깨나 가슴 쪽이 조금 더 또렷하게 느껴질 수 있어요.")
        lines.append("편하게 입으시는 기준이면 한 단계 더 여유 있는 쪽을 같이 보는 게 나아요 :)")
        return " ".join(lines[:4])
    return f"지금 보이는 정보만으로는 {product_name} 사이즈를 너무 단정해서 말씀드리기보다, 비슷한 핏의 다른 상품까지 같이 보는 쪽이 더 정확해요 :)"

def row_blob(rowd: Dict) -> str:
    cols = ["product_name", "category", "sub_category", "style_tags", "coordination_items", "body_cover_features", "recommended_body_type", "product_summary", "fabric", "fit_type"]
    return " ".join(clean_text(rowd.get(c, "")) for c in cols)

def infer_target_category_from_query(user_text: str, current_product: Dict) -> str:
    q = clean_text(user_text)
    if "맨투맨" in q: return "맨투맨"
    if "블라우스" in q: return "블라우스"
    if "셔츠" in q: return "셔츠"
    if "니트" in q: return "니트"
    if "가디건" in q: return "가디건"
    if any(k in q for k in ["자켓", "재킷", "아우터"]): return "자켓"
    if "스커트" in q or "치마" in q: return "스커트"
    if "슬랙스" in q: return "슬랙스"
    if "데님" in q or "청바지" in q: return "데님"
    if any(k in q for k in ["바지", "팬츠"]): return "팬츠"
    current_cat = clean_text(current_product.get("category", ""))
    if current_cat in ["자켓", "블라우스", "셔츠", "맨투맨", "니트", "가디건", "티셔츠"]:
        return "팬츠"
    if current_cat in ["팬츠", "슬랙스", "데님", "스커트"]:
        return "블라우스"
    return ""

def match_target_category(rowd: Dict, target: str) -> bool:
    if not target:
        return True
    corpus = row_blob(rowd)
    if target == "팬츠":
        return any(k in corpus for k in ["팬츠", "바지", "슬랙스", "데님", "청바지"])
    if target == "자켓":
        return any(k in corpus for k in ["자켓", "재킷", "점퍼", "코트", "베스트", "조끼", "아우터"]) and not any(k in corpus for k in ["후드 집업", "후드집업"])
    return target in corpus

def score_tpo(rowd: Dict, user_text: str) -> int:
    q = clean_text(user_text)
    blob = row_blob(rowd)
    score = 0
    if any(k in q for k in ["학교", "행사", "학부모", "상담", "모임", "출근"]):
        if any(k in blob for k in ["클래식", "오피스", "단정", "깔끔", "포멀"]):
            score += 3
        if any(k in blob for k in ["후드", "트레이닝", "스포티"]):
            score -= 3
    if "데일리" in q and any(k in blob for k in ["데일리", "편안", "캐주얼"]):
        score += 2
    return score

def build_product_reason(rowd: Dict, user_text: str, current_product: Dict) -> str:
    blob = row_blob(rowd)
    q = clean_text(user_text)
    reasons = []
    if any(k in q for k in ["학교", "행사", "학부모", "상담", "모임"]):
        reasons.append("너무 캐주얼하지 않고 깔끔하게 입기 좋은 쪽이에요")
    elif "출근" in q:
        reasons.append("출근할 때도 단정하게 이어입기 좋은 쪽이에요")
    if any(k in blob for k in ["루즈", "여유", "오버"]):
        reasons.append("상체가 있는 편이어도 답답한 느낌이 덜한 편이에요")
    elif any(k in blob for k in ["정핏", "기본"]):
        reasons.append("과하게 커 보이지 않고 깔끔하게 떨어지는 쪽이에요")
    if infer_target_category_from_query(user_text, current_product) in ["팬츠", "슬랙스", "데님"]:
        reasons.insert(0, "지금 보시는 상의랑 붙였을 때 전체 라인이 깔끔하게 정리돼요")
    if not reasons:
        reasons = ["전체 실루엣이 부해 보이지 않게 잡아주는 편이에요"]
    return " ".join(reasons[:2])

def recommend_candidates(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> List[Dict]:
    if DB.empty:
        return []
    current = {
        "product_no": normalize_product_no((db_product or {}).get("product_no", "") or product_context.get("product_no", "")),
        "product_name": clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "")),
        "category": clean_text((db_product or {}).get("category", "") or product_context.get("category", "")),
    }
    target = infer_target_category_from_query(user_text, current)
    st.session_state.last_reco_target = target
    body = build_body_context()
    active_size = body["top_size"] if target in ["맨투맨", "블라우스", "셔츠", "니트", "가디건", "자켓", "티셔츠"] else body["bottom_size"]
    seen = set(st.session_state.reco_seen_names)
    items = []
    for _, row in DB.iterrows():
        d = row.to_dict()
        name = clean_text(d.get("product_name", ""))
        if not name or name == current["product_name"] or name in seen:
            continue
        if target and not match_target_category(d, target):
            continue
        if any(k in clean_text(user_text) for k in ["학교", "행사"]) and any(k in row_blob(d) for k in ["후드", "트레이닝", "스포티"]):
            continue
        fake_ctx = {"product_name": name, "category": detect_category_from_name(name, row_blob(d))}
        support = evaluate_size_support(active_size, fake_ctx, d)
        if support["supported"] is False:
            continue
        score = score_tpo(d, user_text)
        if support["supported"] is True:
            score += 3
        elif support["supported"] == "edge":
            score += 1
        if target and target in row_blob(d):
            score += 3
        items.append((score, d))
    items.sort(key=lambda x: x[0], reverse=True)
    out = [d for _, d in items[:5]]
    st.session_state.last_recommendations = out[:3]
    st.session_state.reco_seen_names.extend([clean_text(x.get("product_name", "")) for x in out[:3]])
    return out[:3]

def build_recommendation_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    target = infer_target_category_from_query(user_text, {"product_name": clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "")), "category": clean_text((db_product or {}).get("category", "") or product_context.get("category", ""))})
    recos = recommend_candidates(user_text, product_context, db_product)
    target_word = target or "상품"
    if not recos:
        return f"지금 조건에 딱 맞는 {target_word}이 바로 많이 잡히진 않아서요. 원하시면 조금 더 단정하게 볼지, 편하게 볼지 기준을 맞춰서 다시 골라드릴게요 :)"
    lines = [f"네, 고객님 쪽에 잘 맞을 만한 {target_word}으로 먼저 골라드릴게요."]
    for i, rowd in enumerate(recos, start=1):
        lines.append(f"{i}. {clean_text(rowd.get('product_name', ''))} — {build_product_reason(rowd, user_text, product_context)}")
    lines.append("마음 가는 번호 말씀해주시면 그 상품 기준으로 사이즈감까지 바로 이어서 봐드릴게요 :)")
    return "\n".join(lines)

def build_followup_recommendation_answer(user_text: str) -> Optional[str]:
    q = clean_text(user_text)
    recos = st.session_state.get("last_recommendations", [])
    if not recos:
        return None
    m = re.search(r"([123])번", q)
    if m and any(k in q for k in ["사이즈", "맞", "입", "괜찮"]):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(recos):
            rowd = recos[idx]
            fake_ctx = {"product_name": clean_text(rowd.get("product_name","")), "category": detect_category_from_name(clean_text(rowd.get("product_name","")), row_blob(rowd))}
            size_val, body_label = get_active_user_size(fake_ctx, rowd)
            support = evaluate_size_support(size_val, fake_ctx, rowd)
            if support["supported"] is False:
                return f"{idx+1}번으로 말씀드린 {rowd.get('product_name')}은 고객님 {body_label} {size_val} 기준으로 보면 살짝 타이트하게 느껴질 수 있어요. 편하게 입으시는 기준이면 조금 더 여유 있는 쪽이 좋아요 :)"
            if support["supported"] == "edge":
                return f"{idx+1}번으로 말씀드린 {rowd.get('product_name')}은 고객님 {body_label} {size_val} 기준으로 딱 맞는 쪽에 가까워요. 정리된 느낌 좋아하시면 괜찮고, 편한 쪽 원하시면 한 단계 여유 있는 타입도 같이 볼 수 있어요 :)"
            return f"{idx+1}번으로 말씀드린 {rowd.get('product_name')}은 고객님 {body_label} {size_val} 기준으로 무리 없이 보기 좋은 쪽이에요 :) 지금 기준으로는 편하게 보셔도 되는 쪽이에요."
    if is_followup_reco_size_question(user_text):
        lines = []
        for i, rowd in enumerate(recos[:3], start=1):
            fake_ctx = {"product_name": clean_text(rowd.get("product_name","")), "category": detect_category_from_name(clean_text(rowd.get("product_name","")), row_blob(rowd))}
            size_val, body_label = get_active_user_size(fake_ctx, rowd)
            support = evaluate_size_support(size_val, fake_ctx, rowd)
            summary = "무리 없이 보기 좋은 쪽이에요" if support["supported"] is True else ("딱 맞는 쪽에 가까워요" if support["supported"] == "edge" else "살짝 타이트할 수 있어요")
            lines.append(f"{i}번 {clean_text(rowd.get('product_name',''))}은 {body_label} {size_val} 기준으로 {summary}")
        return "지금 추천드린 것들 기준으로 같이 보면요 :)\n" + "\n".join(lines[:3]) + "\n마음 가는 쪽 있으면 그 상품 기준으로 더 자세히 이어서 봐드릴게요."
    return None

def llm_can_help(user_text: str) -> bool:
    return False  # deterministic first

def safe_llm_fallback(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> str:
    if is_name_question(user_text):
        return build_name_answer(product_context, db_product)
    fast = get_fast_policy_answer(user_text)
    if fast:
        return fast
    if is_recommendation_question(user_text):
        return build_recommendation_answer(user_text, product_context, db_product)
    if is_size_question(user_text):
        return build_size_answer(user_text, product_context, db_product)
    if is_color_question(user_text):
        return build_color_answer(product_context, db_product)
    return "같이 더 정확하게 봐드리려면 궁금한 걸 한 번만 더 짧게 말씀해 주세요 :) 사이즈인지, 코디인지, 배송인지 바로 맞춰서 이어볼게요."

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
        followup_answer = build_followup_recommendation_answer(user_text)
        direct_answers = [
            (followup_answer, "followup"),
            (build_name_answer(product_context, db_product) if is_name_question(user_text) else None, "rule"),
            (get_fast_policy_answer(user_text), "rule"),
            (build_recommendation_answer(user_text, product_context, db_product) if is_recommendation_question(user_text) else None, "rule"),
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
            response_mode = "llm"
        if not answer:
            answer = safe_llm_fallback(user_text, product_context, db_product)
            response_mode = "fallback" if not response_mode else response_mode
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_answer = answer
        write_chat_log("assistant_response", user_text=user_text, answer=answer, response_mode=response_mode, latency_ms=int((time.time()-started)*1000), product_context=product_context)
    except Exception as e:
        answer = "앗, 제가 방금 말을 매끄럽게 못 이었어요. 같은 내용을 한 번만 더 보내주시면 바로 이어서 봐드릴게요 :)"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_answer = answer
        write_chat_log("error", user_text=user_text, answer=answer, response_mode="fallback", fallback_reason="exception", error_text=str(e), product_context=product_context)
    finally:
        st.session_state.is_processing = False

def render_message(role: str, content: str):
    label = "고객님" if role == "user" else "미야언니"
    with st.chat_message(role):
        st.markdown(f"**{label}**")
        st.write(content)

def get_query_value(name: str, default: str = "") -> str:
    try:
        return clean_text(st.query_params.get(name, default))
    except Exception:
        return default

def get_initial_welcome(current_url: str) -> str:
    if current_url and "/product/detail" in current_url:
        return "안녕하세요? 옷 같이 봐드리는 미야언니예요 :) \n지금 보시는 상품 기준으로 제가 같이 봐드릴게요.\n사이즈, 코디, 배송, 교환 중 뭐부터 이야기해볼까요?"
    return "안녕하세요? 옷 같이 봐드리는 미야언니예요 :) \n지금은 일반 상담 모드예요.\n상품 상세페이지에서 채팅창을 열면 그 상품 기준으로 더 정확하게 상담해드릴 수 있어요 :)\n이 창을 닫고 해당 상품 상세페이지에서 채팅창을 다시 클릭해주세요^^"

current_url = get_query_value("url", "")
passed_pn = get_query_value("pn", "")
passed_pname = get_query_value("pname", "")
product_context = fetch_product_context(current_url, passed_pname, passed_pn)
db_product = get_db_product(product_context.get("product_no", "")) if product_context.get("product_no") else None
context_key = f"{product_context.get('product_no','')}|{product_context.get('product_name','')}|{current_url}"

if st.session_state.last_context_key != context_key:
    st.session_state.last_context_key = context_key
    st.session_state.messages = [{"role": "assistant", "content": get_initial_welcome(current_url)}]
    st.session_state.last_recommendations = []
    st.session_state.reco_seen_names = []
    st.session_state.last_reco_target = ""

col1, col2 = st.columns(2)
with col1:
    st.text_input("키", key="body_height")
with col2:
    st.text_input("체중", key="body_weight")
col3, col4 = st.columns(2)
size_options = list(SIZE_ORDER.keys())
with col3:
    st.selectbox("상의", options=size_options, index=size_options.index(st.session_state.body_top) if st.session_state.body_top in size_options else 6, key="body_top")
with col4:
    st.selectbox("하의", options=size_options, index=size_options.index(st.session_state.body_bottom) if st.session_state.body_bottom in size_options else 6, key="body_bottom")
st.caption(f"현재 입력 정보: {build_body_context_text(build_body_context())}")
st.divider()

for m in st.session_state.messages:
    render_message(m["role"], m["content"])

user_input = st.chat_input("메시지를 입력하세요...")
if user_input:
    process_user_message(user_input, product_context, db_product)
    st.rerun()
