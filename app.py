import os
import re
import json
import html
import time
import datetime
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

st.set_page_config(page_title="미야언니", layout="centered", initial_sidebar_state="collapsed")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY가 필요합니다. Streamlit Secrets에 OPENAI_API_KEY를 추가해주세요.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=25.0, max_retries=1)

POLICY_DB = {
    "shipping": {
        "courier": "CJ 대한통운",
        "shipping_fee": 3000,
        "free_shipping_over": 70000,
        "delivery_time": "결제 완료 후 2~4영업일 정도",
        "same_day_dispatch_rule": "오후 2시 이전 주문은 당일 출고",
        "reservation_product": "예약상품 개념 없음",
        "combined_shipping": "합배송 가능(1박스 기준, 박스 크기 초과 시 불가)",
        "dispatch_order": "결제 순서대로 순차 출고",
        "jeju": "제주 및 도서산간 지역은 추가배송비가 자동 부과됩니다."
    },
    "exchange_return": {
        "exchange_possible": "사이즈 교환 가능 / 동일상품 교환 가능 / 타상품 교환 가능",
        "period": "상품 수령 후 7일 이내",
        "exchange_fee": 6000,
        "return_fee_rule": "단순 변심 반품은 반품 후 주문금액이 7만원 이상이면 편도 3,000원 / 7만원 미만이면 왕복 6,000원",
        "defect_wrong": "불량/오배송은 미샵 부담"
    }
}

SIZE_ORDER = {"44": 1, "55": 2, "55반": 3, "66": 4, "66반": 5, "77": 6, "77반": 7, "88": 8, "99": 9}
SIZE_LABELS = {v: k for k, v in SIZE_ORDER.items()}
APPROX_BODY_BUST = {2: 85, 3: 88, 4: 91, 5: 95, 6: 100, 7: 104, 8: 109, 9: 114}
BOTTOM_CATS = ["팬츠", "슬랙스", "데님", "청바지", "바지", "스커트", "치마"]
TOP_CATS = ["티셔츠", "셔츠", "블라우스", "니트", "가디건", "맨투맨", "자켓", "재킷", "점퍼", "코트", "베스트", "조끼"]
COLOR_CANDIDATES = ["블랙", "화이트", "아이보리", "그레이", "베이지", "브라운", "네이비", "핑크", "소라", "블루", "카키", "민트", "레드", "옐로우"]

SYSTEM_PROMPT = """
너는 미샵 쇼핑친구 미야언니다.
4050 여성 고객을 옆에서 같이 봐주는 믿음 가는 MD처럼 상담한다.

반드시 지켜야 할 규칙:
1. 현재 상품명은 current_product_name에 들어있는 이름만 사용한다. 모르면 '지금 보시는 상품'이라고 말한다.
2. 추천 상품명은 allowed_recommendation_candidates에 들어있는 이름만 사용한다. 없는 상품명을 절대 만들지 않는다.
3. 사이즈는 confirmed_size_support가 false면 추천한다고 말하지 않는다.
4. 컬러는 confirmed_colors 안에 있는 것만 말한다.
5. 데이터가 부족하면 추측하지 말고 짧고 솔직하게 말한다.
6. 답변은 3~6문장, 자연스러운 MD 상담체, 먼저 결론부터 말한다.
7. 메뉴명, 로그인 텍스트, 사이트 네비게이션 같은 잡텍스트를 상품정보로 취급하지 않는다.
8. 자켓/재킷/점퍼/코트/셔츠/블라우스/니트/가디건/맨투맨/티셔츠는 상의로 보고 top_size만 사용한다.
9. 팬츠/슬랙스/바지/데님/청바지/스커트/치마는 하의로 보고 bottom_size만 사용한다.
10. 상의 상품에서는 허리, 힙, 허벅지 같은 하의 중심 표현을 쓰지 않는다.
11. 하의 상품에서는 어깨, 가슴, 팔통 같은 상의 중심 표현을 쓰지 않는다.
12. 추천 요청이 들어오면 현재 상품 설명을 반복하지 말고, 요청한 카테고리와 맞는 추천만 보여준다.
""".strip()


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\xa0", " ")
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


def expand_size_text(size_text: str) -> List[int]:
    text = clean_text(size_text)
    if not text:
        return []
    text = text.replace("~", "-")
    found: List[int] = []
    ordered = ["44", "55반", "55", "66반", "66", "77반", "77", "88", "99"]
    for token in ordered:
        if token in text:
            rank = size_rank(token)
            if rank:
                found.append(rank)
    for a, b in re.findall(r"(44|55반|55|66반|66|77반|77|88|99)\s*-\s*(44|55반|55|66반|66|77반|77|88|99)", text):
        ra, rb = size_rank(a), size_rank(b)
        if ra and rb:
            start, end = min(ra, rb), max(ra, rb)
            found.extend(list(range(start, end + 1)))
    m = re.search(r"(44|55반|55|66반|66|77반|77|88|99)\s*까지", text)
    if m:
        rb = size_rank(m.group(1))
        if rb:
            found.extend(list(range(2, rb + 1)))
    if "free" in text.lower() or "f(" in text.lower() or text.upper() == "FREE":
        if 2 not in found:
            found.extend([2, 3, 4, 5, 6])
    return sorted(set(found))


def ensure_state() -> None:
    defaults = {
        "messages": [],
        "last_context_key": "",
        "body_height": "",
        "body_weight": "",
        "body_top": "",
        "body_bottom": "",
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
        import csv
        exists = os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass


def context_uses_top_size(product_context: Dict, db_product: Optional[Dict]) -> bool:
    name = clean_text((db_product or {}).get("product_name", "")) or clean_text(product_context.get("product_name", ""))
    category = clean_text((db_product or {}).get("category", "")) or clean_text(product_context.get("category", ""))
    sub_category = clean_text((db_product or {}).get("sub_category", ""))
    corpus = f"{name} {category} {sub_category}"

    if any(k in corpus for k in ["자켓", "재킷", "점퍼", "코트", "셔츠", "블라우스", "니트", "가디건", "맨투맨", "티셔츠", "베스트", "조끼"]):
        return True
    if any(k in corpus for k in ["팬츠", "슬랙스", "바지", "데님", "청바지", "스커트", "치마"]):
        return False
    return True

def get_active_user_size(product_context: Dict, db_product: Optional[Dict]) -> Tuple[str, str]:
    body = build_body_context()
    if context_uses_top_size(product_context, db_product):
        return clean_text(body.get("top_size", "")), "상의"
    return clean_text(body.get("bottom_size", "")), "하의"


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
        no = qs.get("product_no", [""])[0] or qs.get("pn", [""])[0]
        return normalize_product_no(no)
    except Exception:
        return ""


def sanitize_product_name(name: str) -> str:
    text = clean_text(name)
    if not text:
        return ""
    bad_pieces = [
        "LOGIN", "JOIN", "MY PAGE", "MYPAGE", "CART", "ABOUT", "SHOP", "COMMUNITY",
        "TIME SALE", "KRW", "미샵", "MISHARP", "{#item", "{#html", "기본 정보", "상품명"
    ]
    for piece in bad_pieces:
        text = text.replace(piece, " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"★+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|/>")
    if len(text) < 3:
        return ""
    return text


def extract_meta_name(soup: BeautifulSoup) -> str:
    candidates: List[str] = []
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


def detect_category_from_name(name: str, raw_text: str) -> str:
    corpus = f"{clean_text(name)} {clean_text(raw_text)}"
    mapping = {
        "팬츠": ["슬랙스", "팬츠", "바지", "데님", "청바지", "배기핏"],
        "스커트": ["스커트", "치마"],
        "블라우스": ["블라우스"],
        "셔츠": ["셔츠"],
        "티셔츠": ["티셔츠", "맨투맨", "탑"],
        "니트": ["니트", "가디건"],
        "자켓": ["자켓", "재킷", "점퍼", "트렌치", "코트", "베스트", "조끼"],
        "원피스": ["원피스"],
    }
    for cat, words in mapping.items():
        if any(w in corpus for w in words):
            return cat
    return "기타"


def extract_colors_from_text(text: str) -> List[str]:
    out: List[str] = []
    for color in COLOR_CANDIDATES:
        if color in text and color not in out:
            out.append(color)
    return out


def split_detail_sections(text: str) -> Dict[str, str]:
    t = clean_text(text)
    if not t:
        return {"summary": "", "material": "", "fit": "", "size_tip": ""}
    material = []
    fit = []
    size_tip = []
    for sentence in re.split(r"(?<=[.!?])\s+|\s*/\s*", t):
        s = clean_text(sentence)
        if not s:
            continue
        if any(k in s for k in ["면", "코튼", "폴리", "레이온", "울", "아크릴", "스판", "나일론", "혼용", "%", "소재", "원단"]):
            material.append(s)
        if any(k in s for k in ["핏", "루즈", "정핏", "와이드", "세미", "커버", "복부", "허벅지", "힙", "라인", "여유"]):
            fit.append(s)
        if any(k in s for k in ["사이즈", "추천", "44", "55", "66", "77", "88", "FREE", "free", "L(", "M(", "S("]):
            size_tip.append(s)
    return {
        "summary": t[:1400],
        "material": " / ".join(material)[:350],
        "fit": " / ".join(fit)[:350],
        "size_tip": " / ".join(size_tip)[:350],
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_product_context(url: str, passed_name: str = "", passed_product_no: str = "") -> Dict:
    safe_name = sanitize_product_name(passed_name)
    safe_no = normalize_product_no(passed_product_no) or extract_product_no_from_url(url)
    fallback_ctx = {
        "product_no": safe_no,
        "product_name": safe_name or "지금 보시는 상품",
        "category": "기타",
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
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
    except Exception:
        return fallback_ctx

    soup = BeautifulSoup(r.text, "html.parser")
    meta_name = extract_meta_name(soup)
    product_name = safe_name or meta_name

    # Prefer DB name when product_no matches.
    db_row = get_db_product(safe_no)
    if db_row and clean_text(db_row.get("product_name")):
        product_name = clean_text(db_row.get("product_name"))

    for t in soup(["script", "style", "noscript", "header", "footer"]):
        t.decompose()

    raw_text = soup.get_text("\n")
    raw_text = raw_text.replace("\r", "\n")
    raw_text = re.sub(r"\n{2,}", "\n", raw_text)
    raw_text = clean_text(raw_text)
    sections = split_detail_sections(raw_text)
    colors = extract_colors_from_text(raw_text)
    category = detect_category_from_name(product_name, raw_text)

    if not product_name:
        product_name = "지금 보시는 상품"

    return {
        "product_no": safe_no,
        "product_name": product_name,
        "category": category,
        "summary": sections["summary"],
        "material": sections["material"],
        "fit": sections["fit"],
        "size_tip": sections["size_tip"],
        "raw_excerpt": raw_text[:4000],
        "colors": colors,
    }


def build_body_context() -> Dict[str, str]:
    return {
        "height_cm": clean_text(st.session_state.body_height),
        "weight_kg": clean_text(st.session_state.body_weight),
        "top_size": clean_text(st.session_state.body_top),
        "bottom_size": clean_text(st.session_state.body_bottom),
    }


def build_body_context_text(body_ctx: Dict[str, str]) -> str:
    if not any(body_ctx.values()):
        return "입력된 체형 정보 없음"
    return (
        f"키: {body_ctx.get('height_cm') or '-'}cm, 체중: {body_ctx.get('weight_kg') or '-'}kg, "
        f"상의: {body_ctx.get('top_size') or '-'}, 하의: {body_ctx.get('bottom_size') or '-'}"
    )


def get_fast_policy_answer(user_text: str) -> Optional[str]:
    q = clean_text(user_text).replace(" ", "")
    if any(k in q for k in ["배송비", "무료배송"]):
        return (
            f"배송은 {POLICY_DB['shipping']['courier']}를 이용하고 있고요 :)\n"
            f"배송비는 {POLICY_DB['shipping']['shipping_fee']:,}원이고, "
            f"{POLICY_DB['shipping']['free_shipping_over']:,}원 이상이면 무료배송이에요."
        )
    if any(k in q for k in ["출고", "당일출고", "언제와", "언제와요", "배송언제"]):
        return (
            f"{POLICY_DB['shipping']['same_day_dispatch_rule']}예요 :)\n"
            f"보통은 {POLICY_DB['shipping']['delivery_time']} 정도로 봐주시면 되고, 결제 순서대로 순차 출고되고 있어요."
        )
    if "교환" in q:
        return (
            "교환은 가능해요 :)\n"
            f"{POLICY_DB['exchange_return']['period']} 안에 접수해주시면 되고, "
            f"단순 변심 교환은 왕복 {POLICY_DB['exchange_return']['exchange_fee']:,}원으로 안내드리고 있어요."
        )
    if any(k in q for k in ["반품", "환불"]):
        return (
            "반품도 가능해요 :)\n"
            f"{POLICY_DB['exchange_return']['period']} 안에 접수해주시면 되고, "
            f"{POLICY_DB['exchange_return']['return_fee_rule']} 기준으로 진행돼요."
        )
    return None


def parse_color_options(product_context: Dict, db_product: Optional[Dict]) -> List[str]:
    colors: List[str] = []
    for source in [clean_text((db_product or {}).get("color_options", "")), clean_text((product_context or {}).get("raw_excerpt", ""))]:
        for part in re.split(r"[;,/|]", source):
            token = clean_text(part)
            if token in COLOR_CANDIDATES and token not in colors:
                colors.append(token)
        for token in extract_colors_from_text(source):
            if token not in colors:
                colors.append(token)
    return colors


def parse_page_size_options(product_context: Dict, db_product: Optional[Dict]) -> List[Dict]:
    text = " ".join([
        clean_text((product_context or {}).get("size_tip", "")),
        clean_text((product_context or {}).get("summary", "")),
        clean_text((db_product or {}).get("size_range", "")),
    ])
    options: List[Dict] = []
    seen = set()
    for pat in [
        r"([A-Za-z가-힣]+)\s*\((44|55반|55|66반|66|77반|77|88|99)\s*-\s*(44|55반|55|66반|66|77반|77|88|99)\)",
        r"([A-Za-z가-힣]+)\s*\((44|55반|55|66반|66|77반|77|88|99)\)",
    ]:
        for match in re.finditer(pat, text):
            label = clean_text(match.group(1)).upper()
            if label in {"COLOR", "SIZE", "OPTION", "옵션", "컬러"}:
                continue
            if len(match.groups()) == 3:
                size_desc = f"{match.group(2)}-{match.group(3)}"
            else:
                size_desc = match.group(2)
            ranks = expand_size_text(size_desc)
            if ranks:
                key = (label, tuple(ranks))
                if key not in seen:
                    seen.add(key)
                    options.append({"label": label, "size_desc": size_desc, "ranks": ranks})
    return options

def parse_float_value(value) -> Optional[float]:
    text = clean_text(value).replace(",", "")
    if not text:
        return None
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
                chest = float(m.group(1))
                chest_type = "circumference"
            else:
                m = re.search(r"가슴단면[^0-9]*(\d+(?:\.\d+)?)", raw)
                if m:
                    chest = float(m.group(1))
                    chest_type = "flat"
    if chest is None:
        return {"status": None, "reason": ""}
    garment_chest = chest * 2 if chest_type.startswith("flat") else chest
    body_bust = APPROX_BODY_BUST.get(user_rank)
    if not garment_chest or not body_bust:
        return {"status": None, "reason": ""}
    ease = garment_chest - body_bust
    needed = infer_fit_need_cm(product_context, db_product)
    if ease < max(2, needed - 5):
        return {"status": False, "reason": f"가슴둘레 기준으로는 여유가 크지 않아 보여요(의류 약 {int(round(garment_chest))}cm)."}
    if ease < needed:
        return {"status": "edge", "reason": f"가슴둘레 기준으로는 경계선에 가까워요(의류 약 {int(round(garment_chest))}cm)."}
    return {"status": True, "reason": f"가슴둘레 기준으로는 여유가 있는 편이에요(의류 약 {int(round(garment_chest))}cm)."}



def evaluate_size_support(user_size_text: str, product_context: Dict, db_product: Optional[Dict]) -> Dict:
    user_rank = size_rank(user_size_text)
    if not user_rank:
        return {"supported": None, "reason": "", "matched_option": None, "confidence": "unknown"}

    page_options = parse_page_size_options(product_context, db_product)
    chest_signal = chest_support_signal(user_rank, product_context, db_product)

    if page_options:
        all_ranks = sorted({r for opt in page_options for r in opt["ranks"]})
        max_rank = max(all_ranks) if all_ranks else None
        matched = None
        for opt in page_options:
            if user_rank in opt["ranks"]:
                matched = opt
                break
        if not matched:
            return {
                "supported": False,
                "reason": f"최대 {rank_to_size(max_rank)}까지로 보여요." if max_rank else "지금 보이는 옵션 범위가 고객님 사이즈보다 작아 보여요.",
                "matched_option": None,
                "confidence": "page",
            }
        boundary = max_rank is not None and user_rank >= max_rank
        if chest_signal.get("status") is False:
            return {
                "supported": False,
                "reason": chest_signal.get("reason", "") or f"{matched['label']} 기준으로는 타이트할 수 있어요.",
                "matched_option": matched,
                "confidence": "page+measure",
            }
        if boundary or chest_signal.get("status") == "edge":
            return {
                "supported": "edge",
                "reason": chest_signal.get("reason", "") or f"{matched['label']} 기준으로 상단 경계에 가까워 보여요.",
                "matched_option": matched,
                "confidence": "page+measure",
            }
        return {
            "supported": True,
            "reason": chest_signal.get("reason", "") or f"{matched['label']} 기준으로는 무리 없는 쪽으로 보여요.",
            "matched_option": matched,
            "confidence": "page+measure",
        }

    db_range = clean_text((db_product or {}).get("size_range", ""))
    ranks = expand_size_text(db_range)
    if ranks:
        max_rank = max(ranks)
        if user_rank not in ranks:
            return {
                "supported": False,
                "reason": f"최대 {rank_to_size(max_rank)}까지로 보여요.",
                "matched_option": None,
                "confidence": "db",
            }
        if chest_signal.get("status") is False:
            return {
                "supported": False,
                "reason": chest_signal.get("reason", "") or "권장 범위에는 있어도 실측상 타이트할 수 있어요.",
                "matched_option": None,
                "confidence": "db+measure",
            }
        if user_rank >= max_rank or chest_signal.get("status") == "edge":
            return {
                "supported": "edge",
                "reason": chest_signal.get("reason", "") or f"{user_size_text}까지는 포함되지만 경계선에 가까워 보여요.",
                "matched_option": None,
                "confidence": "db+measure",
            }
        return {
            "supported": True,
            "reason": chest_signal.get("reason", "") or "권장 범위 안쪽으로 보여요.",
            "matched_option": None,
            "confidence": "db+measure",
        }

    if chest_signal.get("status") in [False, "edge", True]:
        return {
            "supported": chest_signal.get("status"),
            "reason": chest_signal.get("reason", ""),
            "matched_option": None,
            "confidence": "measure-only",
        }

    return {"supported": None, "reason": "", "matched_option": None, "confidence": "unknown"}

def is_size_question(user_text: str) -> bool:
    q = clean_text(user_text).replace(" ", "")
    return any(k in q for k in ["사이즈", "맞을까", "맞을까요", "맞아", "핏", "작을까", "클까", "여유", "타이트", "77", "66반", "88"])


def is_name_question(user_text: str) -> bool:
    q = clean_text(user_text).replace(" ", "")
    return any(k in q for k in ["이옷이름", "상품명", "상품이름", "이름뭐", "이옷이뭐야", "품명"])


def is_color_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["컬러", "색상", "무슨 색", "어떤 색", "블랙", "아이보리", "베이지", "네이비", "핑크", "그레이"])


def is_recommendation_question(user_text: str) -> bool:
    q = clean_text(user_text)
    return any(k in q for k in ["추천", "어울리는", "같이 입", "코디", "매치", "무슨 바지", "어떤 바지", "무슨 치마", "잘 어울리는", "다른", "비슷한", "학교에", "행사", "입고갈"])


def current_product_dict(product_context: Dict, db_product: Optional[Dict]) -> Dict:
    return {
        "product_no": normalize_product_no((db_product or {}).get("product_no", "") or product_context.get("product_no", "")),
        "product_name": clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품"),
        "category": clean_text((db_product or {}).get("category", "") or product_context.get("category", "") or "기타"),
        "sub_category": clean_text((db_product or {}).get("sub_category", "")),
        "size_range": clean_text((db_product or {}).get("size_range", "")),
        "style_tags": clean_text((db_product or {}).get("style_tags", "")),
        "coordination_items": clean_text((db_product or {}).get("coordination_items", "")),
        "body_cover_features": clean_text((db_product or {}).get("body_cover_features", "")),
        "fabric": clean_text((db_product or {}).get("fabric", "")) or clean_text(product_context.get("material", "")),
    }


def row_blob(rowd: Dict) -> str:
    cols = [
        "product_name", "category", "sub_category", "style_tags", "coordination_items",
        "body_cover_features", "recommended_body_type", "product_summary", "fabric"
    ]
    return " ".join(clean_text(rowd.get(c, "")) for c in cols)


def is_bottom_product(rowd: Dict) -> bool:
    corpus = row_blob(rowd)
    return any(k in corpus for k in BOTTOM_CATS) and not any(k in corpus for k in ["셔츠", "블라우스", "가디건", "니트", "자켓", "점퍼", "코트", "맨투맨", "티셔츠"])


def is_top_product(rowd: Dict) -> bool:
    corpus = row_blob(rowd)
    return any(k in corpus for k in TOP_CATS)


def infer_target_category_from_query(user_text: str, current_product: Dict) -> str:
    q = clean_text(user_text)
    if any(k in q for k in ["바지", "슬랙스", "팬츠", "데님", "청바지"]):
        return "팬츠"
    if any(k in q for k in ["스커트", "치마"]):
        return "스커트"
    if any(k in q for k in ["자켓", "재킷", "아우터"]):
        return "자켓"
    if "맨투맨" in q:
        return "맨투맨"
    if "티셔츠" in q:
        return "티셔츠"
    if "블라우스" in q:
        return "블라우스"
    if "셔츠" in q:
        return "셔츠"
    if "가디건" in q:
        return "가디건"
    if "니트" in q:
        return "니트"

    current_cat = clean_text(current_product.get("category", ""))
    current_sub = clean_text(current_product.get("sub_category", ""))
    current_name = clean_text(current_product.get("product_name", ""))
    corpus = f"{current_cat} {current_sub} {current_name}"

    if any(k in q for k in ["어울리는", "코디", "같이 입", "매치"]):
        if any(k in corpus for k in ["자켓", "재킷", "블라우스", "셔츠", "니트", "가디건", "맨투맨", "티셔츠"]):
            return "팬츠"
        if any(k in corpus for k in BOTTOM_CATS):
            return "블라우스"

    if "맨투맨" in corpus:
        return "맨투맨"
    if "블라우스" in corpus:
        return "블라우스"
    if "셔츠" in corpus:
        return "셔츠"
    if "가디건" in corpus:
        return "가디건"
    if "니트" in corpus:
        return "니트"
    if any(k in corpus for k in ["자켓", "재킷", "아우터", "점퍼", "코트"]):
        return "팬츠"
    if any(k in corpus for k in BOTTOM_CATS):
        return "블라우스"
    return ""

def category_matches_target(explicit: str, target: str) -> bool:
    if not target:
        return True
    if target == "자켓":
        return any(k in explicit for k in ["자켓", "재킷"])
    if target == "블라우스":
        return "블라우스" in explicit
    if target == "셔츠":
        return "셔츠" in explicit and "블라우스" not in explicit
    if target == "맨투맨":
        return "맨투맨" in explicit and not any(k in explicit for k in ["후드", "집업"])
    if target == "티셔츠":
        return "티셔츠" in explicit and "맨투맨" not in explicit
    if target == "가디건":
        return "가디건" in explicit
    if target == "니트":
        return "니트" in explicit
    if target == "스커트":
        return any(k in explicit for k in ["스커트", "치마"])
    if target == "팬츠":
        return any(k in explicit for k in ["팬츠", "슬랙스", "바지", "데님", "청바지"])
    return True


def build_no_reco_message(target: str) -> str:
    if target in ["블라우스", "셔츠"]:
        return "지금 조건에 바로 맞는 블라우스가 많이 잡히진 않아서요. 원하시면 조금 더 단정한 쪽인지, 편하게 입는 쪽인지 기준 맞춰서 다시 골라드릴게요 :)"
    if target == "맨투맨":
        return "지금 조건에 바로 맞는 맨투맨이 많이 잡히진 않아서요. 원하시면 조금 더 여유핏 쪽인지, 깔끔한 기본핏 쪽인지 기준 맞춰서 다시 골라드릴게요 :)"
    if target == "자켓":
        return "지금 조건에 바로 맞는 자켓이 많이 잡히진 않아서요. 원하시면 학교용처럼 단정한 쪽인지, 평소에도 편하게 입는 쪽인지 기준 맞춰서 다시 골라드릴게요 :)"
    if target in ["팬츠", "스커트"]:
        return "지금 옷이랑 바로 잘 붙는 하의가 많이 잡히진 않아서요. 원하시면 슬랙스 느낌인지, 데님 느낌인지 기준 맞춰서 다시 골라드릴게요 :)"
    return "지금 조건에 딱 맞는 상품이 바로 많이 잡히진 않아서요. 원하시면 원하는 느낌을 조금 더 말씀해주시면 그 기준으로 다시 골라드릴게요 :)"


def build_product_reason(rowd: Dict, user_text: str) -> List[str]:
    reasons: List[str] = []
    name = clean_text(rowd.get("product_name", ""))
    sub = clean_text(rowd.get("sub_category", ""))
    style = clean_text(rowd.get("style_tags", ""))
    fit = clean_text(rowd.get("fit_type", ""))
    cover = clean_text(rowd.get("body_cover_features", ""))
    q = clean_text(user_text)

    if any(k in q for k in ["학교", "행사", "학부모", "상담", "모임"]):
        if any(k in style for k in ["클래식", "오피스룩", "단정"]):
            reasons.append("학교 행사에 입기에도 너무 캐주얼하지 않고 깔끔한 쪽이에요")
        else:
            reasons.append("과하게 힘준 느낌 없이 단정하게 입기 좋은 쪽이에요")
    elif "출근" in q:
        reasons.append("출근할 때도 단정하게 이어입기 좋은 쪽이에요")
    elif "데일리" in q:
        reasons.append("평소에도 부담 없이 손이 잘 가는 쪽이에요")

    if any(k in sub + ' ' + name for k in ["자켓", "재킷"]):
        if any(k in fit for k in ["여유", "루즈", "오버"]):
            reasons.append("상체가 있는 편이어도 답답한 느낌이 덜한 편이에요")
        elif any(k in fit for k in ["정핏", "기본"]):
            reasons.append("핏이 과하게 크지 않아서 단정하게 보이기 좋아요")
        else:
            reasons.append("어깨선이 너무 과하지 않게 떨어져서 깔끔하게 입기 좋아요")
    elif any(k in sub + ' ' + name for k in ["블라우스", "셔츠"]):
        reasons.append("얼굴 쪽이 답답해 보이지 않고 단정하게 받쳐주기 좋아요")
    elif any(k in sub + ' ' + name for k in ["팬츠", "슬랙스", "데님", "스커트"]):
        reasons.append("지금 보시는 상의랑 붙였을 때 전체 라인이 깔끔하게 정리돼요")

    if any(k in cover for k in ["팔뚝", "상체", "가슴"]):
        reasons.append("상체라인 부담을 조금 덜어주는 쪽이에요")
    elif any(k in cover for k in ["허리", "복부", "뱃살", "힙"]):
        reasons.append("전체 실루엣이 부해 보이지 않게 잡아주는 편이에요")

    out = []
    for r in reasons:
        if r and r not in out:
            out.append(r)
    return out[:2] if out else ["지금 찾으시는 느낌으로 무난하게 손이 갈 만한 쪽이에요"]




def save_recommendations(recos: List[Dict]) -> None:
    try:
        cleaned: List[Dict] = []
        for reco in recos:
            rowd = reco.get("_full_row", {}) if isinstance(reco, dict) else {}
            cleaned.append({
                "product_name": clean_text(reco.get("product_name", "")),
                "product_no": normalize_product_no(clean_text(rowd.get("product_no", "") or reco.get("product_no", ""))),
                "category": clean_text(reco.get("category", "")),
                "sub_category": clean_text(reco.get("sub_category", "")),
                "size_range": clean_text(reco.get("size_range", "")),
                "reasons": reco.get("reasons", [])[:2] if isinstance(reco, dict) else [],
                "_full_row": rowd if isinstance(rowd, dict) else {},
            })
        st.session_state.last_recommendations = cleaned
    except Exception:
        st.session_state.last_recommendations = []

def get_recommendation_reference_index(user_text: str) -> Optional[int]:
    q = clean_text(user_text).replace(" ", "")
    mapping = {
        0: ["1번", "첫번째", "첫번째상품", "첫번째옷", "첫번째바지", "첫째"],
        1: ["2번", "두번째", "두번째상품", "두번째옷", "두번째바지", "둘째"],
        2: ["3번", "세번째", "세번째상품", "세번째옷", "세번째바지", "셋째"],
    }
    for idx, words in mapping.items():
        if any(w in q for w in words):
            return idx
    if any(w in q for w in ["방금추천", "추천해준", "그거", "그상품", "그옷", "그바지"]):
        return 0
    return None

def get_followup_recommendation(user_text: str) -> Optional[Dict]:
    idx = get_recommendation_reference_index(user_text)
    recos = st.session_state.get("last_recommendations", []) or []
    if idx is None or idx >= len(recos):
        return None
    return recos[idx]

def recommendation_to_context(reco: Dict) -> Tuple[Dict, Optional[Dict]]:
    rowd = reco.get("_full_row", {}) if isinstance(reco, dict) else {}
    db_like = rowd if rowd else {
        "product_no": clean_text(reco.get("product_no", "")),
        "product_name": clean_text(reco.get("product_name", "")),
        "category": clean_text(reco.get("category", "")),
        "sub_category": clean_text(reco.get("sub_category", "")),
        "size_range": clean_text(reco.get("size_range", "")),
    }
    raw_blob = " ".join([
        clean_text(db_like.get("product_name", "")),
        clean_text(db_like.get("category", "")),
        clean_text(db_like.get("sub_category", "")),
        clean_text(db_like.get("fit_type", "")),
        clean_text(db_like.get("body_cover_features", "")),
        clean_text(db_like.get("style_tags", "")),
        clean_text(db_like.get("coordination_items", "")),
        clean_text(db_like.get("product_summary", "")),
        clean_text(db_like.get("fabric", "")),
        clean_text(db_like.get("size_range", "")),
        clean_text(db_like.get("color_options", "")),
    ])
    context = {
        "product_no": normalize_product_no(clean_text(db_like.get("product_no", ""))),
        "product_name": clean_text(db_like.get("product_name", "") or reco.get("product_name", "") or "추천드린 상품"),
        "category": clean_text(db_like.get("category", "") or reco.get("category", "") or "기타"),
        "sub_category": clean_text(db_like.get("sub_category", "") or reco.get("sub_category", "")),
        "summary": clean_text(db_like.get("product_summary", "")),
        "material": clean_text(db_like.get("fabric", "")),
        "fit": " / ".join([x for x in [clean_text(db_like.get("fit_type", "")), clean_text(db_like.get("body_cover_features", ""))] if x]),
        "size_tip": clean_text(db_like.get("size_range", "") or reco.get("size_range", "")),
        "raw_excerpt": raw_blob,
        "colors": parse_color_options({"raw_excerpt": raw_blob}, db_like if db_like else None),
    }
    return context, db_like

def build_followup_recommendation_answer(user_text: str) -> Optional[str]:
    q = clean_text(user_text)
    recos = st.session_state.get("last_recommendations", []) or []

    if recos and is_size_question(user_text) and any(k in q for k in ["추천해준", "추천드린", "지금 추천", "이 바지들", "이 상품들", "이 옷들", "내 사이즈에는 맞아"]):
        lines = []
        for i, reco in enumerate(recos[:3], start=1):
            reco_context, reco_db = recommendation_to_context(reco)
            user_size, body_label = get_active_user_size(reco_context, reco_db)
            if not user_size:
                continue
            size_eval = evaluate_size_support(user_size, reco_context, reco_db)
            name = reco_context["product_name"]
            if size_eval["supported"] is False:
                lines.append(f"{i}. {name}은 고객님 {body_label} {user_size} 기준이면 여유 있게 맞는 쪽보다는 조금 타이트할 수 있어요.")
            elif size_eval["supported"] == "edge":
                lines.append(f"{i}. {name}은 고객님 {body_label} {user_size} 기준이면 가능권에는 들어오지만 여유 있는 편은 아니에요.")
            elif size_eval["supported"] is True:
                lines.append(f"{i}. {name}은 고객님 {body_label} {user_size} 기준으로 무리 없는 쪽으로 보여요.")
            else:
                lines.append(f"{i}. {name}은 지금 보이는 정보만으로는 확답보다 실측을 같이 보는 쪽이 더 정확해요.")
        if lines:
            lines.append("마음 가는 번호 있으면 그 상품 하나만 더 자세히 이어서 봐드릴게요 :)")
            return "\n".join(lines)

    reco = get_followup_recommendation(user_text)
    if not reco:
        return None
    reco_context, reco_db = recommendation_to_context(reco)
    idx = (get_recommendation_reference_index(user_text) or 0) + 1

    if is_name_question(user_text) or any(k in q for k in ["어떤 옷", "어떤 바지", "어떤 상품", "설명", "알려줘", "뭐야"]):
        reasons = [clean_text(x) for x in reco.get("reasons", []) if clean_text(x)]
        reason_line = " ".join(reasons[:2]) if reasons else "지금 문의하신 자리랑 잘 어울리는 쪽으로 먼저 골라드린 상품이에요."
        return f"{idx}번으로 추천드린 상품은 {reco_context['product_name']}이에요 :)\n{reason_line}\n원하시면 이 상품 기준으로 사이즈감도 바로 이어서 봐드릴게요."

    if is_size_question(user_text):
        user_size, body_label = get_active_user_size(reco_context, reco_db)
        if not user_size:
            return f"{idx}번으로 추천드린 {reco_context['product_name']} 기준으로 보려면 고객님 평소 {body_label} 사이즈를 먼저 알려주세요 :)"
        size_eval = evaluate_size_support(user_size, reco_context, reco_db)
        matched = size_eval.get("matched_option")
        reason = clean_text(size_eval.get("reason", ""))
        name = reco_context["product_name"]
        if size_eval["supported"] is False:
            return f"{idx}번으로 추천드린 {name}은 고객님 {body_label} {user_size} 기준이면 넉넉하게 맞는 쪽으로 보긴 어려워요.\n{reason or '현재 확인되는 사이즈 범위가 고객님보다 작게 잡혀 있어요.'}\n편하게 입는 기준이라면 강하게 추천드리긴 어렵고, 조금 더 여유 있는 쪽을 같이 보시는 게 안전해요."
        if size_eval["supported"] == "edge":
            label = matched["label"] if matched else "현재 옵션"
            return f"{idx}번으로 추천드린 {name}은 고객님 {body_label} {user_size} 기준이면 가능권에는 들어오지만 경계선에 가까워요.\n{reason or (label + ' 기준으로 상단 사이즈에 가까워 보여요.')}\n딱 맞는 느낌으로는 가능할 수 있지만, 여유 있게 입으실 거면 조금 더 편한 쪽이 나을 수 있어요."
        if size_eval["supported"] is True:
            if matched:
                return f"{idx}번으로 추천드린 {name}은 고객님 {body_label} {user_size} 기준이면 {matched['label']} 쪽으로 보시면 돼요 :)\n{reason or '현재 확인되는 범위 안쪽으로 보여요.'}\n부담 없이 입는 쪽으로는 비교적 안정적인 편이에요."
            return f"{idx}번으로 추천드린 {name}은 고객님 {body_label} {user_size} 기준으로 무리 없는 편으로 보여요 :)\n{reason}\n실측까지 같이 보면 더 정확하지만, 지금 기준으로는 비교적 안정적인 쪽이에요."
        db_range = clean_text((reco_db or {}).get("size_range", ""))
        if db_range:
            return f"{idx}번으로 추천드린 {name}은 현재 {db_range} 쪽으로 먼저 안내되는 상품이에요 :)\n지금 정보만으로 딱 잘라 말씀드리기보다는 실측을 같이 보면 더 정확해요."
        return f"{idx}번으로 추천드린 {name}은 지금 보이는 정보만으로는 확답보다 실측을 같이 보는 쪽이 더 정확해요 :)"

    if is_color_question(user_text):
        ans = build_color_answer(reco_context, reco_db)
        if ans:
            return f"{idx}번으로 추천드린 {reco_context['product_name']} 기준으로 보면, {ans}"
        return f"{idx}번으로 추천드린 {reco_context['product_name']}은 현재 컬러 정보가 또렷하게 확인되진 않아요."

    return None

def recommend_products_for_query(user_text: str, current_product: Dict, body_ctx: Dict[str, str], limit: int = 3) -> List[Dict]:
    if DB.empty:
        return []
    target = infer_target_category_from_query(user_text, current_product)
    current_no = normalize_product_no(current_product.get("product_no", ""))
    top_rank = size_rank(body_ctx.get("top_size", ""))
    bottom_rank = size_rank(body_ctx.get("bottom_size", ""))
    q = clean_text(user_text)
    want_more = any(k in q for k in ["다른", "또", "더", "말고", "없어"])
    seen_names = set(st.session_state.get("reco_seen_names", []) or []) if want_more else set()

    scored: List[Tuple[int, Dict]] = []
    for _, row in DB.iterrows():
        rowd = row.to_dict()
        row_no = normalize_product_no(rowd.get("product_no", ""))
        if current_no and row_no == current_no:
            continue
        name = clean_text(rowd.get("product_name", ""))
        if not name or name in seen_names:
            continue

        category = clean_text(rowd.get("category", ""))
        sub = clean_text(rowd.get("sub_category", ""))
        fit = clean_text(rowd.get("fit_type", ""))
        style = clean_text(rowd.get("style_tags", ""))
        cover = clean_text(rowd.get("body_cover_features", ""))
        summary = clean_text(rowd.get("product_summary", ""))
        explicit = f"{category} {sub} {name}"
        ranks = expand_size_text(clean_text(rowd.get("size_range", "")))

        if not category_matches_target(explicit, target):
            continue

        if any(k in q for k in ["학교", "행사", "학부모", "상담", "모임"]):
            if any(k in explicit for k in ["후드", "집업", "트레이닝", "조거", "야상"]):
                continue

        if "후드집업" in q or "후드 집업" in q or "후드말고" in q or "후드 말고" in q:
            if any(k in explicit for k in ["후드", "집업"]):
                continue

        if target in ["자켓", "블라우스", "셔츠", "가디건", "니트", "맨투맨", "티셔츠"] and top_rank and ranks and top_rank not in ranks:
            continue
        if target in ["팬츠", "스커트"] and bottom_rank and ranks and bottom_rank not in ranks:
            continue

        score = 0
        score += 18 if category_matches_target(explicit, target) else 0
        if any(k in q for k in ["학교", "행사", "학부모", "상담", "모임"]):
            if any(k in style for k in ["클래식", "오피스룩", "단정"]):
                score += 10
            if any(k in explicit for k in ["후드", "집업", "맨투맨", "야상"]):
                score -= 20
        if current_product.get("style_tags"):
            current_tags = set(clean_text(x) for x in re.split(r"[;,/|]", current_product.get("style_tags", "")) if clean_text(x))
            row_tags = set(clean_text(x) for x in re.split(r"[;,/|]", style) if clean_text(x))
            score += len(current_tags & row_tags) * 2
        if target in ["자켓", "블라우스", "셔츠", "가디건", "니트", "맨투맨", "티셔츠"] and any(k in fit for k in ["여유", "루즈", "오버"]):
            score += 3
        if target in ["팬츠", "스커트"] and any(k in cover for k in ["허리", "힙", "허벅지", "복부"]):
            score += 3
        if target in ["자켓", "블라우스", "셔츠", "가디건", "니트", "맨투맨", "티셔츠"] and any(k in cover for k in ["팔뚝", "상체", "가슴"]):
            score += 3
        if ranks:
            score += 1

        scored.append((score, rowd))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict] = []
    seen = set()
    for score, rowd in scored:
        name = clean_text(rowd.get("product_name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        out.append({
            "product_name": name,
            "product_no": normalize_product_no(clean_text(rowd.get("product_no", ""))),
            "category": clean_text(rowd.get("category", "")),
            "sub_category": clean_text(rowd.get("sub_category", "")),
            "size_range": clean_text(rowd.get("size_range", "")),
            "reasons": build_product_reason(rowd, user_text),
            "_full_row": rowd,
        })
        if len(out) >= limit:
            break
    return out

def build_name_answer(product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", ""))
    if not name or name == "지금 보시는 상품":
        return None
    return f"지금 보시는 상품은 {name}이에요 :)"


def build_color_answer(product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    colors = parse_color_options(product_context, db_product)
    if not colors:
        return None
    return f"현재 확인되는 컬러는 {', '.join(colors)} 쪽이에요. 없는 컬러를 임의로 말씀드리기보다는 지금 보이는 옵션 기준으로 같이 봐드릴게요 :)"


def build_size_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    if not is_size_question(user_text):
        return None
    user_size, body_label = get_active_user_size(product_context, db_product)
    if not user_size:
        return f"{body_label} 사이즈를 같이 입력해주시면 더 정확하게 봐드릴 수 있어요 :) 지금은 고객님 평소 {body_label} 사이즈를 먼저 알려주세요."

    current_name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")
    size_eval = evaluate_size_support(user_size, product_context, db_product)
    reason = clean_text(size_eval.get("reason", ""))
    # clean internal phrases
    for bad in ["DB 기준으로는", "DB 기준", "현재 페이지 기준으로는", "현재 페이지 기준으로", "현재 확인되는 기준으로는", "현재 확인되는 기준으로"]:
        reason = reason.replace(bad, "")
    reason = clean_text(reason)

    if body_label == "상의":
        tight_extra = "상체가 있는 편이라고 하셔서 어깨나 가슴 쪽은 조금 더 또렷하게 느껴지실 수 있어요." if any(k in clean_text(user_text) for k in ["상체", "어깨", "가슴"]) else ""
        petite_extra = "기장은 과하게 긴 쪽은 아니라 키가 아담하셔도 아주 부담스럽진 않을 거예요." if any(k in clean_text(user_text) for k in ["키가 작", "아담", "작은 편"]) else ""
    else:
        tight_extra = "허리나 힙 쪽은 체형에 따라 조금 더 타이트하게 느껴질 수 있어요."
        petite_extra = "기장은 평소 선호하시는 느낌에 따라 체감이 달라질 수 있어요."

    if size_eval["supported"] is False:
        parts = [f"고객님 {body_label} {user_size} 기준이면 {current_name}은 편하게 맞는 쪽보다는 살짝 타이트하게 느껴질 수 있어요."]
        if reason:
            parts.append(reason)
        if tight_extra:
            parts.append(tight_extra)
        parts.append("평소 딱 맞게 입는 것보다 조금 편한 핏을 좋아하시면, 한 단계 더 여유 있는 쪽을 같이 보는 게 나아요.")
        return " ".join(parts)

    if size_eval["supported"] == "edge":
        parts = [f"고객님 {body_label} {user_size} 기준이면 {current_name}은 아예 못 입는 쪽은 아닌데, 여유 있게 맞는 느낌보다는 경계선에 가까워요."]
        if reason:
            parts.append(reason)
        if tight_extra:
            parts.append(tight_extra)
        parts.append("정핏으로 입으시면 가능할 수 있지만, 조금 편하게 입으시는 스타일이면 더 여유 있는 쪽이 만족도는 높아요.")
        return " ".join(parts)

    if size_eval["supported"] is True:
        parts = [f"고객님 {body_label} {user_size} 기준이면 {current_name}은 무리 없이 입으실 수 있는 쪽으로 보여요 :)"]
        if reason:
            parts.append(reason)
        if petite_extra and body_label == "상의":
            parts.append(petite_extra)
        parts.append("다만 원하시는 핏이 딱 맞는 쪽인지, 조금 여유 있는 쪽인지에 따라 체감은 조금 달라질 수 있어요.")
        return " ".join(parts)

    db_range = clean_text((db_product or {}).get("size_range", ""))
    if db_range:
        return f"지금 보이는 사이즈 흐름으로는 {current_name}이 {db_range} 정도에 가까운 편이에요 :) 원하시면 제가 비슷한 상품까지 같이 골라드릴게요."
    return f"지금 정보만으로는 {current_name} 사이즈를 딱 잘라 말씀드리기보다, 비슷한 핏의 다른 상품까지 같이 보는 쪽이 더 정확해요 :)"


def build_recommendation_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    if not is_recommendation_question(user_text):
        return None

    current_product = current_product_dict(product_context, db_product)
    body_ctx = build_body_context()
    q = clean_text(user_text)

    if any(k in q for k in ["자켓", "재킷"]) and any(k in q for k in ["바지", "슬랙스", "팬츠", "데님", "청바지"]):
        jacket_recos = recommend_products_for_query("자켓 추천 " + user_text, current_product, body_ctx, limit=2)
        pants_recos = recommend_products_for_query("바지 추천 " + user_text, current_product, body_ctx, limit=2)
        if not jacket_recos and not pants_recos:
            return build_no_reco_message("자켓")
        merged = jacket_recos + pants_recos
        save_recommendations(merged)
        prev_seen = list(st.session_state.get("reco_seen_names", []) or [])
        prev_seen.extend([r.get("product_name", "") for r in merged])
        st.session_state.reco_seen_names = list(dict.fromkeys([x for x in prev_seen if x]))
        st.session_state.last_reco_target = "코디"
        lines = ["학교나 행사에 맞춰 자켓이랑 바지를 같이 골라드릴게요 :)"]
        n = 1
        if jacket_recos:
            lines.append("자켓은 이런 쪽이 좋아요.")
            for reco in jacket_recos:
                reason_text = " ".join([clean_text(x) for x in reco.get("reasons", []) if clean_text(x)][:2]).strip() or "단정하게 받쳐주기 좋은 쪽이에요."
                lines.append(f"{n}. {reco['product_name']} — {reason_text}")
                n += 1
        if pants_recos:
            lines.append("바지는 이런 쪽이 잘 어울려요.")
            for reco in pants_recos:
                reason_text = " ".join([clean_text(x) for x in reco.get("reasons", []) if clean_text(x)][:2]).strip() or "전체 라인이 깔끔하게 정리돼 보여요."
                lines.append(f"{n}. {reco['product_name']} — {reason_text}")
                n += 1
        lines.append("마음 가는 번호 말씀해주시면 그 상품 기준으로 사이즈감까지 바로 이어서 봐드릴게요 :)")
        return "\n".join(lines)

    recos = recommend_products_for_query(user_text, current_product, body_ctx, limit=3)
    target = infer_target_category_from_query(user_text, current_product)
    if not recos:
        st.session_state.last_recommendations = []
        return build_no_reco_message(target)

    save_recommendations(recos)
    prev_seen = list(st.session_state.get("reco_seen_names", []) or [])
    prev_seen.extend([r.get("product_name", "") for r in recos])
    st.session_state.reco_seen_names = list(dict.fromkeys([x for x in prev_seen if x]))
    st.session_state.last_reco_target = target

    opener_map = {
        "자켓": "네, 고객님 쪽에 조금 더 잘 맞을 만한 자켓으로 먼저 골라드릴게요.",
        "블라우스": "네, 고객님 쪽에 잘 맞을 만한 블라우스로 먼저 골라드릴게요.",
        "셔츠": "네, 고객님 쪽에 잘 맞을 만한 셔츠로 먼저 골라드릴게요.",
        "맨투맨": "네, 고객님 쪽에 잘 맞을 만한 맨투맨으로 먼저 골라드릴게요.",
        "티셔츠": "네, 고객님 쪽에 잘 맞을 만한 티셔츠로 먼저 골라드릴게요.",
        "가디건": "네, 고객님 쪽에 잘 맞을 만한 가디건으로 먼저 골라드릴게요.",
        "니트": "네, 고객님 쪽에 잘 맞을 만한 니트로 먼저 골라드릴게요.",
        "팬츠": "네, 지금 옷이랑 잘 이어입기 좋은 바지로 먼저 골라드릴게요.",
        "스커트": "네, 지금 옷이랑 잘 이어입기 좋은 스커트로 먼저 골라드릴게요.",
    }
    opener = opener_map.get(target, "네, 고객님 쪽에 잘 맞을 만한 쪽으로 먼저 골라드릴게요.")

    lines = [opener]
    for i, reco in enumerate(recos, start=1):
        reasons = [clean_text(x) for x in reco.get("reasons", []) if clean_text(x)]
        reason_text = " ".join(reasons[:2]).strip() or "지금 찾으시는 느낌으로 무난하게 손이 갈 만한 쪽이에요."
        lines.append(f"{i}. {reco['product_name']} — {reason_text}")
    lines.append("마음 가는 번호 말씀해주시면 그 상품 기준으로 사이즈감까지 바로 이어서 봐드릴게요 :)")
    return "\n".join(lines)

def trim_text(text: str, max_len: int = 500) -> str:
    t = clean_text(text)
    return t if len(t) <= max_len else t[:max_len] + "…"


def slim_current_context(product_context: Dict, db_product: Optional[Dict], user_text: str) -> Dict:
    current = current_product_dict(product_context, db_product)
    colors = parse_color_options(product_context, db_product)
    body = build_body_context()
    active_size, active_label = get_active_user_size(product_context, db_product)
    size_eval = evaluate_size_support(active_size, product_context, db_product) if active_size else {"supported": None, "reason": ""}
    recos = recommend_products_for_query(user_text, current, body, limit=3) if is_recommendation_question(user_text) else []
    return {
        "current_product_name": current.get("product_name") or "지금 보시는 상품",
        "current_product_no": current.get("product_no", ""),
        "current_product_category": current.get("category", ""),
        "current_product_sub_category": current.get("sub_category", ""),
        "active_body_label": active_label,
        "active_body_size": active_size,
        "confirmed_colors": colors[:8],
        "confirmed_size_support": size_eval.get("supported"),
        "size_support_reason": trim_text(size_eval.get("reason", ""), 220),
        "body_context": body,
        "page_summary": trim_text(product_context.get("summary", ""), 700),
        "page_fit": trim_text(product_context.get("fit", ""), 260),
        "page_material": trim_text(product_context.get("material", ""), 220),
        "db_size_range": trim_text(clean_text((db_product or {}).get("size_range", "")), 80),
        "db_style_tags": trim_text(clean_text((db_product or {}).get("style_tags", "")), 120),
        "db_coordination_items": trim_text(clean_text((db_product or {}).get("coordination_items", "")), 120),
        "allowed_recommendation_candidates": recos[:3],
        "policy_db": POLICY_DB,
    }

def llm_can_help(user_text: str) -> bool:
    # deterministic first for risky areas
    if is_name_question(user_text) or is_size_question(user_text) or is_color_question(user_text):
        return False
    fast = get_fast_policy_answer(user_text)
    if fast:
        return False
    return True


def call_llm(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    pack = slim_current_context(product_context, db_product, user_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "참고 데이터(JSON):\n" + json.dumps(pack, ensure_ascii=False)},
    ]
    for m in st.session_state.messages[-4:]:
        messages.append({"role": m["role"], "content": trim_text(m["content"], 300)})
    messages.append({"role": "user", "content": trim_text(user_text, 350)})
    last_error = None
    for wait in (0, 1.2):
        if wait:
            time.sleep(wait)
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.35,
                max_tokens=260,
            )
            content = clean_text(resp.choices[0].message.content or "")
            if not content:
                continue
            return content
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
        lambda: build_size_answer(user_text, product_context, db_product),
        lambda: build_recommendation_answer(user_text, product_context, db_product),
        lambda: build_color_answer(product_context, db_product) if is_color_question(user_text) else None,
    ]:
        ans = builder()
        if ans:
            return ans
    return "지금 문의가 잠시 몰려서 답변 연결이 늦어지고 있어요. 같은 내용을 잠깐 뒤 한 번만 다시 보내주시면 바로 이어서 도와드릴게요 :)"


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
            (build_recommendation_answer(user_text, product_context, db_product), "rule"),
            (build_name_answer(product_context, db_product) if is_name_question(user_text) else None, "rule"),
            (get_fast_policy_answer(user_text), "rule"),
            (build_size_answer(user_text, product_context, db_product), "rule"),
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
        write_chat_log("assistant_response", user_text=user_text, answer=answer, response_mode=response_mode, latency_ms=int((time.time()-started)*1000), product_context=product_context)
    except Exception as e:
        answer = "앗, 제가 말을 잇다가 잠깐 꼬였어요. 같은 내용을 한 번만 더 보내주시면 바로 이어서 봐드릴게요 :)"
        st.session_state.last_answer = answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        write_chat_log("error", user_text=user_text, answer=answer, response_mode="error", error_text=f"{type(e).__name__}: {e}", latency_ms=int((time.time()-started)*1000), product_context=product_context)
    finally:
        st.session_state.is_processing = False

# query params
qp = st.query_params
current_url = clean_text(qp.get("url", "") or "")
product_no_q = normalize_product_no(clean_text(qp.get("pn", "") or ""))
product_name_q = clean_text(qp.get("pname", "") or "")
product_no = product_no_q or extract_product_no_from_url(current_url)

context_key = f"{current_url}|{product_no}|{product_name_q}"
if context_key != st.session_state.last_context_key:
    st.session_state.last_context_key = context_key
    st.session_state.messages = []
    st.session_state.last_user_hash = ""
    st.session_state.last_answer = ""
    st.session_state.last_recommendations = []
    st.session_state.reco_seen_names = []
    st.session_state.last_reco_target = ""

product_context = fetch_product_context(current_url, product_name_q, product_no) if current_url else {
    "product_no": product_no,
    "product_name": "지금 보시는 상품",
    "category": "기타",
    "summary": "",
    "material": "",
    "fit": "",
    "size_tip": "",
    "raw_excerpt": "",
    "colors": [],
}
db_product = get_db_product(product_no)
if db_product and clean_text(db_product.get("product_name", "")):
    product_context["product_name"] = clean_text(db_product.get("product_name", ""))

# ---------- UI ----------
st.markdown("""
<style>
header[data-testid="stHeader"] {display:none;}
div[data-testid="stToolbar"] {display:none;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
.block-container{max-width:760px;padding-top:0.22rem !important;padding-bottom:11.0rem !important;}
:root{--miya-accent:#0f6a63;--miya-title:#303443;--miya-sub:#5f6471;--miya-muted:#8f94a3;--miya-divider:#ccccd2;--miya-bot-bg:#071b4e;--miya-user-bg:#dff0ec;--miya-user-text:#1f3b36;--miya-page-bg:#f6f7fb;}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {color: var(--miya-title);background: var(--miya-page-bg) !important;}
div[data-testid="stChatInput"]{position:fixed!important;left:50%!important;transform:translateX(-50%)!important;bottom:14px!important;width:min(720px,calc(100% - 24px))!important;background:transparent!important;}
div[data-testid="stChatInput"] textarea{background:#ffffff!important;border-radius:999px!important;}
@media (max-width: 768px){.block-container{max-width:100%;padding-top:0.14rem!important;padding-bottom:11.6rem!important;}div[data-testid="stChatInput"]{bottom:64px!important;width:calc(100% - 16px)!important;}}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center; margin:0 0 16px 0;">
      <div style="font-size:31px; font-weight:800; line-height:1.1; letter-spacing:-0.02em; color:#303443;">
        미샵 쇼핑친구 <span style="color:#0f6a63;">미야언니</span>
      </div>
      <div style="margin-top:6px; font-size:13.5px; line-height:1.35; color:#5f6471;">
        24시간 쇼핑 결정에 도움드리는 스마트한 쇼핑친구
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-top:2px; margin-bottom:4px;">
      <div style="font-size:13px; font-weight:700; line-height:1.2; color:#303443; margin-bottom:4px;">
        사이즈 입력<span style="font-size:11px; font-weight:500; color:#7a7f8c;">(더 구체적인 상담 가능)</span>
      </div>
      <div style="padding:6px 8px 0 8px; border:1px solid rgba(0,0,0,0.04); border-radius:14px; background:transparent;">
    """,
    unsafe_allow_html=True,
)

row1 = st.columns(2, gap="small")
with row1[0]:
    st.session_state.body_height = st.text_input("키", value=st.session_state.body_height, placeholder="cm", key="body_height_input")
with row1[1]:
    st.session_state.body_weight = st.text_input("체중", value=st.session_state.body_weight, placeholder="kg", key="body_weight_input")

size_options = ["", "44", "55", "55반", "66", "66반", "77", "77반", "88"]
row2 = st.columns(2, gap="small")
with row2[0]:
    current_top = st.session_state.body_top if st.session_state.body_top in size_options else ""
    st.session_state.body_top = st.selectbox("상의", options=size_options, index=size_options.index(current_top), key="body_top_input")
with row2[1]:
    current_bottom = st.session_state.body_bottom if st.session_state.body_bottom in size_options else ""
    st.session_state.body_bottom = st.selectbox("하의", options=size_options, index=size_options.index(current_bottom), key="body_bottom_input")

st.markdown("</div></div>", unsafe_allow_html=True)

body_summary = build_body_context_text(build_body_context())
if any(build_body_context().values()):
    st.markdown(f'<div style="margin-top:2px; margin-bottom:2px; font-size:10.8px; color:#7a7f8c;">현재 입력 정보: {html.escape(body_summary)}</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    current_url_lower = (current_url or "").lower()
    is_detail_page = ("/product/detail" in current_url_lower) or ("product_no=" in current_url_lower) or bool(product_no)
    if is_detail_page:
        welcome = (
            "안녕하세요? 옷 같이 봐드리는 미야언니예요 :)\n"
            "'지금 보시는 상품' 기준으로 제가 같이 봐드릴게요.\n"
            "사이즈, 코디, 배송, 교환 중 뭐부터 이야기해볼까요?"
        )
    else:
        welcome = (
            "안녕하세요? 옷 같이 봐드리는 미야언니예요 :)\n"
            "지금은 일반 상담 모드예요.\n"
            "상품 상세페이지에서 채팅창을 열면 그 상품 기준으로 더 정확하게 상담해드릴 수 있어요 :)\n"
            "이 창을 닫고 해당 상품 상세페이지에서 채팅창을 다시 클릭해주세요^^"
        )
    st.session_state.messages.append({"role": "assistant", "content": welcome})

st.divider()

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

user_input = st.chat_input("메시지를 입력하세요.")
if user_input:
    process_user_message(user_input, product_context, db_product)
    st.rerun()
