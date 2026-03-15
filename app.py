import os
import re
import json
import html
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI

st.set_page_config(
    page_title="미야언니",
    layout="centered",
    initial_sidebar_state="collapsed"
)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY가 필요합니다. Streamlit Secrets에 OPENAI_API_KEY를 추가해주세요.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

POLICY_DB = {
    "shipping": {
        "courier": "CJ 대한통운",
        "shipping_fee": 3000,
        "free_shipping_over": 70000,
        "delivery_time": "결제 완료 후 2~4일 (영업일 기준)",
        "same_day_dispatch_rule": "오후 2시 이전 주문은 당일 출고",
        "reservation_product": "예약상품 개념 없음",
        "combined_shipping": "합배송 가능(1박스 기준). 단 박스크기 초과 시 합배송 불가",
        "dispatch_order": "결제 순서대로 순차 출고",
        "jeju": "제주 및 도서산간 지역은 추가배송비가 자동 부과됩니다."
    },
    "exchange_return": {
        "exchange_possible": "사이즈 교환 가능 / 동일상품 교환 가능 / 타상품 교환 가능",
        "period": "상품 수령 후 7일 이내",
        "exchange_fee": 6000,
        "return_fee_rule": "단순 변심 반품: 반품 후 주문금액이 7만원 이상이면 편도 3,000원 / 7만원 미만이면 왕복 6,000원",
        "defect_wrong": "불량/오배송은 미샵 부담입니다."
    }
}

SYSTEM_PROMPT = """
너는 '미샵 쇼핑친구 미야언니'다.
4050 여성 고객이 쇼핑할 때, 옆에서 같이 봐주는 믿음 가는 언니처럼 대화한다.

핵심 역할:
- 지금 보시는 상품 기준으로 사이즈 / 코디 / 컬러 / 배송 / 교환 상담을 도와준다.
- 고객이 덜 고민하고 덜 헷갈리게 도와준다.
- 너무 딱딱하지 않게, 그러나 가볍지도 않게 답한다.
- 반품을 줄이는 방향으로 솔직하고 안전하게 말한다.

매우 중요한 규칙:
- 실제 미샵 상품명으로 확인된 이름만 사용한다.
- 후보 상품으로 전달된 이름 외에는 새 상품명을 만들지 않는다.
- 컬러는 현재 페이지나 DB에서 확인된 옵션 안에서만 말한다. 모르면 추측하지 않는다.
- 고객 상의 사이즈가 현재 상품 권장 범위를 넘으면 '잘 맞는다', '편하게 맞는다', '여유 있다'고 말하지 않는다.
- 이런 경우에는 현재 상품은 보수적으로 안내하고, 전달된 대체 추천 후보가 있으면 그 안에서만 추천한다.
- 가격/배송/교환 관련 답변은 제공된 데이터만 사용한다.

말투 규칙:
- 친근한 대화체
- 자연스럽게 설명하고, 답변 패턴이 매번 똑같지 않게 조금씩 다르게 말한다
- 고객 체형 정보가 있으면 꼭 참고해서 말한다
- 정보가 부족하면 짧게 필요한 부분만 다시 물어본다
- 답변은 3~7문장 정도로 충분히 설명하되 장황하지 않게 한다
"""

GENERIC_NAMES = {"미샵", "misharp", "MISHARP", "미샵여성", "Misharp", "지금 보시는 상품"}
SIZE_ORDER = {"44": 1, "55": 2, "55반": 3, "66": 4, "66반": 5, "77": 6, "77반": 7, "88": 8, "88반": 9, "99": 10}
COLOR_WORDS = [
    "블랙", "화이트", "아이보리", "크림", "베이지", "카멜", "브라운", "모카", "차콜", "그레이", "회색",
    "네이비", "블루", "소라", "민트", "카키", "그린", "핑크", "로즈", "와인", "레드", "버건디",
    "옐로우", "머스타드", "오렌지", "퍼플", "라벤더"
]


def ensure_state():
    defaults = {
        "messages": [],
        "last_context_key": "",
        "body_height": "",
        "body_weight": "",
        "body_top": "",
        "body_bottom": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def qp_value(v):
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""


def clean_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def is_generic_name(name: str) -> bool:
    name = clean_text(name)
    return (not name) or (name in GENERIC_NAMES) or (len(name) <= 2)


@st.cache_data(show_spinner=False)
def load_db():
    candidates = ["misharp_miya_db.csv", "misharp_miya_db (1).csv"]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [clean_text(c) for c in df.columns]
            for col in df.columns:
                df[col] = df[col].fillna("").map(clean_text)
            if "product_no" in df.columns:
                df["product_no"] = df["product_no"].astype(str).str.replace(".0", "", regex=False).map(clean_text)
            return df
    return None


def split_sections(text: str) -> dict:
    if not text:
        return {"summary": "", "material": "", "fit": "", "size_tip": "", "shipping": "", "colors": ""}

    lines = [clean_text(x) for x in text.split("\n")]
    lines = [x for x in lines if x]
    joined = "\n".join(lines)

    def extract_by_keywords(keywords, max_len=1400):
        matched = []
        for line in lines:
            if any(k in line for k in keywords):
                matched.append(line)
        return " / ".join(matched)[:max_len]

    return {
        "summary": joined[:2600],
        "material": extract_by_keywords(["소재", "원단", "혼용", "%", "면", "폴리", "레이온", "아크릴", "울", "스판", "비스코스", "나일론"]),
        "fit": extract_by_keywords(["핏", "여유", "라인", "체형", "복부", "팔뚝", "허벅지", "힙", "루즈", "와이드", "슬림", "정핏", "세미", "커버"]),
        "size_tip": extract_by_keywords(["사이즈", "정사이즈", "추천", "44", "55", "55반", "66", "66반", "77", "77반", "88", "S", "M", "L", "XL", "FREE", "F(", "L("]),
        "shipping": extract_by_keywords(["배송", "출고", "교환", "반품", "배송비"]),
        "colors": extract_by_keywords(COLOR_WORDS),
    }


def guess_category(name: str, text: str) -> str:
    corpus = f"{name} {text}"
    mapping = {
        "슬랙스": ["슬랙스", "팬츠", "바지"],
        "블라우스": ["블라우스"],
        "셔츠": ["셔츠"],
        "티셔츠": ["티셔츠", "탑"],
        "니트": ["니트", "가디건"],
        "자켓": ["자켓", "재킷"],
        "원피스": ["원피스"],
        "데님": ["데님", "청바지"],
        "코트": ["코트"],
        "맨투맨": ["맨투맨", "스웻"],
        "아우터": ["점퍼", "후드집업", "집업"]
    }
    for cat, keywords in mapping.items():
        if any(k in corpus for k in keywords):
            return cat
    return "기타"


def extract_product_no(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "product_no" in qs and qs["product_no"]:
            return clean_text(qs["product_no"][0])
    except Exception:
        pass
    m = re.search(r"product_no=(\d+)", url)
    return m.group(1) if m else ""


def get_db_product(df: pd.DataFrame | None, product_no: str) -> dict | None:
    if df is None or not product_no or "product_no" not in df.columns:
        return None
    rows = df[df["product_no"].astype(str) == str(product_no)]
    if len(rows) == 0:
        return None
    return rows.iloc[0].to_dict()


def parse_size_tokens(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    tokens = []
    ordered = ["44", "55반", "55", "66반", "66", "77반", "77", "88반", "88", "99"]
    for token in ordered:
        if token in text:
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def supported_size_labels(size_range: str) -> list[str]:
    text = clean_text(size_range)
    if not text:
        return []

    if "-" in text:
        parts = [clean_text(x) for x in text.split("-")]
        if len(parts) == 2 and parts[0] in SIZE_ORDER and parts[1] in SIZE_ORDER:
            start = SIZE_ORDER[parts[0]]
            end = SIZE_ORDER[parts[1]]
            labels = [k for k, v in sorted(SIZE_ORDER.items(), key=lambda x: x[1]) if start <= v <= end]
            return labels

    tokens = parse_size_tokens(text)
    return sorted(tokens, key=lambda x: SIZE_ORDER.get(x, 999))


def max_supported_size(size_range: str) -> str:
    labels = supported_size_labels(size_range)
    return labels[-1] if labels else ""


def size_over_limit(user_top: str, size_range: str) -> bool:
    user_top = clean_text(user_top)
    if user_top not in SIZE_ORDER:
        return False
    max_size = max_supported_size(size_range)
    if max_size not in SIZE_ORDER:
        return False
    return SIZE_ORDER[user_top] > SIZE_ORDER[max_size]


def is_size_question(user_text: str) -> bool:
    t = clean_text(user_text).replace(" ", "")
    keywords = ["사이즈", "맞을까", "맞나요", "맞아", "커요", "작아요", "타이트", "여유", "추천", "몇사이즈", "어떤사이즈", "부담", "낄", "끼", "맞을지"]
    return any(k in t for k in keywords)


def is_color_question(user_text: str) -> bool:
    t = clean_text(user_text)
    return any(k in t for k in ["컬러", "색상", "색", "무슨색", "무슨 컬러", "어떤 색", "어떤컬러"])


def extract_available_colors(page_sections: dict | None, db_product: dict | None) -> list[str]:
    found = []

    if db_product:
        color_text = clean_text(db_product.get("color_options", ""))
        if color_text:
            for part in re.split(r"[;,/|]+", color_text):
                part = clean_text(part)
                if part:
                    found.append(part)

    if page_sections:
        color_text = clean_text(page_sections.get("colors", ""))
        if color_text:
            for color in COLOR_WORDS:
                if color in color_text:
                    found.append(color)

    return list(dict.fromkeys([x for x in found if x]))


def choose_safe_product_name(page_name: str, db_product: dict | None) -> str:
    db_name = clean_text(db_product.get("product_name", "")) if db_product else ""
    page_name = clean_text(page_name)
    if db_name and not is_generic_name(db_name):
        return db_name
    if page_name and not is_generic_name(page_name):
        return page_name
    return "지금 보시는 상품"


def extract_page_product_name(soup: BeautifulSoup, passed_name: str = "") -> str:
    candidates = []
    if passed_name:
        candidates.append(passed_name)
    selectors = [
        "#span_product_name", "#span_product_name_mobile", ".infoArea #span_product_name",
        ".infoArea .headingArea h2", ".infoArea .headingArea h3", ".headingArea h2", ".headingArea h3", "title"
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            candidates.append(el.get_text(" ", strip=True))
    for c in candidates:
        c = clean_text(c)
        c = re.sub(r"\s*\|\s*.*$", "", c)
        c = re.sub(r"\s*-\s*미샵.*$", "", c)
        c = re.sub(r"\s*-\s*MISHARP.*$", "", c, flags=re.I)
        if c and not is_generic_name(c):
            return c
    return clean_text(passed_name) if passed_name else ""


def fetch_product_context(url: str, passed_name: str = "") -> dict | None:
    if not url:
        return None

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=12)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    page_name = extract_page_product_name(soup, passed_name)

    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    raw_text = soup.get_text("\n")
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

    sections = split_sections(raw_text)
    category = guess_category(page_name, raw_text)

    return {
        "product_name": page_name,
        "category": category,
        "summary": sections["summary"],
        "material": sections["material"],
        "fit": sections["fit"],
        "size_tip": sections["size_tip"],
        "shipping": sections["shipping"],
        "colors": sections["colors"],
        "raw_excerpt": raw_text[:4000]
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_product_context_cached(url: str, passed_name: str = "") -> dict | None:
    try:
        return fetch_product_context(url, passed_name)
    except Exception as e:
        safe_name = clean_text(passed_name)
        if is_generic_name(safe_name):
            safe_name = "지금 보시는 상품"
        return {
            "product_name": safe_name,
            "category": "기타",
            "summary": "",
            "material": "",
            "fit": "",
            "size_tip": "",
            "shipping": "",
            "colors": "",
            "raw_excerpt": f"[상품 정보를 가져오지 못했습니다: {e}]"
        }


def get_fast_policy_answer(user_text: str) -> str | None:
    q = user_text.replace(" ", "").lower()

    if any(k in q for k in ["배송비", "무료배송"]):
        return (
            f"배송은 {POLICY_DB['shipping']['courier']}를 이용하고 있고요 :) \n"
            f"배송비는 {POLICY_DB['shipping']['shipping_fee']:,}원이에요. "
            f"{POLICY_DB['shipping']['free_shipping_over']:,}원 이상이면 무료배송으로 적용돼요."
        )

    if any(k in q for k in ["언제출고", "출고", "당일출고"]):
        return (
            f"{POLICY_DB['shipping']['same_day_dispatch_rule']}예요 :) \n"
            f"보통은 {POLICY_DB['shipping']['delivery_time']} 정도 생각해주시면 되고, "
            f"결제 순서대로 순차 출고되고 있어요."
        )

    if any(k in q for k in ["교환", "사이즈교환"]):
        return (
            "교환은 가능해요 :) \n"
            f"{POLICY_DB['exchange_return']['exchange_possible']}이고, "
            f"{POLICY_DB['exchange_return']['period']} 안에 접수해주시면 돼요. \n"
            f"단순 변심 교환은 왕복 {POLICY_DB['exchange_return']['exchange_fee']:,}원으로 안내드리고 있어요."
        )

    if any(k in q for k in ["반품", "환불"]):
        return (
            "반품도 가능해요 :) \n"
            f"{POLICY_DB['exchange_return']['period']} 안에 접수해주시면 되고, "
            f"{POLICY_DB['exchange_return']['return_fee_rule']} 기준으로 진행돼요. \n"
            f"불량이나 오배송이면 배송비는 미샵에서 부담해드려요."
        )

    return None


def build_body_context() -> dict:
    return {
        "height_cm": clean_text(st.session_state.body_height),
        "weight_kg": clean_text(st.session_state.body_weight),
        "top_size": clean_text(st.session_state.body_top),
        "bottom_size": clean_text(st.session_state.body_bottom),
    }


def build_body_context_text(body_ctx: dict) -> str:
    if not any(body_ctx.values()):
        return "입력된 체형 정보 없음"
    return (
        f"키: {body_ctx.get('height_cm') or '-'}cm, "
        f"체중: {body_ctx.get('weight_kg') or '-'}kg, "
        f"상의: {body_ctx.get('top_size') or '-'}, "
        f"하의: {body_ctx.get('bottom_size') or '-'}"
    )


def get_similar_products(df: pd.DataFrame | None, db_product: dict | None, user_top: str, limit: int = 3) -> list[dict]:
    if df is None or not db_product:
        return []

    category = clean_text(db_product.get("category", ""))
    sub_category = clean_text(db_product.get("sub_category", ""))
    current_no = clean_text(db_product.get("product_no", ""))

    work = df.copy()
    if category:
        work = work[work["category"].astype(str) == category]
    if sub_category:
        exact = work[work["sub_category"].astype(str) == sub_category]
        if len(exact) > 0:
            work = exact

    work = work[work["product_no"].astype(str) != current_no]

    if user_top in SIZE_ORDER:
        user_rank = SIZE_ORDER[user_top]
        def can_cover(val):
            mx = max_supported_size(val)
            return SIZE_ORDER.get(mx, 0) >= user_rank
        work = work[work["size_range"].astype(str).map(can_cover)]

    if "fit_type" in work.columns and clean_text(db_product.get("fit_type", "")):
        same_fit = work[work["fit_type"].astype(str).str.contains(clean_text(db_product.get("fit_type", "")), na=False)]
        if len(same_fit) > 0:
            work = same_fit

    cols = [c for c in ["product_no", "product_name", "size_range", "color_options", "fit_type", "sub_category"] if c in work.columns]
    return work[cols].head(limit).to_dict("records")


def build_size_guard_answer(user_text: str, db_product: dict | None, page_context: dict | None, similar_products: list[dict]) -> str | None:
    body = build_body_context()
    user_top = clean_text(body.get("top_size", ""))
    if not user_top or not db_product:
        return None
    if not is_size_question(user_text):
        return None
    if not size_over_limit(user_top, db_product.get("size_range", "")):
        return None

    product_name = choose_safe_product_name(page_context.get("product_name", "") if page_context else "", db_product)
    max_size = max_supported_size(db_product.get("size_range", "")) or db_product.get("size_range", "")

    lines = [
        f"고객님 상의 {user_top} 기준이면 {product_name}은 상품 기준상 {max_size}까지로 보여서 편하게 맞는다고 보긴 어려워요.",
        "특히 상체 여유를 원하시면 답답하거나 핏이 타이트하게 느껴질 수 있어요."
    ]

    if similar_products:
        lines.append("대신 비슷한 무드로 좀 더 안전하게 보실 만한 상품을 같이 추천드릴게요.")
        for p in similar_products[:3]:
            pname = clean_text(p.get("product_name", ""))
            prange = clean_text(p.get("size_range", ""))
            if pname:
                lines.append(f"- {pname} ({prange})")
    else:
        lines.append("같은 카테고리에서 더 여유 있는 상품 쪽으로 보시는 게 안전해요.")

    return "\n".join(lines)


def build_color_guard_answer(user_text: str, available_colors: list[str], page_context: dict | None, db_product: dict | None) -> str | None:
    if not is_color_question(user_text):
        return None
    product_name = choose_safe_product_name(page_context.get("product_name", "") if page_context else "", db_product)
    if available_colors:
        joined = ", ".join(available_colors)
        return f"{product_name}은 현재 확인되는 컬러가 {joined} 정도예요 :) 확인되지 않은 컬러는 제가 추측해서 말씀드리지 않을게요. 원하시면 이 중에서 고객님 분위기에 더 잘 받는 쪽도 같이 골라드릴게요."
    return f"{product_name} 컬러는 지금 화면 정보에서 확실하게 확인되는 옵션만 안내드리려고 해요. 현재 컬러 정보가 또렷하게 잡히지 않아서, 상세 옵션을 한 번 더 확인해주시면 그 기준으로 봐드릴게요."


def get_llm_answer(user_text: str, current_url: str, product_no: str, page_context: dict | None, db_product: dict | None, similar_products: list[dict], available_colors: list[str]) -> str:
    body_context = build_body_context()
    safe_product_name = choose_safe_product_name(page_context.get("product_name", "") if page_context else "", db_product)

    context_pack = {
        "policy_db": POLICY_DB,
        "viewer_context": {
            "url": current_url,
            "is_product_page": bool(product_no),
            "product_no": product_no
        },
        "body_context": body_context,
        "page_context": page_context,
        "db_product": db_product,
        "safe_product_name": safe_product_name,
        "available_colors": available_colors,
        "similar_product_candidates": similar_products,
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "참고 데이터(JSON):\n" + json.dumps(context_pack, ensure_ascii=False)},
    ]

    history = st.session_state.messages[-8:]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.62,
        max_tokens=480
    )
    return resp.choices[0].message.content.strip()


def process_user_message(user_text: str, current_url: str, product_no: str, page_context: dict | None, db_product: dict | None, similar_products: list[dict], available_colors: list[str]):
    st.session_state.messages.append({"role": "user", "content": user_text})

    fast = get_fast_policy_answer(user_text)
    if fast:
        st.session_state.messages.append({"role": "assistant", "content": fast})
        return

    size_guard = build_size_guard_answer(user_text, db_product, page_context, similar_products)
    if size_guard:
        st.session_state.messages.append({"role": "assistant", "content": size_guard})
        return

    color_guard = build_color_guard_answer(user_text, available_colors, page_context, db_product)
    if color_guard:
        st.session_state.messages.append({"role": "assistant", "content": color_guard})
        return

    answer = get_llm_answer(user_text, current_url, product_no, page_context, db_product, similar_products, available_colors)
    st.session_state.messages.append({"role": "assistant", "content": answer})


ensure_state()
qp = st.query_params
current_url = qp_value(qp.get("url", ""))
product_no = qp_value(qp.get("pn", "")) or extract_product_no(current_url)
product_name_q = qp_value(qp.get("pname", ""))

context_key = f"{current_url}|{product_no}|{product_name_q}"
if context_key != st.session_state.last_context_key:
    st.session_state.last_context_key = context_key
    st.session_state.messages = []

DB = load_db()
page_context = fetch_product_context_cached(current_url, product_name_q) if current_url else None
db_product = get_db_product(DB, product_no)
if page_context and db_product and is_generic_name(page_context.get("product_name", "")) and db_product.get("product_name"):
    page_context["product_name"] = clean_text(db_product.get("product_name", ""))

similar_products = get_similar_products(DB, db_product, clean_text(st.session_state.body_top), limit=3)
available_colors = extract_available_colors(page_context, db_product)

st.markdown("""
<style>
header[data-testid="stHeader"] {display:none;}
div[data-testid="stToolbar"] {display:none;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

.block-container{
  max-width:760px;
  padding-top:0.22rem !important;
  padding-bottom:11.0rem !important;
}

:root{
  --miya-accent:#0f6a63;
  --miya-title:#303443;
  --miya-sub:#5f6471;
  --miya-muted:#7a7f8c;
  --miya-divider:#ccccd2;
  --miya-bot-bg:#071b4e;
  --miya-user-bg:#dff0ec;
  --miya-user-text:#1f3b36;
  --miya-input-bg:#1d2130;
  --miya-input-text:#f3f6ff;
  --miya-input-placeholder:#aab2c7;
}

div[data-testid="column"]{min-width:0 !important;}

div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label{
  color:var(--miya-title) !important;
  font-weight:700 !important;
  font-size:11.5px !important;
}

div[data-testid="stTextInput"] input,
div[data-baseweb="select"] > div{border-radius:12px !important;}

div[data-testid="stTextInput"],
div[data-testid="stSelectbox"]{margin-bottom:-2px !important;}

hr{
  margin-top:4px !important;
  margin-bottom:4px !important;
  border-color:var(--miya-divider) !important;
}

div[data-testid="stChatInput"]{
  position:fixed !important;
  left:50% !important;
  transform:translateX(-50%) !important;
  bottom:68px !important;
  width:min(720px, calc(100% - 24px)) !important;
  z-index:9999 !important;
}

div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] input{
  background:var(--miya-input-bg) !important;
  color:var(--miya-input-text) !important;
  caret-color:var(--miya-input-text) !important;
}

div[data-testid="stChatInput"] textarea::placeholder,
div[data-testid="stChatInput"] input::placeholder{
  color:var(--miya-input-placeholder) !important;
  opacity:1 !important;
}

div[data-testid="stChatInput"] button{
  background:#2e3447 !important;
}

div[data-testid="stChatInput"] button svg{
  fill:#f3f6ff !important;
}

@media (max-width: 768px){
  .block-container{
    max-width:100%;
    padding-top:0.14rem !important;
    padding-bottom:11.6rem !important;
  }

  div[data-testid="stHorizontalBlock"]{gap:6px !important;}
  div[data-testid="stHorizontalBlock"] > div{flex:1 1 0 !important; min-width:0 !important;}

  div[data-testid="stTextInput"] label,
  div[data-testid="stSelectbox"] label{font-size:11px !important;}

  div[data-testid="stTextInput"],
  div[data-testid="stSelectbox"]{margin-bottom:-4px !important;}

  hr{margin-top:3px !important; margin-bottom:3px !important;}

  div[data-testid="stChatInput"]{
    bottom:64px !important;
    width:calc(100% - 16px) !important;
  }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center; margin:0 0 16px 0;">
      <div style="font-size:31px; font-weight:800; line-height:1.1; letter-spacing:-0.02em; color:#303443;">
        미샵 쇼핑친구 <span style="color:#0f6a63;">미야언니</span>
      </div>
      <div style="margin-top:6px; font-size:13.5px; line-height:1.35; color:#5f6471;">
        24시간 언제나 미샵님들 쇼핑 판단에 도움드리는 스마트한 쇼핑친구
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="margin-top:2px; margin-bottom:4px;">
      <div style="font-size:13px; font-weight:700; line-height:1.2; color:#303443; margin-bottom:4px;">
        사이즈 입력<span style="font-size:11px; font-weight:500; color:#7a7f8c;">(더 구체적인 상담 가능)</span>
      </div>
      <div style="padding:6px 8px 0 8px; border:1px solid rgba(0,0,0,.04); border-radius:14px; background:transparent;">
    """,
    unsafe_allow_html=True
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
    st.markdown(
        f'<div style="margin-top:2px; margin-bottom:2px; font-size:10.8px; color:#7a7f8c;">현재 입력 정보: {html.escape(body_summary)}</div>',
        unsafe_allow_html=True
    )

if not st.session_state.messages:
    current_url_lower = (current_url or "").lower()
    is_detail_page = (("/product/detail" in current_url_lower) or ("product_no=" in current_url_lower) or bool(product_no))

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
            "상품 상세페이지에서 채팅창을 열면 그 상품 기준으로 더 정확하게 상담해드릴 수 있어요.\n\n"
            "궁금한 상품이 있으면 이 채팅창을 끄고\n"
            "상품 페이지에서 다시 채팅창을 열어주세요 :)"
        )

    st.session_state.messages.append({"role": "assistant", "content": welcome})

st.divider()

for msg in st.session_state.messages:
    safe_text = html.escape(msg["content"]).replace("\n", "<br>")
    if msg["role"] == "user":
        st.markdown(
            (
                '<div style="display:flex; justify-content:flex-end; width:100%; margin:2px 0 4px 0;">'
                '<div style="max-width:92%;">'
                '<div style="display:block; font-size:12px; font-weight:700; line-height:1.15; color:#0f6a63; text-align:right; margin:0 6px 1px 0;">고객님</div>'
                f'<div style="padding:10px 14px 10px 10px; border-radius:18px; border-bottom-right-radius:6px; font-size:15px; line-height:1.5; white-space:pre-wrap; word-break:keep-all; background:#dff0ec; color:#1f3b36; border:1px solid rgba(15,106,99,.14);">{safe_text}</div>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            (
                '<div style="display:flex; justify-content:flex-start; width:100%; margin:2px 0 4px 0;">'
                '<div style="max-width:92%;">'
                '<div style="display:block; font-size:12px; font-weight:700; line-height:1.15; color:#5f6471; margin:0 0 1px 6px;">미야언니</div>'
                f'<div style="padding:10px 14px 10px 10px; border-radius:18px; border-bottom-left-radius:6px; font-size:15px; line-height:1.5; white-space:pre-wrap; word-break:keep-all; background:#071b4e; color:#ffffff; border:1px solid rgba(255,255,255,.08);">{safe_text}</div>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True
        )

user_input = st.chat_input("메시지를 입력하세요…")
if user_input:
    # refresh similar products after size input
    similar_products = get_similar_products(DB, db_product, clean_text(st.session_state.body_top), limit=3)
    process_user_message(user_input, current_url, product_no, page_context, db_product, similar_products, available_colors)
    st.rerun()
