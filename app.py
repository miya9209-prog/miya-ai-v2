
import os
import re
import json
import html
import requests
import pandas as pd
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
- 반품을 줄이는 방향으로 솔직하고 안전하게 말한다.
- 고객이 특정 아이템 추천을 원하면 DB 안에서 실제 상품명을 골라 이유와 함께 제안한다.

말투 규칙:
- 친근한 MD 상담체
- "옵션 중", "표기상", "기준으로 보면", "가능성이 높아요" 같은 딱딱한 표현은 되도록 쓰지 않는다.
- "이 상품은", "고객님 체형이면", "같이 입으시면"처럼 부드럽고 자연스럽게 말한다.
- 상품명이 확실할 때만 상품명을 쓴다.
- 상품명이 불확실하면 "지금 보시는 상품"이라고 말한다.
- 고객 체형 정보가 있으면 꼭 참고해서 말한다.
- 정보가 부족하면 짧게 필요한 부분만 다시 물어본다.

중요 규칙:
- 상품명은 반드시 제공된 DB 추천 후보 안에서만 말한다. 없는 미샵 상품명을 절대 만들지 않는다.
- 컬러는 확인된 옵션 안에서만 말한다. 없는 컬러를 추측해서 말하지 않는다.
- 현재 상품의 사이즈 제한을 넘는 고객에게 "잘 맞는다", "여유 있다", "추천드린다"라고 말하지 않는다.
- 가격/옵션/스펙은 현재 페이지와 제공된 데이터 기준으로만 말하고 지어내지 않는다.
- 추천 상품을 말할 때는 왜 어울리는지 1~2가지 이유를 같이 붙인다.

답변 스타일:
- 3~7문장 내외
- 먼저 질문에 바로 답하고
- 이어서 이유를 자연스럽게 풀어주고
- 추천이 필요한 질문이면 상품명 1~3개를 실제로 제시한다
- 마지막에는 필요할 때만 짧게 추가 질문을 붙인다

배송/교환 규칙:
- 정책 관련 답변은 반드시 POLICY_DB 기준으로만 말한다
"""

GENERIC_NAMES = {"미샵", "misharp", "MISHARP", "미샵여성", "Misharp"}
SIZE_ORDER = {"44": 1, "55": 2, "55반": 3, "66": 4, "66반": 5, "77": 6, "77반": 7, "88": 8, "99": 9}
SIZE_LABELS = {v: k for k, v in SIZE_ORDER.items()}


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


ensure_state()

qp = st.query_params
current_url = qp.get("url", "") or ""
product_no = qp.get("pn", "") or ""
product_name_q = qp.get("pname", "") or ""

context_key = f"{current_url}|{product_no}|{product_name_q}"
if context_key != st.session_state.last_context_key:
    st.session_state.last_context_key = context_key
    st.session_state.messages = []


def clean_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def is_generic_name(name: str) -> bool:
    if not name:
        return True
    name = clean_text(name)
    return name in GENERIC_NAMES or len(name) <= 2


def normalize_product_no(value: str) -> str:
    value = clean_text(value)
    if value.endswith('.0'):
        value = value[:-2]
    return value


@st.cache_data(ttl=600, show_spinner=False)
def load_product_db():
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


def get_db_product(product_no_value: str):
    if DB.empty or not product_no_value or "product_no" not in DB.columns:
        return None
    target = normalize_product_no(product_no_value)
    rows = DB[DB["product_no"] == target]
    if len(rows) == 0:
        return None
    return rows.iloc[0].to_dict()


def split_sections(text: str) -> dict:
    if not text:
        return {"summary": "", "material": "", "fit": "", "size_tip": "", "shipping": ""}

    lines = [clean_text(x) for x in text.split("\n")]
    lines = [x for x in lines if x]
    joined = "\n".join(lines)

    def extract_by_keywords(keywords, max_len=1600):
        matched = []
        for line in lines:
            if any(k in line for k in keywords):
                matched.append(line)
        return " / ".join(matched)[:max_len]

    return {
        "summary": joined[:3000],
        "material": extract_by_keywords(["소재", "원단", "혼용", "%", "면", "폴리", "레이온", "아크릴", "울", "스판", "비스코스", "나일론"]),
        "fit": extract_by_keywords(["핏", "여유", "라인", "체형", "복부", "팔뚝", "허벅지", "힙", "루즈", "와이드", "슬림", "정핏", "세미", "커버"]),
        "size_tip": extract_by_keywords(["사이즈", "정사이즈", "추천", "44", "55", "55반", "66", "66반", "77", "77반", "88", "99", "FREE", "L(", "M(", "S(", "XL(", "F("]),
        "shipping": extract_by_keywords(["배송", "출고", "교환", "반품", "배송비"])
    }


def guess_category(name: str, text: str) -> str:
    corpus = f"{name} {text}"
    mapping = {
        "슬랙스": ["슬랙스", "팬츠", "바지"],
        "블라우스": ["블라우스"],
        "셔츠": ["셔츠"],
        "티셔츠": ["티셔츠", "탑", "맨투맨"],
        "니트": ["니트", "가디건"],
        "자켓": ["자켓", "재킷"],
        "원피스": ["원피스"],
        "데님": ["데님", "청바지"],
        "코트": ["코트"],
        "맨투맨": ["맨투맨"],
    }
    for cat, keywords in mapping.items():
        if any(k in corpus for k in keywords):
            return cat
    return "기타"


def fetch_product_context(url: str, passed_name: str = "") -> dict | None:
    if not url:
        return None

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=12)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    raw_text = soup.get_text("\n")
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

    product_name = clean_text(passed_name)
    if is_generic_name(product_name):
        product_name = "지금 보시는 상품"

    sections = split_sections(raw_text)
    category = guess_category(product_name, raw_text)

    return {
        "product_name": product_name,
        "category": category,
        "summary": sections["summary"],
        "material": sections["material"],
        "fit": sections["fit"],
        "size_tip": sections["size_tip"],
        "shipping": sections["shipping"],
        "raw_excerpt": raw_text[:6000]
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


def size_rank(token: str):
    return SIZE_ORDER.get(clean_text(token), None)


def expand_size_text(size_text: str):
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
    m = re.findall(r"(44|55반|55|66반|66|77반|77|88|99)\s*-\s*(44|55반|55|66반|66|77반|77|88|99)", text)
    for a, b in m:
        ra, rb = size_rank(a), size_rank(b)
        if ra and rb and ra <= rb:
            found.extend(list(range(ra, rb + 1)))
    m2 = re.search(r"(44|55반|55|66반|66|77반|77|88|99)\s*까지", text)
    if m2:
        rb = size_rank(m2.group(1))
        ra = size_rank("55")
        if ra and rb and ra <= rb:
            found.extend(list(range(ra, rb + 1)))
    return sorted(set(found))


def parse_page_size_options(product_context: dict):
    text = " ".join([
        clean_text((product_context or {}).get("size_tip", "")),
        clean_text((product_context or {}).get("summary", "")),
        clean_text((product_context or {}).get("raw_excerpt", ""))[:2500],
    ])
    text = text.replace("\n", " ")
    options = []
    seen = set()

    patterns = [
        r"([A-Za-z가-힣]+)\s*\((44|55반|55|66반|66|77반|77|88|99)\s*-\s*(44|55반|55|66반|66|77반|77|88|99)\)",
        r"([A-Za-z가-힣]+)\s*\((44|55반|55|66반|66|77반|77|88|99)\)",
    ]
    for pat in patterns:
        for match in re.finditer(pat, text):
            label = clean_text(match.group(1))
            size_desc = clean_text(match.group(0).split("(", 1)[1].rstrip(")"))
            ranks = expand_size_text(size_desc)
            if label and ranks:
                key = (label, tuple(ranks))
                if key not in seen:
                    seen.add(key)
                    options.append({"label": label, "size_desc": size_desc, "ranks": ranks})
    # normalize weird labels
    clean_opts = []
    for opt in options:
        label = opt["label"].upper()
        if label in {"COLOR", "SIZE", "OPTION", "옵션", "컬러"}:
            continue
        clean_opts.append(opt)
    return clean_opts


def parse_color_options(product_context: dict, db_product: dict | None):
    candidates = []
    if db_product and db_product.get("color_options"):
        for part in re.split(r"[;,/|]", db_product.get("color_options", "")):
            t = clean_text(part)
            if t:
                candidates.append(t)
    text = clean_text((product_context or {}).get("raw_excerpt", ""))
    for color in ["블랙", "화이트", "아이보리", "그레이", "베이지", "핑크", "네이비", "카키", "브라운", "소라", "블루", "레드", "옐로우", "민트"]:
        if color in text:
            candidates.append(color)
    out = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def find_best_size_option(user_top: str, product_context: dict, db_product: dict | None):
    user_rank = size_rank(user_top)
    if not user_rank:
        return {"supported": None, "reason": "", "matched_option": None, "options": []}

    options = parse_page_size_options(product_context)
    if options:
        for opt in options:
            if user_rank in opt["ranks"]:
                return {"supported": True, "reason": f"현재 페이지 기준으로 {opt['label']} 사이즈가 고객님 상의를 커버해요.", "matched_option": opt, "options": options}
        max_rank = max(max(o["ranks"]) for o in options)
        return {"supported": False, "reason": f"현재 페이지 기준으로 가능한 최대 사이즈가 {SIZE_LABELS.get(max_rank, '')}까지로 보여요.", "matched_option": None, "options": options}

    db_range = clean_text((db_product or {}).get("size_range", ""))
    ranks = expand_size_text(db_range)
    if ranks:
        if user_rank in ranks:
            return {"supported": True, "reason": f"DB 기준으로 고객님 상의 사이즈는 권장 범위 안에 있어요.", "matched_option": None, "options": []}
        max_rank = max(ranks)
        return {"supported": False, "reason": f"DB 기준으로 가능한 최대 사이즈가 {SIZE_LABELS.get(max_rank, '')}까지로 보여요.", "matched_option": None, "options": []}

    return {"supported": None, "reason": "", "matched_option": None, "options": []}


def is_size_question(user_text: str):
    q = clean_text(user_text).replace(" ", "")
    keywords = ["사이즈", "맞을까", "맞나요", "맞아", "큰가", "작을까", "작아요", "타이트", "여유", "l사이즈", "f사이즈", "free", "88", "77반", "66반"]
    return any(k in q for k in keywords)


def is_color_question(user_text: str):
    q = clean_text(user_text)
    keywords = ["컬러", "색상", "무슨색", "어떤색", "색감", "블랙", "그레이", "베이지", "핑크", "아이보리"]
    return any(k in q for k in keywords)


def is_recommendation_question(user_text: str):
    q = clean_text(user_text)
    keywords = ["추천", "골라", "어울리는", "같이 입", "코디", "매치", "어떤 바지", "무슨 바지", "무슨 치마", "학교방문룩", "출근룩", "하객룩"]
    return any(k in q for k in keywords)


def infer_target_category_from_query(user_text: str):
    q = clean_text(user_text)
    mapping = [
        ("팬츠", ["바지", "슬랙스", "팬츠", "데님", "청바지"]),
        ("스커트", ["스커트", "치마"]),
        ("자켓", ["자켓", "재킷", "아우터"]),
        ("가디건", ["가디건"]),
        ("니트", ["니트"]),
        ("셔츠", ["셔츠"]),
        ("블라우스", ["블라우스"]),
        ("원피스", ["원피스"]),
    ]
    for target, words in mapping:
        if any(w in q for w in words):
            return target
    return ""


def build_product_reason(rowd: dict, current_product: dict | None, user_text: str):
    reasons = []
    fit = clean_text(rowd.get("fit_type", ""))
    cover = clean_text(rowd.get("body_cover_features", ""))
    style = clean_text(rowd.get("style_tags", ""))
    coord = clean_text(rowd.get("coordination_items", ""))
    summary = clean_text(rowd.get("product_summary", ""))

    if "학교" in user_text or "방문" in user_text:
        if any(k in style for k in ["클래식", "데일리", "페미닌"]):
            reasons.append("학교 방문룩으로 깔끔하게 입기 좋아요")
        else:
            reasons.append("과하게 튀지 않아서 단정하게 매치하기 좋아요")

    if fit:
        fit_map = {
            "정핏": "핏이 단정하게 떨어져서",
            "세미루즈": "너무 붙지 않게 떨어져서",
            "루즈": "편안하게 입기 좋고",
            "세미와이드": "다리라인을 편하게 커버해줘서",
            "와이드": "하체라인을 자연스럽게 정리해줘서",
        }
        for key, sentence in fit_map.items():
            if key in fit:
                reasons.append(sentence)
                break

    if cover:
        if "뱃살커버" in cover:
            reasons.append("복부라인 부담을 덜어줘요")
        elif "힙커버" in cover:
            reasons.append("힙라인이 드러나는 부담이 적어요")
        elif "허리라인보정" in cover:
            reasons.append("허리선이 좀 더 정돈돼 보여요")

    if not reasons and summary:
        reasons.append(summary[:40])

    # dedupe keep order
    out=[]
    seen=set()
    for r in reasons:
        r=clean_text(r)
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out[:2]


def recommend_products_for_query(user_text: str, current_product: dict | None, body_ctx: dict, limit=3):
    if DB.empty:
        return []

    target = infer_target_category_from_query(user_text)
    current_no = normalize_product_no((current_product or {}).get("product_no", ""))
    bottom_size = clean_text(body_ctx.get("bottom_size", ""))
    top_size = clean_text(body_ctx.get("top_size", ""))
    bottom_rank = size_rank(bottom_size) if bottom_size else None
    top_rank = size_rank(top_size) if top_size else None

    candidates = DB.copy()
    if current_no and "product_no" in candidates.columns:
        candidates["product_no"] = candidates["product_no"].map(normalize_product_no)
        candidates = candidates[candidates["product_no"] != current_no]

    if target:
        mask = (candidates.get("category", "").astype(str).str.contains(target, na=False)) | (candidates.get("sub_category", "").astype(str).str.contains(target, na=False))
        filtered = candidates[mask]
        if len(filtered) > 0:
            candidates = filtered

    current_coord = clean_text((current_product or {}).get("coordination_items", ""))
    current_style = clean_text((current_product or {}).get("style_tags", ""))

    scored = []
    for _, row in candidates.iterrows():
        rowd = row.to_dict()
        name = clean_text(rowd.get("product_name", ""))
        if not name:
            continue

        size_text = clean_text(rowd.get("size_range", ""))
        ranks = expand_size_text(size_text)
        if target in ["팬츠", "스커트"] and bottom_rank and ranks and bottom_rank not in ranks:
            continue
        if target not in ["팬츠", "스커트"] and top_rank and ranks and top_rank not in ranks:
            continue

        score = 0
        row_cat = clean_text(rowd.get("category", ""))
        row_sub = clean_text(rowd.get("sub_category", ""))
        if target and (target in row_cat or target in row_sub):
            score += 6
        if current_coord and any(x and x in current_coord for x in [row_cat, row_sub, target]):
            score += 4
        if current_style and clean_text(rowd.get("style_tags", "")):
            overlap = set([clean_text(x) for x in re.split(r"[;,/|]", current_style) if clean_text(x)]) & set([clean_text(x) for x in re.split(r"[;,/|]", clean_text(rowd.get("style_tags", ""))) if clean_text(x)])
            score += len(overlap) * 2
        if clean_text(rowd.get("recommended_age", "")) in ["4050", "40", "50"]:
            score += 1
        if clean_text(rowd.get("body_cover_features", "")):
            score += 1
        scored.append((score, rowd))

    scored.sort(key=lambda x: x[0], reverse=True)
    out=[]
    seen=set()
    for score, rowd in scored:
        name = clean_text(rowd.get("product_name", ""))
        if name in seen:
            continue
        seen.add(name)
        out.append({
            "product_name": name,
            "category": clean_text(rowd.get("category", "")),
            "sub_category": clean_text(rowd.get("sub_category", "")),
            "size_range": clean_text(rowd.get("size_range", "")),
            "reasons": build_product_reason(rowd, current_product, user_text),
        })
        if len(out) >= limit:
            break
    return out


def build_recommendation_answer(user_text: str, product_context: dict | None, db_product: dict | None):
    if not is_recommendation_question(user_text):
        return None

    body_ctx = build_body_context()
    recos = recommend_products_for_query(user_text, db_product, body_ctx, limit=3)
    if not recos:
        return None

    target = infer_target_category_from_query(user_text)
    target_kor = {"팬츠":"바지", "스커트":"스커트", "자켓":"자켓", "가디건":"가디건", "니트":"니트", "셔츠":"셔츠", "블라우스":"블라우스", "원피스":"원피스"}.get(target, "아이템")
    opener = f"이런 자리에는 {target_kor}를 너무 힘줘 보이기보다 깔끔하게 잡아주는 쪽이 잘 어울려요." if target else "이럴 때는 지금 보시는 상품이랑 톤이 맞는 아이템으로 같이 보시는 게 좋아요."

    lines = [opener, "미샵에서 같이 보기 좋은 상품으로는 아래 쪽을 먼저 추천드릴게요."]
    for r in recos:
        reason = " / ".join(r["reasons"]) if r.get("reasons") else "무난하게 매치하기 좋아요"
        size_tail = f" · {r['size_range']}" if r.get("size_range") else ""
        lines.append(f"- {r['product_name']}{size_tail}: {reason}")

    if target in ["팬츠", "스커트"] and clean_text(body_ctx.get("bottom_size", "")):
        lines.append(f"고객님 하의 {body_ctx.get('bottom_size')} 기준으로 너무 타이트한 느낌보다 편하게 떨어지는 핏 위주로 골랐어요.")

    return "\n".join(lines)



def recommend_alternative_products(db_product: dict | None, user_top: str, limit=3):
    if DB.empty or not db_product or not user_top:
        return []
    user_rank = size_rank(user_top)
    if not user_rank:
        return []
    category = clean_text(db_product.get("category", ""))
    sub_category = clean_text(db_product.get("sub_category", ""))
    current_no = normalize_product_no(db_product.get("product_no", ""))

    candidates = DB.copy()
    if category:
        candidates = candidates[candidates["category"] == category]
    if sub_category:
        same_sub = candidates[candidates["sub_category"] == sub_category]
        if len(same_sub) > 0:
            candidates = same_sub
    if current_no:
        candidates = candidates[candidates["product_no"] != current_no]

    scored = []
    for _, row in candidates.iterrows():
        rowd = row.to_dict()
        ranks = expand_size_text(rowd.get("size_range", ""))
        if not ranks:
            continue
        if user_rank not in ranks:
            continue
        score = 0
        if clean_text(rowd.get("fit_type", "")) and clean_text(rowd.get("fit_type", "")) == clean_text(db_product.get("fit_type", "")):
            score += 2
        if clean_text(rowd.get("style_tags", "")) and clean_text(db_product.get("style_tags", "")):
            overlap = set(re.split(r"[;,/|]", rowd.get("style_tags", ""))) & set(re.split(r"[;,/|]", db_product.get("style_tags", "")))
            score += len([x for x in overlap if clean_text(x)])
        score += max(ranks)
        scored.append((score, rowd))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    seen = set()
    for _, rowd in scored:
        name = clean_text(rowd.get("product_name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        out.append({
            "product_name": name,
            "size_range": clean_text(rowd.get("size_range", "")),
            "fit_type": clean_text(rowd.get("fit_type", "")),
            "body_cover_features": clean_text(rowd.get("body_cover_features", "")),
        })
        if len(out) >= limit:
            break
    return out


def build_size_guard_answer(user_text: str, product_context: dict | None, db_product: dict | None):
    body = build_body_context()
    user_top = body.get("top_size", "")
    if not user_top or not is_size_question(user_text):
        return None

    size_eval = find_best_size_option(user_top, product_context, db_product)
    supported = size_eval.get("supported")
    current_name = clean_text((db_product or {}).get("product_name", "")) or clean_text((product_context or {}).get("product_name", "")) or "지금 보시는 상품"

    if supported is False:
        recos = recommend_alternative_products(db_product, user_top, limit=3)
        lines = [
            f"고객님 상의 {user_top} 기준이면 지금 보시는 상품은 편하게 맞는다고 보긴 어려워요.",
            size_eval.get("reason", "현재 확인되는 사이즈 범위가 조금 작게 보여요."),
        ]
        if recos:
            lines.append("대신 같은 분위기에서 고객님 사이즈를 더 안정적으로 커버하는 상품으로 같이 골라드릴게요.")
            for r in recos:
                tail = []
                if r.get("size_range"):
                    tail.append(r["size_range"])
                if r.get("fit_type"):
                    tail.append(r["fit_type"])
                if r.get("body_cover_features"):
                    tail.append(r["body_cover_features"])
                tail_text = " / ".join([x for x in tail if x])
                if tail_text:
                    lines.append(f"- {r['product_name']} ({tail_text})")
                else:
                    lines.append(f"- {r['product_name']}")
        else:
            lines.append("이럴 땐 한 사이즈 더 여유 있게 나오는 맨투맨이나 상의를 같이 보는 쪽이 더 안전해요.")
        return "\n".join(lines)

    if supported is True and size_eval.get("matched_option"):
        opt = size_eval["matched_option"]
        return (
            f"고객님 상의 {user_top} 기준이면 현재 페이지에 있는 옵션 중 {opt['label']} 쪽으로 보시는 게 가장 자연스러워요.\n"
            f"지금 보이는 사이즈 표기상 {opt['label']}는 {opt['size_desc']} 기준이라 고객님 체형에 더 가깝게 맞을 가능성이 높아요.\n"
            f"너무 딱 맞는 느낌보다 편안함을 원하시면 이 옵션을 우선 보시는 쪽이 좋아요."
        )

    return None


def build_color_guard_answer(user_text: str, product_context: dict | None, db_product: dict | None):
    if not is_color_question(user_text):
        return None
    colors = parse_color_options(product_context, db_product)
    if not colors:
        return "현재 확인되는 컬러 옵션이 분명하지 않아서 없는 색을 추측해서 말씀드리긴 어려워요. 상세페이지 옵션창에 보이는 색상 기준으로 다시 같이 봐드릴게요 :)"
    return f"현재 확인되는 컬러는 {', '.join(colors)} 쪽이에요. 없는 색을 추측해서 말씀드리기보다는 지금 보이는 옵션 기준으로 같이 봐드릴게요 :)"


def get_llm_answer(user_text: str, current_url: str, product_no_value: str, product_context: dict | None, db_product: dict | None) -> str:
    body_context = build_body_context()
    confirmed_colors = parse_color_options(product_context, db_product)
    alt_products = recommend_alternative_products(db_product, body_context.get("top_size", ""), limit=3)

    context_pack = {
        "policy_db": POLICY_DB,
        "viewer_context": {
            "url": current_url,
            "is_product_page": bool(product_no_value),
            "product_no": product_no_value
        },
        "body_context": body_context,
        "product_context": product_context,
        "db_product": db_product,
        "confirmed_colors": confirmed_colors,
        "alternative_products": alt_products,
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
        temperature=0.55,
        max_tokens=420
    )
    return resp.choices[0].message.content.strip()


def process_user_message(user_text: str, current_url: str, product_no_value: str, product_context: dict | None, db_product: dict | None):
    st.session_state.messages.append({"role": "user", "content": user_text})

    fast = get_fast_policy_answer(user_text)
    if fast:
        st.session_state.messages.append({"role": "assistant", "content": fast})
        return

    size_guard = build_size_guard_answer(user_text, product_context, db_product)
    if size_guard:
        st.session_state.messages.append({"role": "assistant", "content": size_guard})
        return

    rec_guard = build_recommendation_answer(user_text, product_context, db_product)
    if rec_guard:
        st.session_state.messages.append({"role": "assistant", "content": rec_guard})
        return

    color_guard = build_color_guard_answer(user_text, product_context, db_product)
    if color_guard and ("무슨" in user_text or "어떤" in user_text or "컬러" in user_text or "색상" in user_text):
        st.session_state.messages.append({"role": "assistant", "content": color_guard})
        return

    answer = get_llm_answer(user_text, current_url, product_no_value, product_context, db_product)
    st.session_state.messages.append({"role": "assistant", "content": answer})


product_context = fetch_product_context_cached(current_url, product_name_q) if current_url else None
db_product = get_db_product(product_no)

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
  --miya-muted:#8f94a3;
  --miya-divider:#ccccd2;
  --miya-bot-bg:#071b4e;
  --miya-user-bg:#dff0ec;
  --miya-user-text:#1f3b36;
  --miya-page-bg:#f6f7fb;
  --miya-input-bg:#1f2537;
  --miya-input-text:#f7f9ff;
  --miya-input-placeholder:#d2d7e6;
  --miya-input-shell:#e7eaf2;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
  color: var(--miya-title);
  background: var(--miya-page-bg) !important;
}
[data-testid="stAppViewContainer"] > .main {background: var(--miya-page-bg) !important;}
.block-container{background: var(--miya-page-bg) !important;}

div[data-testid="column"]{
  min-width:0 !important;
}

div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label{
  color:var(--miya-title) !important;
  font-weight:700 !important;
  font-size:11.5px !important;
}

div[data-testid="stTextInput"] input,
div[data-baseweb="select"] > div{
  border-radius:12px !important;
}

div[data-testid="stTextInput"],
div[data-testid="stSelectbox"]{
  margin-bottom:-2px !important;
}

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
  background: transparent !important;
}

div[data-testid="stChatInput"] > div{
  background: transparent !important;
  border-radius: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
  border: none !important;
}

div[data-testid="stChatInput"] textarea {
  background: #1f2740 !important;
  color: #ffffff !important;
  caret-color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-size: 16px !important;
  line-height: 1.35 !important;
  padding-top: 12px !important;
  padding-bottom: 12px !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
  color: #cfd6e6 !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #cfd6e6 !important;
}

div[data-testid="stChatInput"] [data-baseweb="textarea"] {
  background: #1f2740 !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  min-height: 52px !important;
  padding: 0 10px !important;
  display: flex !important;
  align-items: center !important;
}

div[data-testid="stChatInput"] [data-baseweb="textarea"] > div {
  background: transparent !important;
  display: flex !important;
  align-items: center !important;
}

div[data-testid="stChatInput"] button {
  background: #2f3a5f !important;
  color: #ffffff !important;
  border-radius: 14px !important;
}

div[data-testid="stChatInput"] button svg {
  fill: #ffffff !important;
}

@media (max-width: 768px){
  .block-container{
    max-width:100%;
    padding-top:0.14rem !important;
    padding-bottom:11.6rem !important;
  }

  div[data-testid="stHorizontalBlock"]{
    gap:6px !important;
  }

  div[data-testid="stHorizontalBlock"] > div{
    flex:1 1 0 !important;
    min-width:0 !important;
  }

  div[data-testid="stTextInput"] label,
  div[data-testid="stSelectbox"] label{
    font-size:11px !important;
  }

  div[data-testid="stTextInput"],
  div[data-testid="stSelectbox"]{
    margin-bottom:-4px !important;
  }

  hr{
    margin-top:3px !important;
    margin-bottom:3px !important;
  }

  div[data-testid="stChatInput"]{
    bottom:64px !important;
    width:calc(100% - 16px) !important;
  }
  div[data-testid="stChatInput"] > div{
    padding: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
  }
  div[data-testid="stChatInput"] [data-baseweb="textarea"]{
    min-height: 48px !important;
    padding: 0 10px !important;
    display: flex !important;
    align-items: center !important;
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
    st.session_state.body_height = st.text_input(
        "키",
        value=st.session_state.body_height,
        placeholder="cm",
        key="body_height_input"
    )
with row1[1]:
    st.session_state.body_weight = st.text_input(
        "체중",
        value=st.session_state.body_weight,
        placeholder="kg",
        key="body_weight_input"
    )

size_options = ["", "44", "55", "55반", "66", "66반", "77", "77반", "88"]

row2 = st.columns(2, gap="small")
with row2[0]:
    current_top = st.session_state.body_top if st.session_state.body_top in size_options else ""
    st.session_state.body_top = st.selectbox(
        "상의",
        options=size_options,
        index=size_options.index(current_top),
        key="body_top_input"
    )
with row2[1]:
    current_bottom = st.session_state.body_bottom if st.session_state.body_bottom in size_options else ""
    st.session_state.body_bottom = st.selectbox(
        "하의",
        options=size_options,
        index=size_options.index(current_bottom),
        key="body_bottom_input"
    )

st.markdown("</div></div>", unsafe_allow_html=True)

body_summary = build_body_context_text(build_body_context())
if any(build_body_context().values()):
    st.markdown(
        f'<div style="margin-top:2px; margin-bottom:2px; font-size:10.8px; color:#7a7f8c;">현재 입력 정보: {html.escape(body_summary)}</div>',
        unsafe_allow_html=True
    )

if not st.session_state.messages:
    current_url_lower = (current_url or "").lower()

    is_detail_page = (
        ("/product/detail" in current_url_lower) or
        ("product_no=" in current_url_lower) or
        bool(product_no)
    )

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

    st.session_state.messages.append({
        "role": "assistant",
        "content": welcome
    })

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
    process_user_message(user_input, current_url, product_no, product_context, db_product)
    st.rerun()
