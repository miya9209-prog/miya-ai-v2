import os
import re
import json
import html
import time
from difflib import SequenceMatcher

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

client = OpenAI(api_key=OPENAI_API_KEY, timeout=20.0, max_retries=1)

POLICY_DB = {
    "shipping": {
        "courier": "CJ 대한통운",
        "shipping_fee": 3000,
        "free_shipping_over": 70000,
        "delivery_time": "결제 완료 후 2~4영업일 정도",
        "same_day_dispatch_rule": "오후 2시 이전 주문은 당일 출고",
        "combined_shipping": "합배송 가능(1박스 기준, 박스 크기 초과 시 합배송 불가)",
    },
    "exchange_return": {
        "period": "상품 수령 후 7일 이내",
        "exchange_fee": 6000,
        "return_fee_rule": "단순 변심 반품은 반품 후 주문금액이 7만원 이상이면 편도 3,000원, 7만원 미만이면 왕복 6,000원",
        "defect_wrong": "불량·오배송은 미샵 부담",
    },
}

SYSTEM_PROMPT = """
너는 미샵 쇼핑친구 미야언니다.
말투는 친근한 MD 상담체로, 짧고 정확하게 답한다.

절대 규칙:
1) 현재 상품명은 current_product.name 에 있는 이름만 사용한다. 다른 상품명을 만들거나 바꿔 말하지 않는다.
2) 추천 상품명은 recommendation_candidates 안에 있는 이름만 사용한다. 없으면 상품명을 말하지 않는다.
3) 사이즈는 confirmed_size_range / option_size_desc / measurements 안에서 확인된 정보만 근거로 말한다.
4) 확인되지 않은 색상, 옵션, 가격, 사이즈, 실측은 추측하지 않는다.
5) 정보가 애매하면 솔직하게 "현재 페이지 기준으로는 여기까지 확인된다"고 말한다.
6) 고객이 현재 상품 이름을 물으면 current_product.name을 그대로 답한다.

답변 스타일:
- 2~5문장
- 먼저 결론, 다음에 근거
- 고객 체형 정보가 있으면 반영
- 과장 없이 안전하게
""".strip()

SIZE_ORDER = {"44": 1, "55": 2, "55반": 3, "66": 4, "66반": 5, "77": 6, "77반": 7, "88": 8, "99": 9}
SIZE_LABELS = {v: k for k, v in SIZE_ORDER.items()}
COLOR_WORDS = ["블랙", "화이트", "아이보리", "그레이", "베이지", "핑크", "네이비", "카키", "브라운", "소라", "블루", "레드", "옐로우", "민트"]
BLOCKED_PAGE_WORDS = {"LOGIN", "JOIN", "ABOUT", "SHOP", "COMMUNITY", "MY PAGE", "MYPAGE", "CART", "KRW"}


# ---------- helpers ----------
def clean_text(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def norm_name(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^0-9a-z가-힣]+", "", text)
    return text


def normalize_product_no(value: str) -> str:
    s = clean_text(value)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def extract_product_no(url: str) -> str:
    if not url:
        return ""
    m = re.search(r"(?:product_no|product_no=|product_no%3D)(\d+)", url)
    return m.group(1) if m else ""


def dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        item = clean_text(item)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def size_rank(token: str):
    return SIZE_ORDER.get(clean_text(token), None)


def expand_size_text(size_text: str):
    text = clean_text(size_text).replace("~", "-")
    if not text:
        return []
    ordered = ["44", "55반", "55", "66반", "66", "77반", "77", "88", "99"]
    found = []
    for token in ordered:
        if token in text:
            rank = size_rank(token)
            if rank:
                found.append(rank)
    for a, b in re.findall(r"(44|55반|55|66반|66|77반|77|88|99)\s*-\s*(44|55반|55|66반|66|77반|77|88|99)", text):
        ra, rb = size_rank(a), size_rank(b)
        if ra and rb and ra <= rb:
            found.extend(range(ra, rb + 1))
    m = re.search(r"(44|55반|55|66반|66|77반|77|88|99)\s*까지", text)
    if m:
        rb = size_rank(m.group(1))
        if rb:
            found.extend(range(2, rb + 1))
    return sorted(set(found))


def ranks_to_text(ranks):
    if not ranks:
        return ""
    return f"{SIZE_LABELS[min(ranks)]}-{SIZE_LABELS[max(ranks)]}" if len(ranks) > 1 else SIZE_LABELS[ranks[0]]


def infer_category_from_name(name: str) -> str:
    corpus = clean_text(name)
    mapping = [
        ("팬츠", ["슬랙스", "팬츠", "바지", "데님", "청바지"]),
        ("스커트", ["스커트", "치마"]),
        ("원피스", ["원피스"]),
        ("자켓", ["자켓", "재킷"]),
        ("가디건", ["가디건"]),
        ("니트", ["니트"]),
        ("블라우스", ["블라우스"]),
        ("셔츠", ["셔츠"]),
        ("티셔츠", ["맨투맨", "티셔츠", "탑"]),
    ]
    for cat, keywords in mapping:
        if any(k in corpus for k in keywords):
            return cat
    return "기타"


# ---------- state ----------
def ensure_state():
    defaults = {
        "messages": [],
        "last_context_key": "",
        "body_height": "",
        "body_weight": "",
        "body_top": "",
        "body_bottom": "",
        "is_processing": False,
        "last_question_hash": "",
        "last_answer": "",
        "last_question_at": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_state()


# ---------- db ----------
@st.cache_data(ttl=600, show_spinner=False)
def load_product_db():
    path = "misharp_miya_db.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [clean_text(c) for c in df.columns]
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str).map(clean_text)
    if "product_no" in df.columns:
        df["product_no"] = df["product_no"].map(normalize_product_no)
    if "product_name" in df.columns:
        df["product_name_norm"] = df["product_name"].map(norm_name)
    return df


DB = load_product_db()


def get_db_product(product_no: str, page_name: str = ""):
    if DB.empty:
        return None
    pno = normalize_product_no(product_no)
    if pno and "product_no" in DB.columns:
        rows = DB[DB["product_no"] == pno]
        if len(rows):
            return rows.iloc[0].to_dict()

    page_name_norm = norm_name(page_name)
    if page_name_norm and "product_name_norm" in DB.columns:
        exact = DB[DB["product_name_norm"] == page_name_norm]
        if len(exact):
            return exact.iloc[0].to_dict()
        best = None
        best_score = 0.0
        for _, row in DB.iterrows():
            cand = row.to_dict()
            score = SequenceMatcher(None, page_name_norm, cand.get("product_name_norm", "")).ratio()
            if score > best_score:
                best_score = score
                best = cand
        if best and best_score >= 0.90:
            return best
    return None


# ---------- page parsing ----------
def best_meta_content(soup: BeautifulSoup, selectors):
    for attr_name, attr_value in selectors:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            return clean_text(tag.get("content"))
    return ""


def sanitize_page_name(name: str) -> str:
    name = html.unescape(clean_text(name))
    name = re.sub(r"\s*\|\s*.*$", "", name)
    name = re.sub(r"\s*-\s*미샵.*$", "", name)
    if any(b in name.upper() for b in BLOCKED_PAGE_WORDS):
        return ""
    if len(name) < 3:
        return ""
    return name


def extract_name_candidates(soup: BeautifulSoup):
    candidates = []
    candidates.append(best_meta_content(soup, [("property", "og:title"), ("name", "og:title")]))
    if soup.title:
        candidates.append(clean_text(soup.title.get_text(" ", strip=True)))
    for selector in ["h1", ".name", ".prdName", ".headingArea h2", ".headingArea h1", ".infoArea .name"]:
        for tag in soup.select(selector):
            text = clean_text(tag.get_text(" ", strip=True))
            if text:
                candidates.append(text)
    return [sanitize_page_name(x) for x in candidates if sanitize_page_name(x)]


def extract_measurements(text: str):
    patterns = {
        "어깨": [r"어깨단면\s*([0-9.]+)", r"어깨\s*([0-9.]+)"],
        "가슴": [r"가슴둘레\s*([0-9.]+)", r"가슴단면\s*([0-9.]+)", r"가슴\s*([0-9.]+)"],
        "암홀": [r"암홀둘레\s*([0-9.]+)", r"암홀\s*([0-9.]+)"],
        "소매": [r"소매길이\s*([0-9.]+)", r"소매\s*([0-9.]+)"],
        "총장": [r"총장\s*([0-9.]+)", r"길이\s*([0-9.]+)"],
    }
    out = {}
    for label, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, text)
            if m:
                out[label] = m.group(1)
                break
    return out


def parse_size_candidates(text: str):
    ranks = []
    descs = []
    for match in re.findall(r"(44|55반|55|66반|66|77반|77|88|99)\s*-\s*(44|55반|55|66반|66|77반|77|88|99)", text):
        desc = f"{match[0]}-{match[1]}"
        descs.append(desc)
        ranks.extend(expand_size_text(desc))
    for token in ["FREE", "프리", "44", "55", "55반", "66", "66반", "77", "77반", "88", "99"]:
        if token in text:
            descs.append(token)
    for desc in re.findall(r"(44|55반|55|66반|66|77반|77|88|99)\s*까지", text):
        text_desc = f"{desc}까지"
        descs.append(text_desc)
        ranks.extend(expand_size_text(text_desc))
    return dedupe_keep_order(descs), sorted(set(ranks))


def parse_colors(text: str, db_product: dict | None = None):
    colors = []
    if db_product and db_product.get("color_options"):
        colors.extend(re.split(r"[;,/|]", db_product.get("color_options", "")))
    for color in COLOR_WORDS:
        if color in text:
            colors.append(color)
    return dedupe_keep_order(colors)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_page_data(url: str):
    if not url:
        return None
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=12)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    raw_text = re.sub(r"\n{2,}", "\n", soup.get_text("\n", strip=True))
    candidates = extract_name_candidates(soup)
    page_name = candidates[0] if candidates else ""
    size_descs, size_ranks = parse_size_candidates(raw_text)
    measurements = extract_measurements(raw_text)
    return {
        "page_name": page_name,
        "name_candidates": candidates,
        "raw_text": raw_text[:12000],
        "summary": raw_text[:2500],
        "size_descs": size_descs,
        "size_ranks": size_ranks,
        "measurements": measurements,
    }


# ---------- domain logic ----------
def build_body_context():
    return {
        "height_cm": clean_text(st.session_state.body_height),
        "weight_kg": clean_text(st.session_state.body_weight),
        "top_size": clean_text(st.session_state.body_top),
        "bottom_size": clean_text(st.session_state.body_bottom),
    }


def build_body_context_text(body_ctx: dict) -> str:
    if not any(body_ctx.values()):
        return "입력된 체형 정보 없음"
    return f"키: {body_ctx.get('height_cm') or '-'}cm, 체중: {body_ctx.get('weight_kg') or '-'}kg, 상의: {body_ctx.get('top_size') or '-'}, 하의: {body_ctx.get('bottom_size') or '-'}"


def is_size_question(text: str):
    q = clean_text(text).replace(" ", "")
    return any(k in q for k in ["사이즈", "맞을까", "맞을까요", "맞아", "크기", "타이트", "여유", "작을", "클까", "77", "66반", "88"])


def is_color_question(text: str):
    q = clean_text(text)
    return any(k in q for k in ["색", "색상", "컬러", "무슨색", "어떤색", "블랙", "아이보리", "베이지"])


def is_policy_question(text: str):
    q = clean_text(text).replace(" ", "")
    return any(k in q for k in ["배송", "출고", "교환", "반품", "환불", "배송비", "무료배송"])


def is_name_question(text: str):
    q = clean_text(text).replace(" ", "")
    return any(k in q for k in ["이옷이름", "상품명", "제품명", "옷이름", "이름이뭐", "상품이름"])


def is_recommendation_question(text: str):
    q = clean_text(text)
    return any(k in q for k in ["추천", "코디", "같이 입", "어울리는", "매치", "무슨 바지", "어떤 바지", "무슨 치마", "어떤 상의"])


def build_current_product(page_data: dict | None, db_product: dict | None, product_no: str):
    page_name = clean_text((page_data or {}).get("page_name", ""))
    db_name = clean_text((db_product or {}).get("product_name", ""))

    if db_name and page_name:
        ratio = SequenceMatcher(None, norm_name(page_name), norm_name(db_name)).ratio()
        if ratio >= 0.88:
            final_name = db_name
            db_safe = True
        else:
            final_name = page_name
            db_safe = False
    elif page_name:
        final_name = page_name
        db_safe = False
    else:
        final_name = db_name or "지금 보시는 상품"
        db_safe = bool(db_name)

    page_ranks = (page_data or {}).get("size_ranks", [])
    db_ranks = expand_size_text((db_product or {}).get("size_range", ""))
    if page_ranks and db_ranks:
        if set(page_ranks) == set(db_ranks) or set(page_ranks).issubset(set(db_ranks)) or set(db_ranks).issubset(set(page_ranks)):
            confirmed_ranks = sorted(set(page_ranks) | set(db_ranks))
        else:
            confirmed_ranks = page_ranks  # page first when conflict
    else:
        confirmed_ranks = page_ranks or db_ranks

    colors = parse_colors((page_data or {}).get("raw_text", ""), db_product)

    category = clean_text((db_product or {}).get("category", "")) or infer_category_from_name(final_name)
    sub_category = clean_text((db_product or {}).get("sub_category", ""))
    coordination = dedupe_keep_order(re.split(r"[;,/|]", clean_text((db_product or {}).get("coordination_items", ""))))

    return {
        "product_no": normalize_product_no(product_no or (db_product or {}).get("product_no", "")),
        "name": final_name,
        "name_from_db_safe": db_safe,
        "category": category,
        "sub_category": sub_category,
        "confirmed_size_range": ranks_to_text(confirmed_ranks),
        "confirmed_size_ranks": confirmed_ranks,
        "option_size_desc": ", ".join((page_data or {}).get("size_descs", [])[:5]),
        "colors": colors,
        "measurements": (page_data or {}).get("measurements", {}),
        "fabric": clean_text((db_product or {}).get("fabric", "")),
        "style_tags": clean_text((db_product or {}).get("style_tags", "")),
        "body_cover_features": clean_text((db_product or {}).get("body_cover_features", "")),
        "coordination_items": coordination,
        "summary": clean_text((db_product or {}).get("product_summary", "")) or clean_text((page_data or {}).get("summary", ""))[:500],
    }


def find_best_candidates_for_reco(current_product: dict, user_text: str, body_ctx: dict, limit=3):
    if DB.empty:
        return []
    q = clean_text(user_text)
    current_no = normalize_product_no(current_product.get("product_no", ""))
    target_bottom = any(k in q for k in ["바지", "슬랙스", "팬츠", "데님", "청바지", "치마", "스커트"]) or current_product.get("category") in ["티셔츠", "블라우스", "셔츠", "니트", "가디건", "자켓"]
    target_top = any(k in q for k in ["상의", "블라우스", "셔츠", "니트", "가디건"]) or current_product.get("category") in ["팬츠", "스커트"]
    user_bottom_rank = size_rank(body_ctx.get("bottom_size", ""))
    user_top_rank = size_rank(body_ctx.get("top_size", ""))
    preferred = current_product.get("coordination_items", [])

    scored = []
    for _, row in DB.iterrows():
        rowd = row.to_dict()
        name = clean_text(rowd.get("product_name", ""))
        if not name:
            continue
        if current_no and normalize_product_no(rowd.get("product_no", "")) == current_no:
            continue
        row_cat = clean_text(rowd.get("category", ""))
        row_sub = clean_text(rowd.get("sub_category", ""))
        corpus = f"{name} {row_cat} {row_sub} {clean_text(rowd.get('style_tags',''))}"
        ranks = expand_size_text(rowd.get("size_range", ""))

        if target_bottom:
            if not any(k in corpus for k in ["팬츠", "슬랙스", "데님", "청바지", "스커트", "치마", "바지"]):
                continue
            if user_bottom_rank and ranks and user_bottom_rank not in ranks:
                continue
        elif target_top:
            if not any(k in corpus for k in ["블라우스", "셔츠", "니트", "가디건", "티셔츠", "맨투맨", "자켓"]):
                continue
            if user_top_rank and ranks and user_top_rank not in ranks:
                continue

        score = 0
        if preferred and any(p in corpus for p in preferred):
            score += 8
        if current_product.get("style_tags") and clean_text(rowd.get("style_tags", "")):
            overlap = set(re.split(r"[;,/|]", current_product.get("style_tags", ""))) & set(re.split(r"[;,/|]", clean_text(rowd.get("style_tags", ""))))
            score += len([x for x in overlap if clean_text(x)]) * 2
        if "슬랙스" in q and "슬랙스" in corpus:
            score += 5
        if any(k in q for k in ["데님", "청바지"]) and any(k in corpus for k in ["데님", "청바지"]):
            score += 5
        if any(k in q for k in ["학교", "방문", "출근", "오피스"]) and any(k in corpus for k in ["단정", "오피스", "클래식", "데일리", "슬랙스"]):
            score += 4
        if ranks:
            score += 1
        scored.append((score, rowd))

    scored.sort(key=lambda x: x[0], reverse=True)
    out, seen = [], set()
    for score, rowd in scored:
        name = clean_text(rowd.get("product_name", ""))
        if score <= 0 or name in seen:
            continue
        seen.add(name)
        out.append({
            "product_name": name,
            "category": clean_text(rowd.get("category", "")),
            "sub_category": clean_text(rowd.get("sub_category", "")),
            "size_range": clean_text(rowd.get("size_range", "")),
            "style_tags": clean_text(rowd.get("style_tags", "")),
            "body_cover_features": clean_text(rowd.get("body_cover_features", "")),
        })
        if len(out) >= limit:
            break
    return out


def product_name_answer(current_product: dict):
    name = current_product.get("name") or "지금 보시는 상품"
    return f"지금 보시는 상품명은 {name}이에요 :)"


def policy_answer(user_text: str):
    q = clean_text(user_text).replace(" ", "")
    if any(k in q for k in ["배송비", "무료배송"]):
        return f"배송비는 {POLICY_DB['shipping']['shipping_fee']:,}원이고, {POLICY_DB['shipping']['free_shipping_over']:,}원 이상이면 무료배송이에요 :)"
    if any(k in q for k in ["언제출고", "출고", "당일출고"]):
        return f"{POLICY_DB['shipping']['same_day_dispatch_rule']} 기준이고, 보통은 {POLICY_DB['shipping']['delivery_time']} 정도로 봐주시면 돼요 :)"
    if any(k in q for k in ["교환"]):
        return f"교환은 가능하고, {POLICY_DB['exchange_return']['period']} 안에 접수해주시면 돼요. 단순 변심 교환은 왕복 {POLICY_DB['exchange_return']['exchange_fee']:,}원이에요 :)"
    if any(k in q for k in ["반품", "환불"]):
        return f"반품도 가능하고, {POLICY_DB['exchange_return']['period']} 안에 접수해주시면 돼요. {POLICY_DB['exchange_return']['return_fee_rule']} 기준으로 진행돼요 :)"
    return None


def conservative_size_answer(current_product: dict, body_ctx: dict):
    user_top = clean_text(body_ctx.get("top_size", ""))
    if not user_top:
        return "사이즈는 같이 봐드릴 수 있어요 :) 상의 사이즈만 먼저 알려주시면 더 정확하게 말씀드릴게요."

    user_rank = size_rank(user_top)
    ranks = current_product.get("confirmed_size_ranks", [])
    range_text = current_product.get("confirmed_size_range", "")
    option_desc = current_product.get("option_size_desc", "")
    meas = current_product.get("measurements", {})

    if ranks and user_rank:
        max_rank = max(ranks)
        if user_rank > max_rank:
            return (
                f"고객님 상의 {user_top} 기준이면 이 상품은 편하게 맞는 쪽으로 보긴 어려워요. "
                f"현재 확인되는 권장 범위가 {range_text or ranks_to_text(ranks)} 쪽이라 77 이상까지 여유 있다고 제가 말씀드리긴 어려워요."
            )
        if user_rank in ranks:
            extra = []
            if option_desc:
                extra.append(f"페이지에서는 {option_desc} 쪽으로 먼저 보여요")
            if meas.get("가슴"):
                extra.append(f"가슴 기준은 {meas['가슴']} 정도로 확인돼요")
            extra_text = ". ".join(extra)
            if extra_text:
                extra_text = " " + extra_text + "."
            return (
                f"고객님 상의 {user_top} 기준이면 우선 권장 범위 안에는 들어와요 :)"
                f" 현재 확인되는 사이즈 범위는 {range_text or ranks_to_text(ranks)} 쪽이에요.{extra_text}"
                f" 다만 원하시는 핏이 딱 맞는 쪽인지 편하게 입는 쪽인지에 따라 체감은 조금 달라질 수 있어요."
            )

    if range_text:
        return f"현재 페이지와 DB를 같이 보면 이 상품은 {range_text} 쪽으로 먼저 보는 게 안전해요. 고객님 상의 {user_top} 기준으로는 실측도 같이 보는 쪽이 가장 정확해요 :)"
    return "현재 페이지에서 사이즈 정보는 일부 보이지만 제가 안전하게 단정할 만큼 충분하진 않아요. 상세 사이즈표와 실측 기준으로 같이 보는 게 가장 정확해요 :)"


def color_answer(current_product: dict):
    colors = current_product.get("colors", [])
    if not colors:
        return "현재 페이지에서 컬러가 또렷하게 확인되는 건 없어서, 없는 색을 제가 임의로 말씀드리진 않을게요. 옵션창에 보이는 색상 기준으로 봐주시면 가장 정확해요 :)"
    return f"현재 확인되는 컬러는 {', '.join(colors)} 쪽이에요 :)"


def recommendation_answer(current_product: dict, body_ctx: dict, user_text: str):
    candidates = find_best_candidates_for_reco(current_product, user_text, body_ctx, limit=3)
    if not candidates:
        return None
    lines = ["이 상품이랑 같이 보기 좋은 쪽으로 먼저 골라드릴게요 :) "]
    for c in candidates[:2]:
        reason = []
        if c.get("style_tags"):
            if any(k in c["style_tags"] for k in ["단정", "오피스", "클래식"]):
                reason.append("단정하게 받쳐주기 좋아요")
            elif "데일리" in c["style_tags"]:
                reason.append("데일리로 매치하기 편해요")
        if c.get("body_cover_features") and any(k in c["body_cover_features"] for k in ["복부", "허리", "힙"]):
            reason.append("체형 부담을 덜어주기 좋아요")
        if not reason:
            reason.append("같이 입었을 때 전체 코디가 깔끔해요")
        lines.append(f"- {c['product_name']}: {reason[0]}")
    return "\n".join(lines)


def safe_llm_fallback(user_text: str, current_product: dict, body_ctx: dict) -> str:
    if is_name_question(user_text):
        return product_name_answer(current_product)
    if is_policy_question(user_text):
        return policy_answer(user_text)
    if is_size_question(user_text):
        return conservative_size_answer(current_product, body_ctx)
    if is_color_question(user_text):
        return color_answer(current_product)
    rec = recommendation_answer(current_product, body_ctx, user_text)
    if rec:
        return rec
    return f"지금은 답변 연결이 조금 지연되고 있어요. 그래도 현재 기준으로는 {current_product.get('name') or '지금 보시는 상품'} 중심으로 같이 봐드릴게요 :) 궁금한 걸 한 가지씩 물어주시면 더 정확하게 이어서 도와드릴게요."


def get_llm_answer(user_text: str, current_product: dict, body_ctx: dict) -> str:
    reco_candidates = find_best_candidates_for_reco(current_product, user_text, body_ctx, limit=3) if is_recommendation_question(user_text) else []
    context_pack = {
        "current_product": {
            "name": current_product.get("name", "지금 보시는 상품"),
            "category": current_product.get("category", ""),
            "sub_category": current_product.get("sub_category", ""),
            "confirmed_size_range": current_product.get("confirmed_size_range", ""),
            "option_size_desc": current_product.get("option_size_desc", ""),
            "colors": current_product.get("colors", []),
            "measurements": current_product.get("measurements", {}),
            "fabric": current_product.get("fabric", ""),
            "style_tags": current_product.get("style_tags", ""),
            "body_cover_features": current_product.get("body_cover_features", ""),
            "summary": current_product.get("summary", "")[:400],
        },
        "body_context": body_ctx,
        "policy_db": POLICY_DB,
        "recommendation_candidates": reco_candidates,
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "참고 데이터(JSON):\n" + json.dumps(context_pack, ensure_ascii=False)},
    ]
    for m in st.session_state.messages[-4:]:
        messages.append({"role": m["role"], "content": clean_text(m["content"])[:300]})
    messages.append({"role": "user", "content": clean_text(user_text)[:300]})

    last_error = None
    for wait_seconds in (0.0, 1.0):
        if wait_seconds:
            time.sleep(wait_seconds)
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=220,
            )
            content = clean_text(resp.choices[0].message.content or "")
            if content:
                return content
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break
    print(f"[MIYA LLM ERROR] {type(last_error).__name__}: {last_error}")
    return safe_llm_fallback(user_text, current_product, body_ctx)


def process_user_message(user_text: str, current_product: dict):
    if st.session_state.is_processing:
        return

    qhash = norm_name(user_text)
    now = time.time()
    if qhash and qhash == st.session_state.last_question_hash and (now - st.session_state.last_question_at) < 4:
        if st.session_state.last_answer:
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.last_answer})
            return

    st.session_state.is_processing = True
    st.session_state.messages.append({"role": "user", "content": user_text})
    body_ctx = build_body_context()
    try:
        if is_name_question(user_text):
            answer = product_name_answer(current_product)
        elif is_policy_question(user_text):
            answer = policy_answer(user_text)
        elif is_size_question(user_text):
            answer = conservative_size_answer(current_product, body_ctx)
        elif is_color_question(user_text):
            answer = color_answer(current_product)
        else:
            rec = recommendation_answer(current_product, body_ctx, user_text) if is_recommendation_question(user_text) else None
            if rec:
                answer = rec
            else:
                answer = get_llm_answer(user_text, current_product, body_ctx)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_answer = answer
        st.session_state.last_question_hash = qhash
        st.session_state.last_question_at = now
    finally:
        st.session_state.is_processing = False


# ---------- load context ----------
qp = st.query_params
current_url = qp.get("url", "") or ""
product_no = qp.get("pn", "") or ""
product_name_q = qp.get("pname", "") or ""
if not product_no:
    product_no = extract_product_no(current_url)

page_data = fetch_page_data(current_url) if current_url else None
page_name = clean_text(product_name_q) or clean_text((page_data or {}).get("page_name", ""))
db_product = get_db_product(product_no, page_name)
current_product = build_current_product(page_data, db_product, product_no)

context_key = f"{current_url}|{current_product.get('product_no','')}|{current_product.get('name','')}"
if context_key != st.session_state.last_context_key:
    st.session_state.last_context_key = context_key
    st.session_state.messages = []


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
[data-testid="stAppViewContainer"] > .main {background: var(--miya-page-bg) !important;}
.block-container{background: var(--miya-page-bg) !important;}
div[data-testid="column"]{min-width:0 !important;}
div[data-testid="stTextInput"] label,div[data-testid="stSelectbox"] label{color:var(--miya-title) !important;font-weight:700 !important;font-size:11.5px !important;}
div[data-testid="stTextInput"] input,div[data-baseweb="select"] > div{border-radius:12px !important;}
div[data-testid="stTextInput"],div[data-testid="stSelectbox"]{margin-bottom:-2px !important;}
hr{margin-top:4px !important;margin-bottom:4px !important;border-color:var(--miya-divider) !important;}
div[data-testid="stChatInput"]{position:fixed !important;left:50% !important;transform:translateX(-50%) !important;bottom:68px !important;width:min(720px, calc(100% - 24px)) !important;z-index:9999 !important;background: transparent !important;}
div[data-testid="stChatInput"] > div{background: transparent !important;border-radius: 0 !important;padding: 0 !important;box-shadow: none !important;border: none !important;}
div[data-testid="stChatInput"] textarea {background: #1f2740 !important;color: #ffffff !important;caret-color: #ffffff !important;-webkit-text-fill-color: #ffffff !important;font-size: 16px !important;line-height: 1.35 !important;padding-top: 12px !important;padding-bottom: 12px !important;}
div[data-testid="stChatInput"] textarea::placeholder {color: #cfd6e6 !important;opacity: 1 !important;-webkit-text-fill-color: #cfd6e6 !important;}
div[data-testid="stChatInput"] [data-baseweb="textarea"] {background: #1f2740 !important;border-radius: 999px !important;border: 1px solid rgba(255,255,255,0.08) !important;min-height: 52px !important;padding: 0 10px !important;display: flex !important;align-items: center !important;}
div[data-testid="stChatInput"] [data-baseweb="textarea"] > div {background: transparent !important;display: flex !important;align-items: center !important;}
div[data-testid="stChatInput"] button {background: #2f3a5f !important;color: #ffffff !important;border-radius: 14px !important;}div[data-testid="stChatInput"] button svg {fill: #ffffff !important;}
@media (max-width: 768px){.block-container{max-width:100%;padding-top:0.14rem !important;padding-bottom:11.6rem !important;}div[data-testid="stHorizontalBlock"]{gap:6px !important;}div[data-testid="stHorizontalBlock"] > div{flex:1 1 0 !important;min-width:0 !important;}div[data-testid="stTextInput"] label,div[data-testid="stSelectbox"] label{font-size:11px !important;}div[data-testid="stTextInput"],div[data-testid="stSelectbox"]{margin-bottom:-4px !important;}hr{margin-top:3px !important;margin-bottom:3px !important;}div[data-testid="stChatInput"]{bottom:64px !important;width:calc(100% - 16px) !important;}div[data-testid="stChatInput"] > div{padding: 0 !important;border-radius: 0 !important;background: transparent !important;}div[data-testid="stChatInput"] [data-baseweb="textarea"]{min-height: 48px !important;padding: 0 10px !important;display: flex !important;align-items: center !important;}}
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
      <div style="padding:6px 8px 0 8px; border:1px solid rgba(0,0,0,.04); border-radius:14px; background:transparent;">
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
    is_detail_page = (("/product/detail" in current_url_lower) or ("product_no=" in current_url_lower) or bool(current_product.get("product_no")))
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
            "상품 상세페이지에서 채팅창을 열면 그 상품 기준으로 더 정확하게 상담해드릴 수 있어요 :)"
        )
    st.session_state.messages.append({"role": "assistant", "content": welcome})

st.divider()
for msg in st.session_state.messages:
    safe_text = html.escape(msg["content"]).replace("\n", "<br>")
    if msg["role"] == "user":
        st.markdown('<div style="display:flex; justify-content:flex-end; width:100%; margin:2px 0 4px 0;"><div style="max-width:92%;"><div style="display:block; font-size:12px; font-weight:700; line-height:1.15; color:#0f6a63; text-align:right; margin:0 6px 1px 0;">고객님</div><div style="padding:10px 14px 10px 10px; border-radius:18px; border-bottom-right-radius:6px; font-size:15px; line-height:1.5; white-space:pre-wrap; word-break:keep-all; background:#dff0ec; color:#1f3b36; border:1px solid rgba(15,106,99,.14);">'+safe_text+'</div></div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="display:flex; justify-content:flex-start; width:100%; margin:2px 0 4px 0;"><div style="max-width:92%;"><div style="display:block; font-size:12px; font-weight:700; line-height:1.15; color:#5f6471; margin:0 0 1px 6px;">미야언니</div><div style="padding:10px 14px 10px 10px; border-radius:18px; border-bottom-left-radius:6px; font-size:15px; line-height:1.5; white-space:pre-wrap; word-break:keep-all; background:#071b4e; color:#ffffff; border:1px solid rgba(255,255,255,.08);">'+safe_text+'</div></div></div>', unsafe_allow_html=True)

user_input = st.chat_input("메시지를 입력하세요…")
if user_input:
    process_user_message(user_input, current_product)
    st.rerun()
