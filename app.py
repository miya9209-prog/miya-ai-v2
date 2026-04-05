import os
import re
import json
import html
import time
import logging
from datetime import datetime
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

# ── 로깅 (구조화 JSONL - 관리프로그램 연동용) ──────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def _get_file_logger():
    logger = logging.getLogger("miya_chat_file")
    if not logger.handlers:
        log_path = os.path.join(LOG_DIR, "chat_{}.jsonl".format(datetime.now().strftime("%Y%m%d")))
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger

def write_log(
    event_type: str,
    product_no: str = "",
    product_name: str = "",
    user_text: str = "",
    bot_text: str = "",
    response_mode: str = "",
    fallback_reason: str = "",
    is_fallback: bool = False,
    error_text: str = "",
    latency_ms: float = 0,
    session_id: str = "",
):
    """관리프로그램이 바로 파싱 가능한 JSONL 구조화 로그 저장"""
    try:
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "session_id": session_id,
            "product_no": product_no,
            "product_name": product_name,
            "user_text": user_text[:300],
            "bot_text": bot_text[:400],
            "response_mode": response_mode,
            "fallback_reason": fallback_reason,
            "is_fallback": is_fallback,
            "error_text": error_text[:300],
            "latency_ms": round(latency_ms, 1),
        }
        _get_file_logger().info(json.dumps(record, ensure_ascii=False))
    except Exception:
        pass

# ── 정책 DB ───────────────────────────────────────────────────────────────────
POLICY_DB = {
    "shipping": {
        "courier": "CJ 대한통운",
        "shipping_fee": 3000,
        "free_shipping_over": 70000,
        "delivery_time": "결제 완료 후 2~4영업일 정도",
        "same_day_dispatch_rule": "오후 2시 이전 주문은 당일 출고",
        "combined_shipping": "합배송 가능(1박스 기준)",
        "jeju": "제주 및 도서산간 지역은 추가배송비가 자동 부과됩니다.",
    },
    "exchange_return": {
        "period": "상품 수령 후 7일 이내",
        "exchange_fee": 6000,
        "return_fee_rule": "단순 변심: 7만원 이상이면 편도 3,000원 / 7만원 미만이면 왕복 6,000원",
        "defect_wrong": "불량/오배송은 미샵 부담",
    },
}

# ── 상수 ──────────────────────────────────────────────────────────────────────
SIZE_ORDER = {"44": 1, "55": 2, "55반": 3, "66": 4, "66반": 5, "77": 6, "77반": 7, "88": 8, "99": 9}
SIZE_LABELS = {v: k for k, v in SIZE_ORDER.items()}
APPROX_BODY_BUST = {1: 82, 2: 85, 3: 88, 4: 91, 5: 95, 6: 100, 7: 104, 8: 109, 9: 114}
COLOR_CANDIDATES = ["블랙", "화이트", "아이보리", "그레이", "베이지", "브라운", "네이비", "핑크", "소라", "블루", "카키", "민트", "레드", "옐로우", "연그린"]

# DB의 실제 category / sub_category 값 기준
BOTTOM_SUB_CATS = {"데님", "슬랙스", "팬츠", "스커트"}
TOP_MAIN_CATS = {"블라우스/셔츠", "니트/가디건", "아우터", "티셔츠"}
TOP_SUB_CATS = {"셔츠", "블라우스", "니트", "가디건", "자켓", "점퍼", "티셔츠"}
SITUATION_KEYWORDS = ["학교", "방문", "출근", "모임", "데이트", "친구", "여행", "산책", "행사", "파티"]

SYSTEM_PROMPT = (
    "너는 미샵의 24시간 쇼핑 친구 미야언니야.\n"
    "4050 여성 고객이 옆에 믿을 수 있는 친구 MD가 있는 것처럼 상담해.\n\n"
    "말투 규칙:\n"
    "- 기본 존댓말이지만 친근하고 자연스럽게 — 딱딱하지 않게\n"
    "- DB에 의하면, 상품정보에 의하면 같은 기계적 표현 절대 금지\n"
    "- 결론부터 말하고 3~5문장 내외로 간결하게\n"
    "- 공감 한 문장 + 핵심 정보 + 실질적 조언 구조로\n\n"
    "상담 규칙:\n"
    "1. current_product_name 이름만 사용 (모르면 지금 보시는 상품)\n"
    "2. 추천 상품명은 allowed_candidates에 있는 이름만 사용 — 없는 상품명 절대 금지\n"
    "3. size_ok가 false면 맞다고 절대 하지 말 것 — 잘못된 사이즈 안내는 교환/반품 비용과 신뢰 손실\n"
    "4. confirmed_colors 안에 있는 컬러만 언급\n"
    "5. 데이터 없으면 추측하지 말고 솔직하게 말할 것\n"
    "6. user_has_size_input이 false이면 사이즈 판단 절대 금지 — 사이즈 먼저 물어볼 것\n"
    "7. 인사(안녕 등)에는 사이즈/상품 정보 먼저 꺼내지 말 것\n\n"
    "4050 여성 상황별 상담 지침:\n"
    "- 상견례/면접: 격식 있고 신뢰감 주는 스타일 강조\n"
    "- 결혼식 하객: 단정하되 화사하게, 너무 튀지 않게\n"
    "- 시댁 방문/명절: 단정하고 정갈한 인상\n"
    "- 학교/학부모 방문: 단정하고 신뢰감 있는 스타일\n"
    "- 동창회/친구 모임: 세련되고 나이에 맞게 멋있게\n"
    "- 출근/미팅: 깔끔하고 프로페셔널하게\n"
    "- 여행/나들이: 편하고 실용적으로\n"
    "- 데이트/기념일: 여성스럽고 세련되게\n"
    "- 체형 고민 언급 시: 장점 살리고 단점 커버하는 방향으로 공감하며 안내"
)


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_product_no(value) -> str:
    text = clean_text(value)
    return text[:-2] if text.endswith(".0") else text


def size_rank(token: str) -> Optional[int]:
    return SIZE_ORDER.get(clean_text(token))


def rank_to_size(rank: Optional[int]) -> str:
    return SIZE_LABELS.get(rank, "") if rank else ""


def expand_size_text(size_text: str) -> List[int]:
    text = clean_text(size_text).replace("~", "-")
    if not text:
        return []
    found: List[int] = []
    for a, b in re.findall(
        r"(44|55반|55|66반|66|77반|77|88|99)\s*[-~]\s*(44|55반|55|66반|66|77반|77|88|99)", text
    ):
        ra, rb = size_rank(a), size_rank(b)
        if ra and rb:
            found.extend(range(min(ra, rb), max(ra, rb) + 1))
    m = re.search(r"(44|55반|55|66반|66|77반|77|88|99)\s*까지", text)
    if m:
        rb = size_rank(m.group(1))
        if rb:
            found.extend(range(1, rb + 1))
    if not found:
        for token in ["55반", "66반", "77반", "44", "55", "66", "77", "88", "99"]:
            if token in text:
                rank = size_rank(token)
                if rank:
                    found.append(rank)
    if re.search(r"\bfree\b|FREE", text, re.IGNORECASE):
        found.extend([2, 3, 4, 5, 6])
    return sorted(set(found))


def _new_session_id() -> str:
    """페이지 로드 기준 세션 ID 생성"""
    import hashlib
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return "sess_" + hashlib.md5(ts.encode()).hexdigest()[:8]

def ensure_state() -> None:
    defaults = {
        "messages": [], "last_context_key": "",
        "body_height": "", "body_weight": "", "body_top": "", "body_bottom": "",
        "is_processing": False, "last_user_hash": "", "last_user_ts": 0.0,
        "last_answer": "", "last_recommendations": [],
        "session_id": _new_session_id(),  # 관리프로그램 연동용 세션 ID
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_state()


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
    return rows.iloc[0].to_dict() if len(rows) > 0 else None


def extract_product_no_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        qs = parse_qs(urlparse(url).query)
        no = qs.get("product_no", [""])[0] or qs.get("pn", [""])[0]
        return normalize_product_no(no)
    except Exception:
        return ""


def sanitize_product_name(name: str) -> str:
    text = clean_text(name)
    if not text:
        return ""
    for piece in ["LOGIN", "JOIN", "MY PAGE", "MYPAGE", "CART", "ABOUT", "SHOP", "COMMUNITY",
                  "TIME SALE", "KRW", "미샵", "MISHARP", "{#item", "{#html", "기본 정보", "상품명"]:
        text = text.replace(piece, " ")
    text = re.sub(r"\[[^\]]*\]|★+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|/>")
    return text if len(text) >= 3 else ""


def extract_meta_name(soup) -> str:
    for selector in ['meta[property="og:title"]', 'meta[name="og:title"]',
                     'meta[property="twitter:title"]', 'meta[name="title"]']:
        tag = soup.select_one(selector)
        if tag and tag.get("content"):
            s = sanitize_product_name(tag.get("content"))
            if s:
                return s
    if soup.title and soup.title.text:
        s = sanitize_product_name(soup.title.text)
        if s:
            return s
    return ""


def extract_colors_from_text(text: str) -> List[str]:
    return [c for c in COLOR_CANDIDATES if c in text]


def split_detail_sections(text: str) -> Dict[str, str]:
    t = clean_text(text)
    if not t:
        return {"summary": "", "material": "", "fit": "", "size_tip": ""}
    material, fit, size_tip = [], [], []
    for s in re.split(r"(?<=[.!?])\s+|\s*/\s*", t):
        s = clean_text(s)
        if not s:
            continue
        if any(k in s for k in ["면", "코튼", "폴리", "레이온", "울", "아크릴", "스판", "나일론", "혼용", "%", "소재", "원단"]):
            material.append(s)
        if any(k in s for k in ["핏", "루즈", "정핏", "와이드", "세미", "커버", "복부", "허벅지", "힙", "라인", "여유"]):
            fit.append(s)
        if any(k in s for k in ["사이즈", "추천", "44", "55", "66", "77", "88", "FREE", "free", "L(", "M(", "S("]):
            size_tip.append(s)
    return {
        "summary": t[:1400], "material": " / ".join(material)[:350],
        "fit": " / ".join(fit)[:350], "size_tip": " / ".join(size_tip)[:350],
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_product_context(url: str, passed_name: str = "", passed_product_no: str = "") -> Dict:
    safe_name = sanitize_product_name(passed_name)
    safe_no = normalize_product_no(passed_product_no) or extract_product_no_from_url(url)
    fallback = {
        "product_no": safe_no, "product_name": safe_name or "지금 보시는 상품",
        "category": "기타", "sub_category": "",
        "summary": "", "material": "", "fit": "", "size_tip": "", "raw_excerpt": "", "colors": [],
    }
    if not url:
        return fallback
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        r.raise_for_status()
    except Exception:
        return fallback
    soup = BeautifulSoup(r.text, "html.parser")
    meta_name = extract_meta_name(soup)
    product_name = safe_name or meta_name
    db_row = get_db_product(safe_no)
    if db_row and clean_text(db_row.get("product_name")):
        product_name = clean_text(db_row.get("product_name"))
    for t in soup(["script", "style", "noscript", "header", "footer"]):
        t.decompose()
    raw_text = clean_text(re.sub(r"\n{2,}", "\n", soup.get_text("\n").replace("\r", "\n")))
    sections = split_detail_sections(raw_text)
    return {
        "product_no": safe_no,
        "product_name": product_name or "지금 보시는 상품",
        "category": clean_text((db_row or {}).get("category", "")) or "기타",
        "sub_category": clean_text((db_row or {}).get("sub_category", "")) or "",
        "summary": sections["summary"], "material": sections["material"],
        "fit": sections["fit"], "size_tip": sections["size_tip"],
        "raw_excerpt": raw_text[:4000], "colors": extract_colors_from_text(raw_text),
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
        "키: {}cm, 체중: {}kg, 상의: {}, 하의: {}".format(
            body_ctx.get("height_cm") or "-", body_ctx.get("weight_kg") or "-",
            body_ctx.get("top_size") or "-", body_ctx.get("bottom_size") or "-"
        )
    )


def extract_user_body_from_text(user_text: str) -> Dict[str, str]:
    """4050 여성 고객의 다양한 체형 언급 감지"""
    result = {}
    q = clean_text(user_text)

    # ── 키
    if any(k in q for k in ["키가 작", "키작", "소키", "키 작은", "키작은", "단신", "작은키", "키가 많이 작", "키가안 커", "키가크지않"]):
        result["height_hint"] = "단신"
    elif any(k in q for k in ["키가 크", "키크", "키 큰", "키큰", "장신", "큰키", "키가 좀 있"]):
        result["height_hint"] = "장신"

    # ── 상체/가슴/어깨 (실제 4050 표현)
    upper_big = [
        "상체가 크", "상체크", "상체 있는", "상체있는", "상체가 있",
        "상체가 발달", "상체발달", "상체비만",
        "어깨넓", "어깨가 넓", "어깨가넓", "어깨가 커", "어깨가커",
        "가슴크", "가슴이 좀", "가슴이좀", "가슴이 있", "가슴이있", "가슴이 크", "가슴이 커", "가슴이커",
        "가슴 있는", "가슴있는",
        "윗부분이 문제", "윗부분이 작아", "상체가 문제", "윗배", "상반신",
        "브라 사이즈가 크", "컵이 커", "가슴이 풍만",
        "가슴이 커서", "가슴이커서", "가슴때문에", "가슴이항상",
    ]
    upper_small = [
        "상체가 작", "상체작", "어깨좁", "어깨가 좁", "어깨가좁", "상체가 좁",
        "상체가 빈약", "어깨가 좁은",
    ]
    if any(k in q for k in upper_big):
        result["upper_body_hint"] = "상체큰편"
    elif any(k in q for k in upper_small):
        result["upper_body_hint"] = "상체작은편"

    # ── 팔뚝 (4050 여성 자주 언급)
    if any(k in q for k in ["팔뚝", "팔이 굵", "팔이굵", "팔뚝이 굵", "팔이 두꺼", "소매 부분이 걱정", "소매가 걱정"]):
        result["arm_hint"] = "팔뚝굵음"
        if "upper_body_hint" not in result:
            result["upper_body_hint"] = "상체큰편"

    # ── 하체/허벅지/골반
    lower_big = [
        "하체가 크", "하체크", "허벅지", "하체통통", "하체가 통통",
        "허벅지가", "엉덩이가 크", "엉덩이가크", "골반이 넓", "골반넓",
        "하체가 있", "하체있는", "하체비만", "다리가 굵", "다리굵",
        "하체가 문제", "하체가 좀",
    ]
    if any(k in q for k in lower_big):
        result["lower_body_hint"] = "하체통통"

    # ── 복부/배 (4050 여성 가장 많이 언급)
    belly_kws = [
        "배가 나", "복부", "뱃살", "배가나", "배살", "배가 있", "배가있",
        "배나온", "배가 좀", "배가 많이", "배가 볼록", "배불뚝",
        "티 안 날", "티가 날", "티날까", "복부가", "뱃살이",
        "허리가 굵", "허리굵", "허리가 좀 있",
    ]
    if any(k in q for k in belly_kws):
        result["belly_hint"] = "복부커버필요"

    # ── 전체 통통/체형 (직접 언급)
    if any(k in q for k in ["통통한 편", "통통해요", "통통해", "통통합니다", "포동포동", "살이 좀 있", "살집이"]):
        if "upper_body_hint" not in result:
            result["upper_body_hint"] = "상체큰편"
        if "lower_body_hint" not in result:
            result["lower_body_hint"] = "하체통통"

    # ── 키 작고 통통 (4050 자주: "작고 통통")
    if any(k in q for k in ["작고 통통", "작은데 통통", "키작고통통"]):
        result["height_hint"] = "단신"
        result["upper_body_hint"] = "상체큰편"

    return result
def detect_size_from_text(user_text: str) -> Optional[str]:
    for token in ["55반", "66반", "77반", "44", "55", "66", "77", "88", "99"]:
        if token in user_text:
            return token
    return None


def detect_situation_from_text(user_text: str) -> List[str]:
    """4050 여성 고객의 다양한 상황 키워드 감지"""
    found = []
    situation_map = {
        "학교": [
            "학교", "선생님", "학부모", "입학식", "졸업식", "학예회",
            "운동회", "상담", "수업참관", "학교행사", "학교방문",
        ],
        "출근": [
            "출근", "직장", "회사", "사무실", "오피스", "업무",
            "미팅", "비즈니스", "발표", "프레젠테이션",
        ],
        "면접": ["면접", "취업", "채용"],
        "모임": [
            "모임", "동창회", "동문회", "동기", "친목", "송년회",
            "신년회", "환영회", "송별회", "하객룩", "행사",
        ],
        "결혼식": [
            "결혼식", "하객", "예식", "웨딩",
        ],
        "돌잔치": ["돌잔치", "돌", "백일"],
        "상견례": ["상견례", "맞선"],
        "시댁": [
            "시댁", "시어머니", "시부모", "시부모님", "시집",
            "명절", "설날", "추석", "제사", "차례",
        ],
        "친정": ["친정", "친정어머니", "친정부모"],
        "데이트": [
            "남편", "남자친구", "커플", "데이트", "기념일",
            "결혼기념일", "생일", "외식", "저녁식사",
        ],
        "소개팅": ["소개팅", "맞선"],
        "친구": [
            "친구", "친구들", "지인", "이웃", "동네",
        ],
        "여행": [
            "여행", "나들이", "소풍", "피크닉", "캠핑",
        ],
        "교회": ["교회", "성당", "절", "예배", "미사"],
    }
    for situation, keywords in situation_map.items():
        if situation not in found:
            if any(k in user_text for k in keywords):
                found.append(situation)
    return found


def detect_style_from_text(user_text: str) -> str:
    """스타일 요청 감지 — 격식/캐주얼/세련 등"""
    formal_kws = ["격식", "포멀", "세미정장", "정장", "단정", "클래식", "우아", "품위"]
    casual_kws = ["캐주얼", "편하게", "편한", "일상", "데일리", "편안"]
    elegant_kws = ["세련", "세련되게", "멋있게", "스타일리시", "시크", "트렌디"]
    young_kws = ["젊어보이게", "어려보이게", "젊게", "나이보다젊", "동안"]

    q = user_text.replace(" ","")
    if any(k in q for k in formal_kws): return "격식"
    if any(k in q for k in elegant_kws): return "세련"
    if any(k in q for k in young_kws): return "동안"
    if any(k in q for k in casual_kws): return "캐주얼"
    return ""



def is_size_question(user_text: str) -> bool:
    """4050 여성 고객의 다양한 사이즈/핏 질문 감지"""
    if is_recommendation_question(user_text):
        return False
    # 추천 후속 질문 ("추천해준 게 맞아?") → followup에서 처리
    q_tmp = user_text.replace(" ","")
    if any(k in q_tmp for k in ["추천해준","방금추천","추천한거","추천한게"]):
        return False
    q = user_text.replace(" ", "")

    # ── 직접 사이즈 키워드
    size_kws = [
        "사이즈", "맞을까", "맞을까요", "맞아", "맞아요", "맞나요", "맞나",
        "맞겠어요", "맞겠죠", "맞겠나요",
        "핏", "작을까", "작을까요", "클까", "클까요",
        "여유", "여유있게", "여유있어", "여유있나요", "여유로워",
        "타이트", "빡빡", "빡빡할", "빡빡해요",
        "내사이즈", "나한테맞", "나에게맞", "제사이즈",
        "입을수있", "살수있", "나올까", "나오나요", "나오는지",
        "안맞겠지", "안맞을까", "안맞나요", "안맞아",
        "될까", "될까요", "돼요",
        "가능해요", "가능한가요", "가능할까", "가능한지",
        "살수있어", "살수있나요",
        "입어도될", "입어도돼", "입어도되나",
        "사이즈있어", "사이즈있나", "사이즈나와",
        "티안날까", "티가날까", "티날까",
        "어떤사이즈", "몇사이즈",
        # 4050 특유 표현
        "괜찮을까요", "괜찮을까", "괜찮겠어요", "괜찮나요",
        "걱정되는데", "걱정이에요", "걱정돼요",
        "티안날까", "티가날까", "티날까요",
        "빅사이즈", "큰사이즈", "라지", "엑스라지",
        "통통한편", "통통해서", "살이있어서", "살집이",
        "기장이걱정", "기장이짧을", "기장이길",
        "소매부분이", "소매가걱정",
    ]
    if any(k in q for k in size_kws): return True

    # ── 사이즈 숫자
    if detect_size_from_text(user_text): return True

    # ── 자연어 패턴
    patterns = [
        r"이\s*[옷거상품].{0,20}(나한테|나에게|내가|제가|맞|어때|어울|될까|가능|살|입|괜찮)",
        r"(나한테|나에게|내가|제가|저한테).{0,20}(맞|될|가능|입을|돼|살|괜찮)",
        r"(내|제)\s*(사이즈|몸|체형|키|몸무게|몸집).{0,10}(맞|될|가능|있|나|괜찮)",
        r"(이거|이옷|이상품|그거|그옷).{0,15}(될까|맞|가능|살|입|어때|괜찮)",
        r"(큰\s*사이즈|빅\s*사이즈|라지|엑스라지).{0,10}(있|나와|있나|있어)",
        r"(팔뚝|소매|허리|배|복부|가슴|어깨|허벅지|엉덩이).{0,15}(걱정|타이트|빡빡|티날|여유|맞|될|가능|괜찮)",
        r"(통통|살이|포동|살집).{0,10}(입을|맞|될|가능|괜찮)",
        r"(체형|체격|몸매).{0,10}(맞|될|가능|괜찮|어울|입을)",
    ]
    for pat in patterns:
        if re.search(pat, user_text): return True

    # ── 체형 힌트 + 핏/가능성 의도
    body_hints = extract_user_body_from_text(user_text)
    if body_hints and any(k in q for k in ["맞", "어때", "어울", "될까", "입을", "안맞", "역시", "가능", "살", "괜찮", "걱정"]):
        return True

    return False
def is_name_question(user_text: str) -> bool:
    q = user_text.replace(" ", "")
    return any(k in q for k in ["이옷이름", "상품명", "상품이름", "이름뭐", "이옷이뭐야", "품명"])


def is_color_question(user_text: str) -> bool:
    """현재 상품의 컬러 옵션 질문"""
    return any(k in user_text for k in ["컬러", "색상", "무슨 색", "어떤 색"] + COLOR_CANDIDATES)


def is_color_match_question(user_text: str) -> bool:
    """컬러 조합/매치 질문 — 어떤 색이 어울리는지"""
    q = user_text.replace(" ", "")
    kws = [
        "어울리는색", "어울리는컬러", "어울리는색깔",
        "색깔매치", "컬러매치", "색매치", "색조합", "컬러조합",
        "어떤컬러", "어떤색이", "어떤색으로", "무슨색",
        "색상어울", "색어울", "컬러어울",
        "같이입으면좋은색", "코디색",
    ]
    if any(k in q for k in kws): return True
    # 컬러 언급 + 어울림 표현
    has_color = any(c in user_text for c in COLOR_CANDIDATES + ["흰색", "검정", "검은색", "흰", "회색", "하늘색", "연두색", "갈색"])
    has_match = any(k in q for k in ["어울", "매치", "조합", "같이", "함께", "코디"])
    if has_color and has_match: return True
    return False


def is_body_style_question(user_text: str) -> bool:
    """체형 보완 코디 질문 — 날씬해보이게, 키커보이게 등"""
    q = user_text.replace(" ", "")
    kws = [
        "날씬해보이게", "날씬해보이는", "날씬하게보이", "날씬보이",
        "키커보이게", "키커보이는", "키가커보이", "키작아보이지않",
        "키가작아보이지않", "다리길어보이게", "다리길어보이는",
        "균형있어보이게", "균형있게", "균형잡혀",
        "뚱뚱해보이지않", "뚱뚱해보이지않게", "살빠져보이게", "살빠보이",
        "슬림해보이게", "슬림하게", "슬림해보이는",
        "작아보이지않", "통통해보이지않", "통통해보이지않게",
        "보정코디", "체형보완", "체형커버",
        "어때입어야", "어떻게입어야", "코디법", "입는법", "입는방법",
    ]
    if any(k in q for k in kws): return True
    patterns = [
        r"(날씬|슬림|키|다리|균형).{0,10}(보이게|보이는|보이도록|보이려면|보일)",
        r"(작아|뚱뚱|통통|짧아).{0,10}보이지\s*않",
        r"체형.{0,15}(보완|커버|살려|강조|숨기|맞게|맞는)",
        r"(작고\s*통통|키작고|키작은).{0,15}(입는법|코디|추천|어떻게|뭐)",
    ]
    for pat in patterns:
        if re.search(pat, user_text): return True
    return False


def is_recommendation_question(user_text: str) -> bool:
    """4050 여성 고객의 다양한 추천/코디 요청 감지"""
    # 컬러매치/체형코디 질문은 추천이 아닌 별도 로직으로 처리
    if is_color_match_question(user_text): return False
    if is_body_style_question(user_text): return False
    q = user_text.replace(" ", "")

    # ── 명시적 추천/코디 키워드
    reco_kws = [
        "추천해줘", "추천해주세요", "추천좀", "추천부탁", "추천해봐", "추천좀해줘",
        "골라줘", "골라줘요", "골라주세요", "골라봐줘", "골라봐",
        "어울리는", "어울릴", "어울려", "어울리게",
        "같이입", "같이입을", "함께입", "세트로",
        "코디해줘", "코디도와줘", "코디부탁", "코디추천", "코디도움",
        "매치", "매치해줘", "매치되는",
        "비슷한옷", "비슷한상품", "비슷한거", "비슷한느낌",
        "다른옷", "다른상품", "다른거", "다른걸로",
        "다른자켓", "다른아우터", "다른바지", "다른점퍼", "다른맨투맨",
        "다른셔츠", "다른블라우스", "다른가디건", "다른니트", "다른스커트",
        "없어요", "없나요", "없을까",
        "보여줘", "보여주세요", "뭐입", "뭘입", "무엇을입",
        "위아래", "상하의", "전체코디",
        "어떤거", "어떤게좋", "뭐가좋", "뭘살",
        "비슷한스타일", "이런스타일로", "이런비슷한",
        "입을수있는상품", "같은느낌으로", "이런류",
        "비슷한티", "비슷한옷있", "비슷한거있", "비슷한상품있",
        "이것보다", "이거보다", "더큰비슷한", "큰비슷한",
    ]
    if any(k in q for k in reco_kws): return True

    # ── 상황별 패턴 (4050 여성 실생활 — 전수)
    situation_patterns = [
        # 학교/교육 관련
        r"(학교|입학식|졸업식|학부모|학예회|운동회|상담|수업참관).{0,25}(입|코디|추천|뭐|어떤|골라|뭘|어울|좋)",
        r"선생님\s*(만나|뵙|면담|상담).{0,20}(입|코디|추천|뭐|어떤|뭘)",

        # 직장/비즈니스
        r"(출근|회사|직장|사무실|오피스|미팅|프레젠테이션|발표|면접|취업).{0,25}(입|코디|추천|뭐|어떤|뭘|어울|좋)",

        # 가족/명절/시댁
        r"(시댁|친정|명절|설날|추석|제사|차례|시어머니|시부모|시부모님).{0,25}(입|코디|추천|뭐|어떤|뭘|어울|좋)",
        r"(부모님|어머니|아버지)\s*(만나|뵙|방문).{0,20}(입|코디|추천|뭐|어떤|뭘)",

        # 경조사
        r"(결혼식|하객|돌잔치|돌|백일|장례식|추도식|제사|문상).{0,25}(입|코디|추천|뭐|어떤|뭘|어울|좋)",
        r"(상견례|맞선).{0,25}(입|코디|추천|뭐|어떤|뭘|어울|좋|격식)",

        # 사교 모임
        r"(동창회|동문회|동기|선후배|친목|송년회|신년회|환영회|송별회).{0,25}(입|코디|추천|뭐|어떤|뭘|어울|좋)",
        r"(모임|파티|행사|이벤트|축하|기념일).{0,25}(입|코디|추천|뭐|어떤|뭘|어울|좋)",

        # 데이트/소개팅
        r"(남편|와이프|남자친구|여자친구|남편이랑|아내랑|커플|데이트|기념일|소개팅|맞선).{0,25}(입|코디|추천|뭐|어떤|뭘|어울|좋)",
        r"(결혼기념일|생일|데이트|저녁식사|외식).{0,20}(세련|예쁘|맞|입|코디|뭐)",

        # 여행/나들이
        r"(여행|나들이|소풍|피크닉|캠핑|등산|공원|드라이브).{0,25}(입|코디|추천|뭐|어떤|뭘|어울|좋)",

        # 종교
        r"(교회|성당|절|예배|미사).{0,25}(입|코디|추천|뭐|어떤|뭘|어울|좋)",

        # 스타일 요청
        r"(세련되게|우아하게|단정하게|캐주얼하게|편하게|멋있게|예쁘게|젊어보이게|어려보이게).{0,20}(입|코디|추천|뭐|어떤|뭘)",
        r"(세미정장|포멀|캐주얼|격식|클래식|우아).{0,15}(느낌|스타일|룩|코디|추천|입|뭐)",
        r"(나이에\s*맞게|나이답게|나이보다\s*젊게|나잇값).{0,20}(입|코디|추천|뭐|어떤)",

        # 기타 추천 패턴
        r"(입고\s*갈|입을\s*만한|걸칠\s*만한|걸칠|입어야).{0,20}(거|것|옷|상품)",
        r"(뭐\s*입|어떻게\s*입|어떤\s*옷|어떤\s*걸\s*사).{0,15}(좋|나을|어울|될|살|싶)",
        r"(더\s*큰|빅|라지|큰\s*사이즈).{0,10}(없|있|나와|찾)",
        r"(나한테|나에게|내가|제가).{0,10}(맞는|어울리는|맞을).{0,15}(있|없|찾|줘|추천)",
        r"(맞는|어울리는|어울릴).{0,10}(자켓|아우터|옷|바지|상품|점퍼|맨투맨|셔츠).{0,10}(있|없|찾|줘)",
        r"(내|제)\s*(사이즈|몸).{0,10}(맞는|되는|가능한).{0,10}(있|없|찾)",
        r"(88|99|77반).{0,10}(되는|나오는|있는).{0,10}(자켓|아우터|옷|상품|점퍼|바지)",
        r"비슷한\s*(티셔츠|맨투맨|셔츠|바지|자켓|원피스|가디건|니트|옷|상품|거).{0,10}(있|없|나와|찾)",
        r"(이것|이거|이옷)보다.{0,15}(크|넉넉|여유|큰).{0,10}(비슷|같은|이런)",
        r"(더\s*큰|더\s*넉넉한|더\s*여유).{0,15}(비슷한|같은|이런)",
    ]
    for pat in situation_patterns:
        if re.search(pat, user_text): return True

    return False
def get_fast_policy_answer(user_text: str) -> Optional[str]:
    q = user_text.replace(" ", "")
    if any(k in q for k in ["배송비", "무료배송"]):
        return (
            "배송은 {}로 보내드리고 있어요 :)\n"
            "배송비는 {:,}원이고, {:,}원 이상 구매하시면 무료예요.".format(
                POLICY_DB["shipping"]["courier"],
                POLICY_DB["shipping"]["shipping_fee"],
                POLICY_DB["shipping"]["free_shipping_over"],
            )
        )
    if any(k in q for k in ["출고", "당일출고", "언제와", "배송언제", "며칠", "얼마나걸려", "배송기간", "배송얼마나", "언제받아", "언제도착", "빨리와", "빨리받", "언제받을", "언제받나요", "언제오나요", "오늘주문", "오늘시키"]):
        return (
            "{}예요 :)\n보통 {} 정도 보시면 되고, 결제 순서대로 순차 출고돼요.".format(
                POLICY_DB["shipping"]["same_day_dispatch_rule"],
                POLICY_DB["shipping"]["delivery_time"],
            )
        )
    if "교환" in q:
        return (
            "교환은 가능해요 :)\n"
            "{} 안에 접수해주시면 되고, 단순 변심 교환은 왕복 {:,}원으로 안내드리고 있어요.".format(
                POLICY_DB["exchange_return"]["period"], POLICY_DB["exchange_return"]["exchange_fee"]
            )
        )
    if any(k in q for k in ["반품", "환불"]):
        return (
            "반품도 가능해요 :)\n{} 안에 접수해주시면 되고, {} 기준으로 진행돼요.".format(
                POLICY_DB["exchange_return"]["period"], POLICY_DB["exchange_return"]["return_fee_rule"]
            )
        )
    return None


# ── 사이즈 판단 ───────────────────────────────────────────────────────────────

def get_product_size_category(db_product: Optional[Dict], product_context: Dict) -> str:
    """★ 핵심: 상의 상품인지 하의 상품인지 DB sub_category 기준으로 정확히 판단"""
    sub = clean_text((db_product or {}).get("sub_category", "") or product_context.get("sub_category", ""))
    name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", ""))
    if sub in BOTTOM_SUB_CATS:
        return "bottom"
    if "스커트" in sub or "스커트" in name:
        return "bottom"
    if any(k in name for k in ["바지", "팬츠", "슬랙스", "데님", "청바지", "치마"]):
        return "bottom"
    return "top"


def infer_fit_ease_needed(db_product: Optional[Dict], product_context: Dict) -> int:
    sub = clean_text((db_product or {}).get("sub_category", "") or product_context.get("sub_category", ""))
    name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", ""))
    corpus = sub + " " + name
    if any(k in corpus for k in ["코트", "패딩"]):
        return 18
    if any(k in corpus for k in ["자켓", "재킷", "점퍼", "야상"]):
        return 14
    if any(k in corpus for k in ["맨투맨", "후드", "니트", "가디건"]):
        return 10
    if any(k in corpus for k in ["블라우스", "셔츠"]):
        return 8
    return 9


def parse_float_value(value) -> Optional[float]:
    text = clean_text(value).replace(",", "")
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None


def get_garment_chest(db_product: Optional[Dict]) -> Optional[float]:
    if not db_product:
        return None
    chest = parse_float_value(db_product.get("chest", ""))
    chest_type = clean_text(db_product.get("chest_measure_type", "")).lower()
    if chest is None:
        raw = clean_text(db_product.get("raw_measurements", ""))
        if raw:
            m = re.search(r"가슴둘레[^0-9]*(\d+(?:\.\d+)?)", raw)
            if m:
                chest = float(m.group(1)); chest_type = "circumference"
            else:
                m = re.search(r"가슴단면[^0-9]*(\d+(?:\.\d+)?)", raw)
                if m:
                    chest = float(m.group(1)); chest_type = "flat"
    if chest is None:
        return None
    return chest * 2 if chest_type.startswith("flat") else chest


def evaluate_size_support(user_top: str, product_context: Dict, db_product: Optional[Dict]) -> Dict:
    user_rank = size_rank(user_top)
    if not user_rank:
        return {"supported": None, "reason": "", "max_size": "", "confidence": "unknown"}
    db_range = clean_text((db_product or {}).get("size_range", ""))
    size_text = db_range or clean_text(product_context.get("size_tip", ""))
    ranks = expand_size_text(size_text)
    max_rank = max(ranks) if ranks else None
    max_size_label = rank_to_size(max_rank)

    if ranks:
        if user_rank not in ranks:
            return {"supported": False,
                    "reason": "이 상품은 최대 {}까지 나와요.".format(max_size_label),
                    "max_size": max_size_label, "confidence": "db"}
        garment_chest = get_garment_chest(db_product)
        body_bust = APPROX_BODY_BUST.get(user_rank)
        if user_rank == max_rank:
            if garment_chest and body_bust:
                ease = garment_chest - body_bust
                needed = infer_fit_ease_needed(db_product, product_context)
                if ease < 2:
                    return {"supported": False,
                            "reason": "최대 {}까지 나오는 상품인데, 실측 기준으로 여유가 부족할 수 있어요(의류 가슴둘레 약 {}cm).".format(
                                max_size_label, int(round(garment_chest))),
                            "max_size": max_size_label, "confidence": "db+measure"}
                if ease < needed:
                    return {"supported": "edge",
                            "reason": "최대 {}까지 나오는 상품이라 딱 경계 사이즈예요. 살짝 타이트하게 느껴질 수 있어요.".format(max_size_label),
                            "max_size": max_size_label, "confidence": "db+measure"}
            return {"supported": "edge",
                    "reason": "최대 {}까지 나오는 상품이라 고객님이 딱 상단 사이즈에 해당해요.".format(max_size_label),
                    "max_size": max_size_label, "confidence": "db"}
        # 범위 안, 최대 아님
        chest_note = ""
        if garment_chest and body_bust:
            ease = garment_chest - body_bust
            needed = infer_fit_ease_needed(db_product, product_context)
            if ease < 2:
                return {"supported": False,
                        "reason": "사이즈 범위에는 들어오지만 실측 기준으로 여유가 부족해요(의류 가슴둘레 약 {}cm).".format(int(round(garment_chest))),
                        "max_size": max_size_label, "confidence": "db+measure"}
            if ease < needed:
                chest_note = " 다만 가슴 쪽이 살짝 타이트하게 느껴지실 수 있어요(의류 가슴둘레 약 {}cm).".format(int(round(garment_chest)))
        return {"supported": True, "reason": "사이즈 범위 안에 들어와요.{}".format(chest_note),
                "max_size": max_size_label, "confidence": "db+measure" if garment_chest else "db"}
    return {"supported": None, "reason": "", "max_size": "", "confidence": "unknown"}


# ── 상품/색상/추천 헬퍼 ───────────────────────────────────────────────────────

def current_product_dict(product_context: Dict, db_product: Optional[Dict]) -> Dict:
    return {
        "product_no": normalize_product_no((db_product or {}).get("product_no", "") or product_context.get("product_no", "")),
        "product_name": clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품"),
        "category": clean_text((db_product or {}).get("category", "") or product_context.get("category", "") or "기타"),
        "sub_category": clean_text((db_product or {}).get("sub_category", "") or product_context.get("sub_category", "")),
        "size_range": clean_text((db_product or {}).get("size_range", "")),
        "style_tags": clean_text((db_product or {}).get("style_tags", "")),
        "coordination_items": clean_text((db_product or {}).get("coordination_items", "")),
        "body_cover_features": clean_text((db_product or {}).get("body_cover_features", "")),
        "fabric": clean_text((db_product or {}).get("fabric", "")) or clean_text(product_context.get("material", "")),
    }


def parse_color_options(product_context: Dict, db_product: Optional[Dict]) -> List[str]:
    colors: List[str] = []
    for source in [clean_text((db_product or {}).get("color_options", "")),
                   clean_text((product_context or {}).get("raw_excerpt", ""))]:
        for part in re.split(r"[;,/|]", source):
            token = clean_text(part)
            if token in COLOR_CANDIDATES and token not in colors:
                colors.append(token)
        for token in extract_colors_from_text(source):
            if token not in colors:
                colors.append(token)
    return colors


def row_blob(rowd: Dict) -> str:
    return " ".join(clean_text(rowd.get(c, "")) for c in [
        "product_name", "category", "sub_category", "style_tags", "coordination_items",
        "body_cover_features", "recommended_body_type", "product_summary", "fabric"
    ])


def infer_target_category_from_query(user_text: str, current_product: Dict) -> str:
    q = clean_text(user_text)
    # "점퍼나 자켓", "자켓이나 아우터" 같은 복수 표현 → 아우터(자켓+점퍼 통합)로 처리
    if any(k in q for k in ["아우터"]):
        return "아우터"
    # 자켓+점퍼 함께 언급 → 아우터
    has_jacket = any(k in q for k in ["자켓", "재킷"])
    has_jumper = any(k in q for k in ["점퍼", "야상"])
    if has_jacket and has_jumper:
        return "아우터"
    for kw, cat in [
        (["바지", "슬랙스", "팬츠", "데님", "청바지"], "팬츠"),
        (["스커트", "치마"], "스커트"),
        (["자켓", "재킷"], "자켓"),
        (["점퍼", "야상"], "점퍼"),
        (["맨투맨", "후드", "스웨트"], "맨투맨"),
        (["블라우스"], "블라우스"),
        (["셔츠"], "셔츠"),
        (["가디건"], "가디건"),
        (["니트"], "니트"),
        (["티셔츠", "반팔"], "티셔츠"),
        (["원피스"], "원피스"),
    ]:
        if any(k in q for k in kw):
            return cat
    # 코디 세트 요청 ("코디할 아이템", "전체 코디", "코디 추천") → 아우터 우선
    if any(k in q for k in ["코디할 아이템", "전체 코디", "세트 코디", "코디 세트", "코디 추천"]):
        return "코디세트"
    # ★ "비슷한 스타일", "입을 수 있는 상품", "같은 종류" → 현재 상품과 동일 카테고리
    same_cat_kws = ["비슷한스타일", "비슷한상품", "비슷한옷", "비슷한거", "비슷한티",
                    "입을수있는상품", "같은종류", "같은카테고리", "이런스타일",
                    "이런비슷한", "같은느낌", "이런류"]
    current_sub = clean_text(current_product.get("sub_category", ""))
    current_cat = clean_text(current_product.get("category", ""))
    if any(k in q for k in same_cat_kws):
        # 현재 상품 카테고리와 같은 카테고리 반환
        sub_to_cat = {
            "티셔츠": "티셔츠", "맨투맨": "맨투맨", "셔츠": "셔츠",
            "블라우스": "블라우스", "니트": "니트", "가디건": "가디건",
            "자켓": "자켓", "점퍼": "점퍼",
            "슬랙스": "팬츠", "데님": "팬츠", "팬츠": "팬츠",
            "스커트": "스커트", "원피스": "원피스",
        }
        return sub_to_cat.get(current_sub, "")
    # 현재 상품 기준으로 반대편 추천 (어울리는 것)
    if current_sub in BOTTOM_SUB_CATS:
        return "블라우스"
    if current_cat in TOP_MAIN_CATS or current_sub in TOP_SUB_CATS:
        return "팬츠"
    return ""


def build_product_reason(rowd: Dict, user_text: str, body_hints: Dict) -> List[str]:
    """4050 여성 상황별 추천 이유 생성"""
    reasons: List[str] = []
    blob = row_blob(rowd)
    name = clean_text(rowd.get("product_name", ""))
    situations = detect_situation_from_text(user_text)
    style_req = detect_style_from_text(user_text)
    cover = clean_text(rowd.get("body_cover_features", ""))
    sub = clean_text(rowd.get("sub_category", ""))

    # ── 상황별 이유
    if "상견례" in situations or "면접" in situations:
        reasons.append("격식 있는 자리에 단정하고 신뢰감 있는 분위기를 줘요")
    elif "결혼식" in situations:
        if any(k in blob for k in ["우아", "페미닌", "드레시", "러블리"]):
            reasons.append("결혼식 하객룩으로 우아하게 어울려요")
        else:
            reasons.append("결혼식 자리에 깔끔하게 잘 어울려요")
    elif "시댁" in situations:
        reasons.append("시댁 방문처럼 단정함이 필요한 자리에 딱 맞아요")
    elif "학교" in situations:
        reasons.append("학교 방문룩으로 단정하고 신뢰감 있게 입기 좋아요")
    elif "출근" in situations or "미팅" in situations:
        if any(k in blob for k in ["단정", "클래식", "오피스", "세미"]):
            reasons.append("출근룩으로 깔끔하고 세련되게 이어주기 좋아요")
        else:
            reasons.append("직장 자리에 무난하게 잘 어울려요")
    elif "동창회" in situations or "모임" in situations:
        reasons.append("오랜 친구 모임에서 세련되게 보여줄 수 있어요")
    elif "데이트" in situations:
        if any(k in blob for k in ["러블리", "페미닌", "여성스"]):
            reasons.append("기념일 외식이나 데이트 코디로 예쁘게 어울려요")
        else:
            reasons.append("특별한 날 코디로 자연스럽게 잘 어울려요")
    elif "소개팅" in situations:
        reasons.append("소개팅에서 나이에 맞게 세련되고 단정한 인상을 줄 수 있어요")
    elif "여행" in situations:
        reasons.append("여행할 때 편하게 입기 좋은 편이에요")

    # ── 스타일 요청별 이유
    if style_req == "격식" and not reasons:
        reasons.append("격식 있는 자리에 단정하게 잘 어울려요")
    elif style_req == "세련" and not reasons:
        reasons.append("세련된 분위기로 연출하기 좋아요")
    elif style_req == "동안" and not reasons:
        reasons.append("나이보다 젊고 활기차 보이는 스타일이에요")
    elif style_req == "캐주얼" and not reasons:
        reasons.append("편하고 자연스럽게 일상에서 입기 좋아요")

    # ── 체형 커버 이유
    if body_hints.get("belly_hint") and any(k in cover for k in ["복부", "뱃살", "허리"]):
        reasons.append("복부 라인을 자연스럽게 커버해줘요")
    elif body_hints.get("arm_hint") and any(k in cover for k in ["팔뚝", "소매"]):
        reasons.append("팔뚝이 신경 쓰이는 분께 소매 디자인이 도움이 돼요")
    elif body_hints.get("lower_body_hint") and any(k in cover for k in ["힙", "허벅지", "하체"]):
        reasons.append("하체 라인 부담이 적은 편이에요")
    elif body_hints.get("upper_body_hint") == "상체큰편" and any(k in cover for k in ["어깨", "상체", "가슴"]):
        reasons.append("상체 라인을 자연스럽게 잡아줘요")

    # ── 상품 유형별 기본 이유 (이유 없을 때) — sub_category 기반
    if not reasons:
        sub = clean_text(rowd.get("sub_category", ""))
        fit = clean_text(rowd.get("fit_type", ""))
        if sub == "슬랙스" or "슬랙스" in name:
            reasons.append("라인이 정돈되어 상의를 깔끔하게 살려줘요")
        elif sub == "데님" or any(k in name for k in ["데님", "청바지"]):
            reasons.append("편하면서도 세련되게 매치하기 좋아요")
        elif sub in ["자켓"] or any(k in name for k in ["자켓", "재킷"]):
            reasons.append("전체 실루엣을 단정하게 잡아주는 편이에요")
        elif sub == "점퍼" or any(k in name for k in ["점퍼", "야상"]):
            reasons.append("가볍게 걸치기 좋고 활동적인 느낌이에요")
        elif sub == "블라우스" or "블라우스" in name:
            reasons.append("여성스럽고 단정한 분위기를 줘요")
        elif sub == "가디건" or "가디건" in name:
            reasons.append("레이어드하기 좋고 부드러운 느낌이에요")
        elif sub == "니트" or "니트" in name:
            reasons.append("따뜻하고 부드러운 소재로 편하게 입기 좋아요")
        elif sub == "셔츠" or any(k in name for k in ["셔츠"]):
            reasons.append("단정하면서도 활용도 높은 스타일이에요")
        elif sub == "티셔츠" or any(k in name for k in ["티셔츠", "반팔", "5부"]):
            reasons.append("깔끔하고 편하게 데일리로 입기 좋아요")
        elif sub in ["팬츠"] or any(k in name for k in ["팬츠", "바지"]):
            reasons.append("편하면서도 세련된 라인을 만들어줘요")
        elif sub == "스커트" or any(k in name for k in ["스커트", "치마"]):
            reasons.append("여성스럽게 코디하기 좋아요")
        elif sub == "원피스" or "원피스" in name:
            reasons.append("원피스 하나로 완성되는 간편한 코디예요")
        else:
            style_tags = clean_text(rowd.get("style_tags", ""))
            if "클래식" in style_tags or "단정" in style_tags:
                reasons.append("단정하고 세련된 분위기를 줘요")
            elif "캐주얼" in style_tags or "데일리" in style_tags:
                reasons.append("편하게 일상에서 입기 좋아요")
            else:
                reasons.append("깔끔하게 코디하기 좋은 스타일이에요")

    seen, out = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r); out.append(r)
    return out[:2]



def save_recommendations(recos: List[Dict]) -> None:
    try:
        cleaned = []
        for reco in recos:
            rowd = reco.get("_full_row", {})
            cleaned.append({
                "product_name": clean_text(reco.get("product_name", "")),
                "product_no": normalize_product_no(clean_text(rowd.get("product_no", "") or reco.get("product_no", ""))),
                "category": clean_text(reco.get("category", "")),
                "sub_category": clean_text(reco.get("sub_category", "")),
                "size_range": clean_text(reco.get("size_range", "")),
                "reasons": (reco.get("reasons") or [])[:2],
                "_full_row": rowd if isinstance(rowd, dict) else {},
            })
        st.session_state.last_recommendations = cleaned
    except Exception:
        st.session_state.last_recommendations = []


def get_recommendation_reference_index(user_text: str) -> Optional[int]:
    q = user_text.replace(" ", "")
    mapping = {
        0: ["1번", "첫번째", "첫번째상품", "첫번째옷", "첫째"],
        1: ["2번", "두번째", "두번째상품", "두번째옷", "둘째"],
        2: ["3번", "세번째", "세번째상품", "세번째옷", "셋째"],
    }
    for idx, words in mapping.items():
        if any(w in q for w in words):
            return idx
    # 추천 결과 전체 참조 표현 → idx=0 (가장 최근 추천 기준)
    followup_kws = [
        "방금추천", "추천해준", "추천한거", "추천한게", "추천해준게",
        "그거", "그상품", "그옷", "그바지", "그자켓", "그아우터",
        "방금거", "아까거", "아까추천",
    ]
    if any(w in q for w in followup_kws):
        return 0
    return None
def get_followup_recommendation(user_text: str) -> Optional[Dict]:
    idx = get_recommendation_reference_index(user_text)
    recos = st.session_state.get("last_recommendations", []) or []
    if idx is None or idx >= len(recos):
        return None
    return recos[idx]


def recommendation_to_context(reco: Dict) -> Tuple[Dict, Optional[Dict]]:
    rowd = reco.get("_full_row", {}) or {}
    db_like = rowd or {
        "product_no": clean_text(reco.get("product_no", "")),
        "product_name": clean_text(reco.get("product_name", "")),
        "category": clean_text(reco.get("category", "")),
        "sub_category": clean_text(reco.get("sub_category", "")),
        "size_range": clean_text(reco.get("size_range", "")),
    }
    raw_blob = " ".join(clean_text(db_like.get(c, "")) for c in [
        "product_name", "category", "sub_category", "fit_type", "body_cover_features",
        "style_tags", "coordination_items", "product_summary", "fabric", "size_range", "color_options"
    ])
    return {
        "product_no": normalize_product_no(clean_text(db_like.get("product_no", ""))),
        "product_name": clean_text(db_like.get("product_name", "") or reco.get("product_name", "") or "추천드린 상품"),
        "category": clean_text(db_like.get("category", "") or reco.get("category", "") or "기타"),
        "sub_category": clean_text(db_like.get("sub_category", "") or reco.get("sub_category", "")),
        "summary": clean_text(db_like.get("product_summary", "")),
        "material": clean_text(db_like.get("fabric", "")),
        "fit": " / ".join(x for x in [clean_text(db_like.get("fit_type", "")), clean_text(db_like.get("body_cover_features", ""))] if x),
        "size_tip": clean_text(db_like.get("size_range", "") or reco.get("size_range", "")),
        "raw_excerpt": raw_blob,
        "colors": parse_color_options({"raw_excerpt": raw_blob}, db_like if db_like else None),
    }, db_like if db_like else None


def build_name_answer(product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", ""))
    if not name or name == "지금 보시는 상품":
        return None
    return "지금 보시는 상품은 {}이에요 :)".format(name)


def build_color_answer(product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    colors = parse_color_options(product_context, db_product)
    if not colors:
        return None
    return "현재 확인되는 컬러는 {} 쪽이에요. 없는 컬러를 임의로 말씀드리기보다는 지금 보이는 옵션 기준으로 같이 봐드릴게요 :)".format(", ".join(colors))


def build_size_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    if not is_size_question(user_text):
        return None
    body = build_body_context()
    body_hints = extract_user_body_from_text(user_text)
    size_cat = get_product_size_category(db_product, product_context)
    current_name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품")

    if size_cat == "bottom":
        user_size = clean_text(body.get("bottom_size", "")) or detect_size_from_text(user_text) or ""
        size_label = "하의"
        if not user_size:
            return "{}은 하의 상품이에요 :)\n하의 사이즈를 알려주시면 더 정확하게 봐드릴 수 있어요.".format(current_name)
    else:
        user_size = clean_text(body.get("top_size", "")) or detect_size_from_text(user_text) or ""
        size_label = "상의"
        if not user_size:
            if body_hints:
                return "체형 정보 감사해요 :) 상의 사이즈를 알려주시면 {}에 맞는지 더 정확하게 봐드릴 수 있어요.\n평소 상의 사이즈가 어떻게 되세요?".format(current_name)
            return "상의 사이즈를 알려주시면 더 정확하게 봐드릴 수 있어요 :) 평소 상의 사이즈가 어떻게 되세요?"

    size_eval = evaluate_size_support(user_size, product_context, db_product)
    reason = clean_text(size_eval.get("reason", ""))
    max_size = size_eval.get("max_size", "")
    body_note = ""
    if body_hints.get("belly_hint") and size_cat == "top":
        body_note = " 배 부분이 신경 쓰이신다고 하셨으니 상세페이지 실측표에서 허리 부분도 같이 확인해보시는 게 좋아요."
    elif body_hints.get("arm_hint") and size_cat == "top":
        body_note = " 팔뚝이 신경 쓰이신다고 하셨으니 소매 사이즈와 실측표를 같이 봐주시면 좋아요."
    elif body_hints.get("upper_body_hint") == "상체큰편" and size_cat == "top":
        body_note = " 상체가 있는 편이라고 하셨으니 가슴/어깨 쪽 여유도 실측표로 같이 확인해보시는 게 좋아요."

    if size_eval["supported"] is False:
        # ★ 77반→77까지 상품이면 경계+1로 부드럽게 안내
        user_r = size_rank(user_size)
        max_s = max_size or ""
        max_r = SIZE_ORDER.get(max_s, 0) if max_s else 0
        if user_r and max_r and user_r == max_r + 1:
            return "고객님 {} {} 기준이면 {}은 딱 경계 사이즈 바로 위예요.\n이 상품은 최대 {}까지 나와서 핏이 살짝 빡빡하게 느껴질 수 있어요.\n실측표를 꼭 같이 보시는 게 안전해요.{}".format(
                size_label, user_size, current_name, max_s, body_note)
        return "고객님 {} {} 기준이면 {}은 맞는 사이즈가 없어요.\n{}\n다른 상품을 같이 찾아봐드릴까요?{}".format(
            size_label, user_size, current_name, reason or "이 상품은 최대 {}까지 나와요.".format(max_s), body_note)
    if size_eval["supported"] == "edge":
        return "고객님 {} {} 기준이면 {}은 가능은 하지만 딱 경계 사이즈예요.\n{}\n편하게 입는 걸 좋아하시면 실측표를 꼭 같이 보시는 게 안전해요.{}".format(
            size_label, user_size, current_name, reason or "상단 사이즈라 체감이 조금 타이트할 수 있어요.", body_note)
    if size_eval["supported"] is True:
        return "고객님 {} {} 기준이면 {} 사이즈 범위 안에 들어와요 :)\n{}\n원하시는 핏에 따라 체감은 달라질 수 있으니, 상세페이지 실측표도 함께 보시면 더 정확해요.{}".format(
            size_label, user_size, current_name, reason, body_note)
    return "{} 사이즈 정보를 지금 정확히 확인하기 어려워서, 상세페이지 실측표를 같이 보시는 쪽이 제일 안전해요 :)".format(current_name)


def recommend_products_for_query(
    user_text: str, current_product: Dict, body_ctx: Dict[str, str],
    target_category: str = "", limit: int = 3, exclude_names: Optional[set] = None
) -> List[Dict]:
    if DB.empty:
        return []
    if not target_category:
        target_category = infer_target_category_from_query(user_text, current_product)
    current_no = normalize_product_no(current_product.get("product_no", ""))
    top_rank = size_rank(body_ctx.get("top_size", ""))
    bottom_rank = size_rank(body_ctx.get("bottom_size", ""))
    body_hints = extract_user_body_from_text(user_text)
    situations = detect_situation_from_text(user_text)
    exclude_names = exclude_names or set()

    scored: List[Tuple[int, Dict]] = []
    for _, row in DB.iterrows():
        rowd = row.to_dict()
        row_no = normalize_product_no(rowd.get("product_no", ""))
        row_name = clean_text(rowd.get("product_name", ""))
        if (current_no and row_no == current_no) or row_name in exclude_names or not row_name:
            continue
        sub = clean_text(rowd.get("sub_category", ""))
        cat = clean_text(rowd.get("category", ""))
        ranks = expand_size_text(clean_text(rowd.get("size_range", "")))
        score = 0

        # ★ 카테고리 필터 - DB 실제 sub_category 값 기준
        if target_category == "팬츠":
            if not (sub in {"데님", "슬랙스", "팬츠"} or any(k in row_name for k in ["바지", "팬츠", "슬랙스", "데님", "청바지"])):
                continue
            if "스커트" in sub or "스커트" in row_name:
                continue
            if bottom_rank and ranks and bottom_rank not in ranks:
                continue
            score += 15
        elif target_category == "스커트":
            if not ("스커트" in sub or "스커트" in row_name):
                continue
            if bottom_rank and ranks and bottom_rank not in ranks:
                continue
            score += 15
        elif target_category == "아우터":
            # 자켓 + 점퍼 통합 (아우터 전체)
            if not (sub in {"자켓", "점퍼"} or any(k in row_name for k in ["자켓", "재킷", "점퍼", "야상", "코트", "패딩"])):
                continue
            # 사이즈: top_rank 있으면 체크, 없으면 통과
            # ★ 핵심 완화: max(ranks)+1까지 허용 (77반→77까지 자켓 경계 사이즈로 포함)
            if top_rank and ranks:
                max_r = max(ranks)
                if top_rank > max_r + 1:  # max보다 2 이상 크면 제외
                    continue
                if top_rank == max_r + 1:  # 딱 경계+1이면 포함하되 낮은 점수
                    score += 8
                else:
                    score += 15
            else:
                score += 15
            if sub == "자켓":
                score += 3  # 자켓 우선
        elif target_category == "자켓":
            if not (sub == "자켓" or any(k in row_name for k in ["자켓", "재킷"])):
                continue
            if top_rank and ranks:
                max_r = max(ranks)
                if top_rank > max_r + 1:
                    continue
                if top_rank == max_r + 1:
                    score += 8
                else:
                    score += 15
            else:
                score += 15
        elif target_category == "점퍼":
            if not (sub == "점퍼" or any(k in row_name for k in ["점퍼", "야상"])):
                continue
            if top_rank and ranks:
                max_r = max(ranks)
                if top_rank > max_r + 1:
                    continue
                if top_rank == max_r + 1:
                    score += 8
                else:
                    score += 15
            else:
                score += 15
        elif target_category == "맨투맨":
            if not any(k in row_name for k in ["맨투맨", "후드", "스웨트"]):
                continue
            if top_rank and ranks and top_rank not in ranks:
                continue
            score += 15
        elif target_category == "블라우스":
            if not (sub == "블라우스" or "블라우스" in row_name):
                continue
            if top_rank and ranks and top_rank not in ranks:
                continue
            score += 15
        elif target_category == "셔츠":
            if not (sub == "셔츠" or "셔츠" in row_name):
                continue
            if top_rank and ranks and top_rank not in ranks:
                continue
            score += 15
        elif target_category == "가디건":
            if not (sub == "가디건" or "가디건" in row_name):
                continue
            if top_rank and ranks and top_rank not in ranks:
                continue
            score += 15
        elif target_category == "니트":
            if not (sub == "니트" or "니트" in row_name):
                continue
            if top_rank and ranks and top_rank not in ranks:
                continue
            score += 15
        elif target_category == "티셔츠":
            if not (sub == "티셔츠" or cat == "티셔츠"):
                continue
            if top_rank and ranks and top_rank not in ranks:
                continue
            score += 15
        elif target_category == "원피스":
            if not ("원피스" in sub or "원피스" in cat):
                continue
            score += 15
        elif target_category == "코디세트":
            # 상의+하의+아우터 조합 → 자켓/슬랙스 우선
            is_outer = sub in {"자켓", "점퍼"} or any(k in row_name for k in ["자켓", "재킷", "점퍼"])
            is_bottom = sub in BOTTOM_SUB_CATS
            is_top = sub in TOP_SUB_CATS and sub not in {"자켓", "점퍼"}
            if not (is_outer or is_bottom or is_top):
                continue
            # 사이즈 필터 (아우터는 경계 허용)
            if is_outer and top_rank and ranks:
                if top_rank > max(ranks) + 1:
                    continue
            elif is_bottom and bottom_rank and ranks and bottom_rank not in ranks:
                continue
            elif is_top and top_rank and ranks and top_rank not in ranks:
                continue
            score += 12
            if is_outer and sub == "자켓": score += 5  # 자켓 최우선
            if is_bottom and "슬랙스" in sub: score += 3  # 슬랙스 우선

        blob = row_blob(rowd)
        for sit in situations:
            if sit in ["학교", "방문", "출근"] and any(k in blob for k in ["단정", "클래식", "오피스", "세미"]):
                score += 6
            elif sit in ["모임", "친구", "행사"] and any(k in blob for k in ["캐주얼", "데일리"]):
                score += 4
            elif sit == "데이트" and any(k in blob for k in ["러블리", "페미닌", "여성스"]):
                score += 5
        cover = clean_text(rowd.get("body_cover_features", ""))
        if body_hints.get("belly_hint") and any(k in cover for k in ["복부", "뱃살"]):
            score += 5
        if body_hints.get("lower_body_hint") and any(k in cover for k in ["힙", "허벅지"]):
            score += 5
        if current_product.get("style_tags"):
            curr_tags = set(x.strip() for x in re.split(r"[;,/|]", current_product.get("style_tags", "")) if x.strip())
            row_tags = set(x.strip() for x in re.split(r"[;,/|]", rowd.get("style_tags", "")) if x.strip())
            score += len(curr_tags & row_tags) * 2
        scored.append((score, rowd))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict] = []
    seen_names: set = set()
    for _, rowd in scored:
        name = clean_text(rowd.get("product_name", ""))
        if not name or name in seen_names or name in exclude_names:
            continue
        seen_names.add(name)
        out.append({
            "product_name": name,
            "product_no": normalize_product_no(clean_text(rowd.get("product_no", ""))),
            "category": clean_text(rowd.get("category", "")),
            "sub_category": clean_text(rowd.get("sub_category", "")),
            "size_range": clean_text(rowd.get("size_range", "")),
            "reasons": build_product_reason(rowd, user_text, body_hints),
            "_full_row": rowd,
        })
        if len(out) >= limit:
            break
    return out



def _db_size_availability(target_category: str, user_top: str, current_no: str) -> Dict:
    """
    DB 기준으로 해당 카테고리에서 고객 사이즈 수용 가능 상품 현황 반환
    returns: {exact: [(name, size_range)], boundary: [(name, size_range)], none: bool}
    """
    if DB.empty or not user_top:
        return {"exact": [], "boundary": [], "none": False}
    user_r = size_rank(user_top)
    if not user_r:
        return {"exact": [], "boundary": [], "none": False}

    cat_filter = {
        "아우터": lambda sub, name: sub in {"자켓", "점퍼"} or any(k in name for k in ["자켓","재킷","점퍼","야상","코트"]),
        "자켓": lambda sub, name: sub == "자켓" or any(k in name for k in ["자켓","재킷"]),
        "점퍼": lambda sub, name: sub == "점퍼" or any(k in name for k in ["점퍼","야상"]),
        "팬츠": lambda sub, name: sub in {"슬랙스","데님","팬츠"} or any(k in name for k in ["바지","팬츠","슬랙스","데님"]),
        "맨투맨": lambda sub, name: any(k in name for k in ["맨투맨","후드","스웨트"]),
        "블라우스": lambda sub, name: sub == "블라우스" or "블라우스" in name,
        "셔츠": lambda sub, name: sub == "셔츠" or "셔츠" in name,
        "니트": lambda sub, name: sub == "니트" or "니트" in name,
        "가디건": lambda sub, name: sub == "가디건" or "가디건" in name,
    }
    fn = cat_filter.get(target_category)
    if not fn:
        return {"exact": [], "boundary": [], "none": False}

    exact, boundary = [], []
    for _, row in DB.iterrows():
        pno = normalize_product_no(row.get("product_no", ""))
        if current_no and pno == current_no:
            continue
        sub = clean_text(row.get("sub_category", ""))
        name = clean_text(row.get("product_name", ""))
        if not fn(sub, name):
            continue
        ranks = expand_size_text(clean_text(row.get("size_range", "")))
        if not ranks:
            continue
        max_r = max(ranks)
        if user_r in ranks:
            exact.append((name, clean_text(row.get("size_range", ""))))
        elif user_r == max_r + 1:
            boundary.append((name, clean_text(row.get("size_range", ""))))

    return {"exact": exact, "boundary": boundary, "none": len(exact) == 0 and len(boundary) == 0}

def build_recommendation_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    if not is_recommendation_question(user_text):
        return None
    current_product = current_product_dict(product_context, db_product)
    body_ctx = build_body_context()
    target_category = infer_target_category_from_query(user_text, current_product)
    is_re_request = any(k in user_text for k in ["다른", "또 다른", "더 없어", "다른 거"])
    prev_recos = st.session_state.get("last_recommendations") or []
    already = {r["product_name"] for r in prev_recos}
    # ★ 같은 카테고리를 다시 요청하는 경우도 이전 추천 제외 (단순 반복 방지)
    if prev_recos and not is_re_request:
        prev_cat = infer_target_category_from_query(
            " ".join(r.get("sub_category","") for r in prev_recos[:1]),
            current_product
        )
        curr_cat = infer_target_category_from_query(user_text, current_product)
        if prev_cat and curr_cat and prev_cat == curr_cat:
            is_re_request = True
    exclude = already if is_re_request else set()
    recos = recommend_products_for_query(user_text, current_product, body_ctx, target_category, 3, exclude)
    if not recos and is_re_request:
        recos = recommend_products_for_query(user_text, current_product, body_ctx, target_category, 3)
    if not recos:
        # 자켓 경계 사이즈(77반→77)인 경우 경계 허용 재시도
        if target_category in ["자켓", "아우터"] and top_rank:
            recos = recommend_products_for_query(user_text, current_product, body_ctx, target_category, 3, set())
        if not recos:
            size_note = ""
            body_ctx_vals = build_body_context()
            top_s = clean_text(body_ctx_vals.get("top_size", ""))
            if top_s:
                size_note = " (고객님 {} 기준 경계 사이즈 상품도 함께 봐드릴 수 있어요.)".format(top_s)
            return "지금 조건에서 딱 맞는 {} 상품을 찾기 어렵네요.{} 카테고리나 사이즈를 조금 다르게 말씀해주시면 다시 찾아볼게요 :)".format(target_category or "", size_note)
    save_recommendations(recos)
    situations = detect_situation_from_text(user_text)
    if situations:
        opener = "네, {} 자리에 어울릴 만한 쪽으로 골라드릴게요 :)".format(situations[0])
    elif target_category == "팬츠":
        opener = "네, 이 상품이랑 잘 어울리는 바지 쪽으로 먼저 골라드릴게요."
    elif target_category == "스커트":
        opener = "네, 이 상품 분위기랑 잘 맞는 스커트로 먼저 골라드릴게요."
    elif target_category == "아우터":
        opener = "네, 고객님 사이즈에 맞는 아우터 쪽으로 먼저 골라드릴게요."
    elif target_category in ["자켓"]:
        opener = "네, 사이즈에 맞는 자켓 쪽으로 먼저 골라드릴게요."
    elif target_category == "점퍼":
        opener = "네, 사이즈에 맞는 점퍼 쪽으로 먼저 골라드릴게요."
    elif target_category == "맨투맨":
        opener = "네, 고객님 사이즈에 맞는 맨투맨으로 골라드릴게요."
    elif target_category == "코디세트":
        situations_str = situations[0] if situations else "모임"
        opener = "네, {} 자리에 맞는 코디 아이템들을 골라드릴게요 :) 자켓 → 하의 순으로 봐드릴게요.".format(situations_str)
    elif is_re_request:
        opener = "네, 다른 쪽으로 더 골라드릴게요."
    else:
        opener = "네, 고객님께 잘 맞을 만한 쪽으로 먼저 골라드릴게요."
    lines = [opener]
    for i, reco in enumerate(recos, 1):
        reason_text = " ".join((reco.get("reasons") or [])[:2]).strip()
        size_info = " ({})".format(reco["size_range"]) if reco.get("size_range") else ""
        lines.append("{}. {}{} — {}".format(i, reco["product_name"], size_info, reason_text) if reason_text
                     else "{}. {}{}".format(i, reco["product_name"], size_info))
    lines.append("마음 가는 번호 말씀해주시면 그 상품 기준으로 사이즈감도 바로 이어서 봐드릴게요 :)")
    return "\n".join(lines)


def _all_recos_size_summary(user_size: str) -> Optional[str]:
    """'추천해준 게 다 맞아?' 패턴 → 전체 추천 목록 사이즈 한번에 안내"""
    recos = st.session_state.get("last_recommendations") or []
    if not recos or not user_size:
        return None
    user_r = size_rank(user_size)
    if not user_r:
        return None

    lines = ["{}반 기준으로 추천드린 상품들 사이즈 확인해드릴게요 :)".format(user_size) if "반" in user_size
             else "{} 기준으로 추천드린 상품들 사이즈 확인해드릴게요 :)".format(user_size)]
    has_boundary = False
    for i, reco in enumerate(recos, 1):
        reco_context, reco_db = recommendation_to_context(reco)
        name = reco_context["product_name"]
        size_eval = evaluate_size_support(user_size, reco_context, reco_db)
        max_s = size_eval.get("max_size", "")
        max_r = SIZE_ORDER.get(max_s, 0) if max_s else 0

        if size_eval["supported"] is True:
            lines.append("{}. {} — ✅ 사이즈 범위 안에 들어와요".format(i, name))
        elif size_eval["supported"] == "edge":
            lines.append("{}. {} — ⚠️ 딱 경계 사이즈, 실측표 확인 필요".format(i, name))
            has_boundary = True
        elif size_eval["supported"] is False and user_r and max_r and user_r == max_r + 1:
            lines.append("{}. {} ({}) — ⚠️ 최대 {}까지라 경계 바로 위, 실측표 확인 필요".format(
                i, name, max_s, max_s))
            has_boundary = True
        else:
            lines.append("{}. {} — ❌ 사이즈 없어요 (최대 {}까지)".format(i, name, max_s or "확인필요"))

    if has_boundary:
        lines.append("")
        lines.append("⚠️ 표시 상품은 실측표를 꼭 같이 봐주세요. 번호 말씀해주시면 더 자세히 안내해드릴게요 :)")
    return "\n".join(lines)


def _followup_size_answer(idx: int, name: str, user_size: str, reco_context: Dict, reco_db: Optional[Dict]) -> str:
    """추천 번호 선택 시 DB 기준 정직한 사이즈 안내 — 핵심 함수"""
    size_cat = get_product_size_category(reco_db, reco_context)
    size_label = "하의" if size_cat == "bottom" else "상의"

    # 사용할 사이즈 결정
    body = build_body_context()
    if size_cat == "bottom":
        user_size = user_size or clean_text(body.get("bottom_size", "")) or detect_size_from_text("") or ""
    else:
        user_size = user_size or clean_text(body.get("top_size", "")) or ""

    if not user_size:
        return "{}번으로 추천드린 {} 기준으로 사이즈 보려면 고객님 {} 사이즈를 먼저 알려주세요 :)".format(
            idx, name, size_label)

    size_eval = evaluate_size_support(user_size, reco_context, reco_db)
    max_size = size_eval.get("max_size", "")
    user_r = size_rank(user_size)
    max_r = SIZE_ORDER.get(max_size, 0) if max_size else 0

    # ── 경계 (상단 사이즈): 딱 최대 사이즈인 경우 — False 체크보다 반드시 먼저!
    if size_eval["supported"] == "edge":
        reason = clean_text(size_eval.get("reason", ""))
        return (
            "{}번으로 추천드린 {}은 고객님 {} {} 기준으로 딱 상단 사이즈예요.\n"
            "{}\n"
            "실측표를 꼭 같이 보시는 게 안전해요 :)".format(
                idx, name, size_label, user_size,
                reason or "최대 사이즈라 체감이 약간 타이트할 수 있어요.")
        )

    # ── 완전 불가 (범위 초과)
    if size_eval["supported"] is False and not (user_r and max_r and user_r == max_r + 1):
        return (
            "{}번으로 추천드린 {}은 고객님 {} {} 기준으로는 사이즈가 없어요.\n"
            "이 상품은 최대 {}까지 나와요.\n"
            "다른 상품을 찾아드릴까요?".format(
                idx, name, size_label, user_size, max_size or "확인 필요")
        )

    # ── 경계+1 (77반 → 77까지 상품)
    if size_eval["supported"] is False and user_r and max_r and user_r == max_r + 1:
        return (
            "{}번으로 추천드린 {}은 최대 {}까지 나오는 상품이에요.\n"
            "고객님 {} {}이 딱 경계 바로 위라 사이즈상으로는 안 맞아요.\n"
            "실측표를 꼭 확인해보시면 의외로 가능한 경우도 있어요 :)".format(
                idx, name, max_size, size_label, user_size)
        )

    # ── 완전 포함
    if size_eval["supported"] is True:
        reason = clean_text(size_eval.get("reason", ""))
        return (
            "{}번으로 추천드린 {}은 고객님 {} {} 기준으로 사이즈 범위 안에 들어와요 :)\n"
            "{}\n"
            "부담 없이 입는 쪽으로 비교적 안정적인 편이에요.".format(
                idx, name, size_label, user_size,
                reason or "사이즈 범위 안쪽으로 확인돼요.")
        )

    # ── 정보 부족
    db_range = clean_text((reco_db or {}).get("size_range", ""))
    return (
        "{}번으로 추천드린 {}은 현재 {} 쪽이에요.\n"
        "정확한 핏은 상세페이지 실측표를 같이 보시는 쪽이 제일 안전해요 :)".format(
            idx, name, db_range or "사이즈 정보 확인 필요")
    )


def build_followup_recommendation_answer(user_text: str) -> Optional[str]:
    reco = get_followup_recommendation(user_text)
    if not reco:
        return None
    reco_context, reco_db = recommendation_to_context(reco)
    idx = (get_recommendation_reference_index(user_text) or 0) + 1
    name = reco_context["product_name"]

    # ── 상품 설명 요청
    if is_name_question(user_text) or any(k in user_text for k in ["어떤 옷", "어떤 바지", "설명", "알려줘", "뭐야"]):
        reasons = [clean_text(x) for x in (reco.get("reasons") or []) if clean_text(x)]
        reason_line = " ".join(reasons[:2]) if reasons else "고객님 스타일에 맞게 골라드린 상품이에요."
        return "{}번으로 추천드린 상품은 {}이에요 :)\n{}\n사이즈나 코디가 궁금하시면 말씀해주세요.".format(
            idx, name, reason_line)

    # ── 컬러 질문
    if is_color_question(user_text):
        ans = build_color_answer(reco_context, reco_db)
        return "{}번으로 추천드린 {} 기준으로 보면, {}".format(idx, name, ans) if ans else \
               "{}번으로 추천드린 {}은 현재 컬러 정보가 정확히 확인되지 않아요.".format(idx, name)

    # ★ 핵심 분기: 번호 + 추천 요청 → 해당 상품을 기준으로 추천
    # "3번 티랑 바지 추천해줘" → 3번 상품을 current_product로 삼아 추천
    ref_idx = get_recommendation_reference_index(user_text)
    q_raw = user_text.replace(" ", "")

    if is_recommendation_question(user_text):
        # 번호 상품을 기준 상품으로 해서 추천
        fake_product = current_product_dict(reco_context, reco_db)
        body_ctx = build_body_context()
        target_cat = infer_target_category_from_query(user_text, fake_product)
        recos = recommend_products_for_query(user_text, fake_product, body_ctx, target_cat, limit=3)
        if recos:
            save_recommendations(recos)
            situations = detect_situation_from_text(user_text)
            if situations:
                opener = "{}번 {} 기준으로 {} 자리에 어울릴 만한 쪽으로 골라드릴게요 :)".format(idx, name, situations[0])
            elif target_cat == "팬츠":
                opener = "{}번 {}이랑 잘 어울리는 바지 쪽으로 먼저 골라드릴게요.".format(idx, name)
            elif target_cat in ["자켓","아우터","점퍼"]:
                opener = "{}번 {}에 잘 어울리는 아우터로 먼저 골라드릴게요.".format(idx, name)
            else:
                opener = "{}번 {} 기준으로 어울리는 상품을 골라드릴게요.".format(idx, name)
            lines = [opener]
            for i2, r in enumerate(recos, 1):
                reason_text = " ".join((r.get("reasons") or [])[:2]).strip()
                size_info = " ({})".format(r["size_range"]) if r.get("size_range") else ""
                lines.append("{}. {}{} — {}".format(i2, r["product_name"], size_info, reason_text) if reason_text
                             else "{}. {}{}".format(i2, r["product_name"], size_info))
            lines.append("마음 가는 번호 말씀해주시면 사이즈감도 바로 봐드릴게요 :)")
            return "\n".join(lines)
        # 추천 결과 없으면 fallback
        return "{}번 {} 기준으로 조건에 맞는 상품을 찾기 어렵네요. 카테고리를 조금 다르게 말씀해주시면 다시 찾아볼게요 :)".format(idx, name)

    # ── 사이즈 의도 감지
    size_intent_kws = [
        "맞아", "맞아요", "맞나", "맞나요", "맞을까", "맞을까요",
        "될까", "될까요", "되나요", "가능해요", "가능한가요",
        "빡빡", "타이트", "여유", "사이즈",
        "맞는거야", "맞는거죠", "맞는건가요",
    ]
    has_size_intent = (
        detect_size_from_text(user_text) is not None or
        any(k in q_raw for k in size_intent_kws)
    )

    body = build_body_context()
    size_cat = get_product_size_category(reco_db, reco_context)
    user_size = (
        clean_text(body.get("bottom_size", "")) if size_cat == "bottom"
        else clean_text(body.get("top_size", ""))
    ) or detect_size_from_text(user_text) or ""

    # "추천해준 게 다 맞아?" → 전체 목록 사이즈 요약
    is_all_check = (
        ref_idx is not None and has_size_intent and
        clean_text(user_text).replace(" ","") not in ["1번","2번","3번","첫번째","두번째","세번째","첫째","둘째","셋째"] and
        any(k in q_raw for k in ["추천해준","방금추천","추천한거","추천한게","다맞아","전부맞","모두맞","다가능","전부가능"])
    )
    if is_all_check and user_size:
        summary = _all_recos_size_summary(user_size)
        if summary:
            return summary

    # 번호 선택이거나 사이즈 의도 → 개별 사이즈 확인
    if ref_idx is not None or has_size_intent:
        return _followup_size_answer(idx, name, user_size, reco_context, reco_db)

    return None


# ── 컬러매치 기본 지식 ────────────────────────────────────────────────────────
COLOR_MATCH_GUIDE = {
    "화이트": ["블랙", "네이비", "베이지", "카키", "그레이", "데님 계열"],
    "블랙": ["화이트", "그레이", "베이지", "레드", "핑크", "카키"],
    "아이보리": ["블랙", "브라운", "카키", "베이지", "네이비", "버건디"],
    "베이지": ["블랙", "화이트", "브라운", "카키", "네이비", "올리브"],
    "그레이": ["블랙", "화이트", "네이비", "핑크", "버건디"],
    "네이비": ["화이트", "아이보리", "베이지", "그레이"],
    "카키": ["화이트", "아이보리", "베이지", "블랙", "브라운"],
    "핑크": ["화이트", "그레이", "블랙", "네이비", "베이지"],
    "브라운": ["베이지", "아이보리", "카키", "크림", "화이트"],
}

BODY_STYLE_GUIDE = {
    "날씬": [
        "상하의 동일 계열 컬러로 세로 라인을 만들면 날씬해 보여요",
        "밝은 상의 + 어두운 하의 조합이 허리 라인을 잡아줘요",
        "V넥이나 세로 스트라이프는 시선을 위아래로 분산시켜줘요",
        "너무 헐렁하거나 타이트한 것보다 세미핏이 가장 날씬해 보여요",
    ],
    "키": [
        "상하의 같은 계열 색상으로 세로 선을 만들어주세요",
        "하이웨이스트 바지나 치마로 다리를 길어 보이게 해요",
        "크롭 상의 + 하이웨이스트 하의 조합이 효과적이에요",
        "세로 줄무늬, 긴 목걸이, 롱 가디건도 키가 커 보여요",
    ],
    "복부": [
        "허리 위로 살짝 여유 있는 루즈핏이 배 라인을 자연스럽게 가려줘요",
        "A라인 스커트나 와이드 팬츠로 시선을 아래로 분산시켜요",
        "어두운 계열 컬러가 라인을 정돈해줘요",
        "벨트나 허리 포인트 없는 디자인이 편안해 보여요",
    ],
    "하체": [
        "상의에 포인트를 주고 하의는 어두운 컬러를 선택해요",
        "와이드 팬츠나 A라인으로 허벅지 라인을 자연스럽게 덮어요",
        "상의를 약간 볼륨 있게 해서 상하체 균형을 맞춰요",
        "롱 가디건으로 힙 라인을 가려줄 수 있어요",
    ],
    "상체": [
        "V넥이나 보트넥으로 어깨선을 부드럽게 해줘요",
        "하의에 포인트 컬러를 두어 시선을 아래로 유도해요",
        "세미핏이 상체를 자연스럽게 정돈해줘요",
        "어깨 패드나 퍼프소매, 가로 줄무늬 상의는 피하는 게 좋아요",
    ],
}


def build_color_match_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    """컬러 조합/매치 질문 처리"""
    if not is_color_match_question(user_text):
        return None

    current_colors = parse_color_options(product_context, db_product)
    current_name = clean_text(
        (db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품"
    )
    current_sub = clean_text((db_product or {}).get("sub_category", "") or product_context.get("sub_category", ""))

    # 현재 상품 컬러 기반 매치 안내
    if current_colors:
        base_color = current_colors[0]
        matches = COLOR_MATCH_GUIDE.get(base_color, [])
        if matches:
            match_str = ", ".join(matches[:4])
            pairing = "상의" if current_sub in BOTTOM_SUB_CATS else "하의나 아우터"
            return (
                "{} {} 컬러 기준으로 {}에 잘 어울리는 색상은 {}예요 :)\n"
                "톤온톤(비슷한 계열)으로 맞추거나, 포인트 컬러를 주는 방법 모두 잘 어울려요.\n"
                "신발과 가방도 같은 컬러 계열로 맞추면 전체적으로 세련되게 보여요.".format(
                    current_name, base_color, pairing, match_str)
            )

    # 질문에서 색상 직접 언급 시
    color_words = {
        "흰색": "화이트", "흰": "화이트", "검정": "블랙", "검은색": "블랙",
        "회색": "그레이", "하늘색": "소라", "갈색": "브라운",
    }
    for word, mapped in color_words.items():
        if word in user_text:
            matches = COLOR_MATCH_GUIDE.get(mapped, COLOR_MATCH_GUIDE.get("화이트", []))
            return (
                "{} 계열에 잘 어울리는 색상은 {} 등이에요 :)\n"
                "같은 톤으로 통일하거나, 포인트 컬러 한 가지를 더하면 세련되게 완성돼요.\n"
                "신발이나 가방은 블랙이나 베이지로 맞추면 무난하게 코디돼요.".format(
                    word, ", ".join(matches[:4]))
            )

    for color in COLOR_CANDIDATES:
        if color in user_text:
            matches = COLOR_MATCH_GUIDE.get(color, [])
            if matches:
                return (
                    "{} 컬러에 잘 어울리는 색상은 {}예요 :)\n"
                    "톤온톤 매치나 블랙·화이트 같은 베이직 컬러를 베이스로 맞추면 실패 없어요.".format(
                        color, ", ".join(matches[:4]))
                )

    return (
        "컬러 매치는 블랙·화이트·베이지 같은 베이직을 베이스로 맞추면 실패 없어요 :)\n"
        "포인트 컬러 한 가지만 더하면 세련돼 보여요. 구체적인 색상이 있으면 말씀해주세요!"
    )


def build_body_style_answer(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    """체형 보완 코디 질문 처리"""
    if not is_body_style_question(user_text):
        return None

    q = user_text.replace(" ", "")
    body_hints = extract_user_body_from_text(user_text)
    current_name = clean_text(
        (db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "지금 보시는 상품"
    )

    if any(k in q for k in ["날씬", "슬림", "살빠져보이", "뚱뚱해보이지않", "통통해보이지않"]):
        tips = BODY_STYLE_GUIDE["날씬"][:3]
        intro = "날씬해 보이는 코디"
    elif any(k in q for k in ["키커보이", "다리길어보이", "작아보이지않", "키작", "키가작"]):
        tips = BODY_STYLE_GUIDE["키"][:4]
        intro = "키가 커 보이는 코디"
    elif any(k in q for k in ["복부", "배", "뱃살"]) or body_hints.get("belly_hint"):
        tips = BODY_STYLE_GUIDE["복부"][:3]
        intro = "배 라인 보완 코디"
    elif any(k in q for k in ["하체", "허벅지", "엉덩이"]) or body_hints.get("lower_body_hint"):
        tips = BODY_STYLE_GUIDE["하체"][:3]
        intro = "하체 보완 코디"
    elif any(k in q for k in ["상체", "어깨", "팔뚝"]) or body_hints.get("upper_body_hint") == "상체큰편":
        tips = BODY_STYLE_GUIDE["상체"][:3]
        intro = "상체 보완 코디"
    elif any(k in q for k in ["균형", "작고통통", "키작고통통"]) or (
        body_hints.get("height_hint") == "단신" and body_hints.get("lower_body_hint")
    ):
        tips = BODY_STYLE_GUIDE["키"][:2] + BODY_STYLE_GUIDE["날씬"][:2]
        intro = "균형 있어 보이는 코디"
    else:
        tips = [
            "상하의 컬러를 같은 계열로 맞추면 세로 라인이 생겨 전체적으로 정돈돼 보여요",
            "본인이 불편하게 느끼는 부분은 어두운 색이나 루즈핏으로 자연스럽게 가려주세요",
            "예쁘다고 느끼는 부분은 포인트 컬러나 핏으로 강조하면 좋아요",
        ]
        intro = "체형을 살리는 코디"

    lines = ["{} 팁이에요 :)\n".format(intro)]
    for i, tip in enumerate(tips, 1):
        lines.append("{}. {}".format(i, tip))
    if current_name and current_name != "지금 보시는 상품":
        lines.append("\n지금 보시는 {} 기준으로 구체적인 코디가 궁금하시면 말씀해주세요 :)".format(current_name))

    return "\n".join(lines)



def trim_text(text: str, max_len: int = 500) -> str:
    t = clean_text(text)
    return t if len(t) <= max_len else t[:max_len] + "…"


def slim_context_for_llm(product_context: Dict, db_product: Optional[Dict], user_text: str) -> Dict:
    current = current_product_dict(product_context, db_product)
    colors = parse_color_options(product_context, db_product)
    body = build_body_context()
    situations = detect_situation_from_text(user_text)
    # ★ 핵심: 사이즈 입력 없으면 size_ok 아예 전달 안 함 → LLM이 임의 사이즈 추측 방지
    size_eval: Dict = {}
    top_size = clean_text(body.get("top_size", ""))
    if top_size:
        size_eval = evaluate_size_support(top_size, product_context, db_product)
    db_size_range = clean_text((db_product or {}).get("size_range", ""))
    style_req = detect_style_from_text(user_text)
    body_hints = extract_user_body_from_text(user_text)
    return {
        "current_product_name": current.get("product_name") or "지금 보시는 상품",
        "current_category": current.get("category", ""),
        "current_sub_category": current.get("sub_category", ""),
        "confirmed_colors": colors[:6],
        "size_ok": size_eval.get("supported") if top_size else None,
        "size_reason": trim_text(size_eval.get("reason", ""), 180) if top_size else "",
        "user_has_size_input": bool(top_size),
        "db_size_range": trim_text(db_size_range, 80),
        "body_context": body,
        "body_hints": body_hints,
        "situations": situations,
        "style_request": style_req,
        "page_fit": trim_text(product_context.get("fit", ""), 220),
        "page_material": trim_text(product_context.get("material", ""), 180),
        "page_summary": trim_text(product_context.get("summary", ""), 300),
        "allowed_candidates": [],
        "policy_db": POLICY_DB,
        "color_match_note": "컬러매치는 결정론적 로직이 처리. 없는 색상 절대 언급 금지",
        "body_style_note": "체형코디는 결정론적 로직이 처리. 구체적 상품명은 allowed_candidates만 사용",
    }


def call_llm(user_text: str, product_context: Dict, db_product: Optional[Dict]) -> Optional[str]:
    pack = slim_context_for_llm(product_context, db_product, user_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "참고 데이터(JSON):\n" + json.dumps(pack, ensure_ascii=False)},
    ]
    for m in st.session_state.messages[-4:]:
        messages.append({"role": m["role"], "content": trim_text(m["content"], 280)})
    messages.append({"role": "user", "content": trim_text(user_text, 320)})
    for wait in (0, 1.5):
        if wait:
            time.sleep(wait)
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini", messages=messages, temperature=0.3, max_tokens=280,
            )
            content = clean_text(resp.choices[0].message.content or "")
            if content:
                return content
        except (RateLimitError, APITimeoutError, APIError) as e:
            write_log(event_type="error", error_text="LLM_RateLimit: " + str(e)[:200])
            continue
        except Exception as e:
            write_log(event_type="error", error_text="LLM_Error: " + str(e)[:200])
            break
    return None


def llm_can_help(user_text: str) -> bool:
    """LLM 호출 필요 여부 — 결정론적 로직이 처리하는 건 LLM 불필요"""
    if is_name_question(user_text) or is_color_question(user_text):
        return False
    if get_fast_policy_answer(user_text) is not None:
        return False
    if is_recommendation_question(user_text):
        return False
    if is_color_match_question(user_text):
        return False  # build_color_match_answer에서 처리
    if is_body_style_question(user_text):
        return False  # build_body_style_answer에서 처리
    return True


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
    return "지금 잠깐 답변이 늦어지고 있어요. 같은 내용을 한 번만 다시 보내주시면 바로 이어서 도와드릴게요 :)"


def _determine_response_mode(
    followup_ans, reco_ans, name_ans, policy_ans, size_ans, color_ans, used_llm: bool, used_fallback: bool
) -> tuple:
    """어떤 로직으로 답변했는지 판단 → response_mode, fallback_reason 반환"""
    if followup_ans:
        return "rule_followup", ""
    if reco_ans:
        return "rule_reco", ""
    if name_ans:
        return "rule_name", ""
    if policy_ans:
        return "rule_policy", ""
    if size_ans:
        return "rule_size", ""
    if color_ans:
        return "rule_color", ""
    if used_llm:
        return "llm", ""
    if used_fallback:
        return "fallback", "llm_unavailable"
    return "unknown", ""


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

    current_pno = normalize_product_no((db_product or {}).get("product_no", "") or product_context.get("product_no", ""))
    current_pname = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", ""))
    session_id = st.session_state.get("session_id", "")
    t_start = time.time()

    # USER 메시지 로그
    write_log(
        event_type="user_message",
        product_no=current_pno,
        product_name=current_pname,
        user_text=user_text,
        session_id=session_id,
    )

    try:
        # ★ 우선순위: 추천 질문은 반드시 사이즈 판단보다 먼저 처리
        followup_ans = build_followup_recommendation_answer(user_text)
        # ★ 컬러매치/체형코디는 추천보다 먼저 처리 (코디 관련 키워드 충돌 방지)
        color_match_ans = build_color_match_answer(user_text, product_context, db_product)
        body_style_ans = build_body_style_answer(user_text, product_context, db_product)
        # 컬러매치/체형코디가 아닌 경우에만 추천 처리
        reco_ans = build_recommendation_answer(user_text, product_context, db_product) if (
            is_recommendation_question(user_text) and not color_match_ans and not body_style_ans
        ) else None
        name_ans = build_name_answer(product_context, db_product) if is_name_question(user_text) else None
        policy_ans = get_fast_policy_answer(user_text)
        size_ans = build_size_answer(user_text, product_context, db_product)
        color_ans = build_color_answer(product_context, db_product) if is_color_question(user_text) else None

        direct_answers = [followup_ans, color_match_ans, body_style_ans, reco_ans, name_ans, policy_ans, size_ans, color_ans]
        answer = next((a for a in direct_answers if a), None)

        used_llm = False
        used_fallback = False

        if not answer and llm_can_help(user_text):
            answer = call_llm(user_text, product_context, db_product)
            used_llm = bool(answer)

        if not answer:
            answer = safe_llm_fallback(user_text, product_context, db_product)
            used_fallback = True

        latency_ms = (time.time() - t_start) * 1000
        response_mode, fallback_reason = _determine_response_mode(
            followup_ans, reco_ans, name_ans, policy_ans, size_ans, color_ans, used_llm, used_fallback
        )
        is_fallback = used_fallback or response_mode == "fallback"

        st.session_state.last_answer = answer
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # MIYA 응답 로그 (user_text 페어링 포함)
        write_log(
            event_type="assistant_response",
            product_no=current_pno,
            product_name=current_pname,
            user_text=user_text,
            bot_text=answer,
            response_mode=response_mode,
            fallback_reason=fallback_reason,
            is_fallback=is_fallback,
            latency_ms=latency_ms,
            session_id=session_id,
        )

    except Exception as e:
        latency_ms = (time.time() - t_start) * 1000
        write_log(
            event_type="error",
            product_no=current_pno,
            product_name=current_pname,
            user_text=user_text,
            error_text=str(e)[:300],
            latency_ms=latency_ms,
            session_id=session_id,
        )
        err_answer = "잠깐 오류가 생겼어요. 같은 내용을 다시 보내주시면 바로 이어서 도와드릴게요 :)"
        st.session_state.last_answer = err_answer
        st.session_state.messages.append({"role": "assistant", "content": err_answer})
    finally:
        st.session_state.is_processing = False


# ── URL 파라미터 ──────────────────────────────────────────────────────────────
qp = st.query_params
current_url = clean_text(qp.get("url", "") or "")
product_no_q = normalize_product_no(clean_text(qp.get("pn", "") or ""))
product_name_q = clean_text(qp.get("pname", "") or "")
product_no = product_no_q or extract_product_no_from_url(current_url)

context_key = "{}|{}|{}".format(current_url, product_no, product_name_q)
if context_key != st.session_state.last_context_key:
    st.session_state.last_context_key = context_key
    st.session_state.messages = []
    st.session_state.last_user_hash = ""
    st.session_state.last_answer = ""
    st.session_state.last_recommendations = []

product_context = fetch_product_context(current_url, product_name_q, product_no) if current_url else {
    "product_no": product_no, "product_name": "지금 보시는 상품",
    "category": "기타", "sub_category": "",
    "summary": "", "material": "", "fit": "", "size_tip": "", "raw_excerpt": "", "colors": [],
}
db_product = get_db_product(product_no)
if db_product and clean_text(db_product.get("product_name", "")):
    product_context["product_name"] = clean_text(db_product.get("product_name", ""))
    product_context["category"] = clean_text(db_product.get("category", "")) or product_context.get("category", "")
    product_context["sub_category"] = clean_text(db_product.get("sub_category", "")) or product_context.get("sub_category", "")

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
div[data-testid="stTextInput"] label,div[data-testid="stSelectbox"] label{color:var(--miya-title)!important;font-weight:700!important;font-size:11.5px!important;}
div[data-testid="stTextInput"] input,div[data-baseweb="select"] > div{border-radius:12px!important;}
hr{margin-top:4px!important;margin-bottom:4px!important;border-color:var(--miya-divider)!important;}
div[data-testid="stChatInput"]{position:fixed!important;left:50%!important;transform:translateX(-50%)!important;bottom:68px!important;width:min(720px, calc(100% - 24px))!important;z-index:9999!important;background:transparent!important;}
div[data-testid="stChatInput"] > div{background:transparent!important;border-radius:0!important;padding:0!important;box-shadow:none!important;border:none!important;}
div[data-testid="stChatInput"] textarea {background:#1f2740!important;color:#ffffff!important;caret-color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;font-size:16px!important;line-height:1.35!important;padding-top:12px!important;padding-bottom:12px!important;}
div[data-testid="stChatInput"] textarea::placeholder {color:#cfd6e6!important;opacity:1!important;-webkit-text-fill-color:#cfd6e6!important;}
div[data-testid="stChatInput"] [data-baseweb="textarea"] {background:#1f2740!important;border-radius:999px!important;border:1px solid rgba(255,255,255,0.08)!important;min-height:52px!important;padding:0 10px!important;display:flex!important;align-items:center!important;}
div[data-testid="stChatInput"] [data-baseweb="textarea"] > div {background:transparent!important;display:flex!important;align-items:center!important;}
div[data-testid="stChatInput"] button {background:#2f3a5f!important;color:#ffffff!important;border-radius:14px!important;}
div[data-testid="stChatInput"] button svg {fill:#ffffff!important;}
@media (max-width: 768px){.block-container{max-width:100%;padding-top:0.14rem!important;padding-bottom:11.6rem!important;}div[data-testid="stHorizontalBlock"]{gap:6px!important;}div[data-testid="stHorizontalBlock"] > div{flex:1 1 0!important;min-width:0!important;}div[data-testid="stChatInput"]{bottom:64px!important;width:calc(100% - 16px)!important;}}
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
    is_detail_page = ("/product/detail" in current_url_lower) or ("product_no=" in current_url_lower) or bool(product_no)
    product_display_name = clean_text((db_product or {}).get("product_name", "") or product_context.get("product_name", "") or "")
    if is_detail_page and product_display_name and product_display_name != "지금 보시는 상품":
        welcome = (
            f"안녕하세요 :) {product_display_name} 보고 계시는 거죠?\n"
            "사이즈, 코디, 소재, 배송 중 뭐부터 이야기해볼까요?"
        )
    elif is_detail_page:
        welcome = (
            "안녕하세요? 옷 같이 봐드리는 미야언니예요 :)\n"
            "지금 보시는 상품 기준으로 사이즈, 코디, 배송 뭐든 물어봐주세요."
        )
    else:
        welcome = (
            "안녕하세요? 옷 같이 봐드리는 미야언니예요 :)\n"
            "지금은 일반 상담 모드예요.\n"
            "상품 상세페이지에서 채팅창을 열면 그 상품 기준으로 더 정확하게 상담해드릴 수 있어요 :)\n\n"
            "이 채팅창을 닫고, 궁금하신 상품 상세페이지에서 채팅창을 다시 열어 상담을 진행해주세요 ^^"
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
            f'<div style="padding:10px 14px 10px 10px; border-radius:18px; border-bottom-right-radius:6px; font-size:15px; line-height:1.5; white-space:pre-wrap; word-break:keep-all; background:#dff0ec; color:#1f3b36; border:1px solid rgba(15,106,99,.14);">{safe_text}</div>'
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

user_input = st.chat_input("메시지를 입력하세요...")
if user_input:
    process_user_message(user_input, product_context, db_product)
    st.rerun()
