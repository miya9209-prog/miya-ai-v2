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
    "4050 여성 고객을 옆에서 같이 봐주는 믿음직한 MD처럼 상담해.\n\n"
    "말투 규칙:\n"
    "- 기본 존댓말이지만 친구처럼 자연스럽게\n"
    "- DB에 의하면, 상품정보에 의하면 같은 기계적 표현 금지\n"
    "- 결론부터 말하고 3~5문장 내외로 간결하게\n\n"
    "상담 규칙:\n"
    "1. current_product_name 이름만 사용 (모르면 지금 보시는 상품)\n"
    "2. 추천 상품명은 allowed_candidates에 있는 이름만 사용 - 없는 상품명 절대 금지\n"
    "3. size_ok가 false면 맞다고 절대 하지 말 것\n"
    "4. confirmed_colors 안에 있는 컬러만 언급\n"
    "5. 데이터 없으면 추측하지 말고 솔직하게 말할 것\n"
    "6. 상황(학교 방문, 모임 등)이 언급되면 그 상황에 맞는 스타일 조언 포함\n"
    "7. user_has_size_input이 false이면 사이즈 관련 판단을 절대 하지 말 것 - 사이즈 입력을 먼저 요청할 것\n"
    "8. 인사말(안녕, 안녕하세요 등)에는 사이즈나 상품 정보를 먼저 꺼내지 말고 반갑게 인사 후 무엇을 도와드릴지만 물어볼 것"
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
    """자연어에서 체형 힌트 추출 — 다양한 말투 대응"""
    result = {}
    q = clean_text(user_text)

    # ── 키
    if any(k in q for k in ["키가 작", "키작", "소키", "키 작은", "키작은", "키가 많이", "단신", "작은키", "키가안커"]):
        result["height_hint"] = "단신"
    elif any(k in q for k in ["키가 크", "키크", "키 큰", "키큰", "장신", "큰키"]):
        result["height_hint"] = "장신"

    # ── 상체/가슴 (다양한 표현)
    upper_big_kws = [
        "상체가 크", "상체크", "상체 있는", "상체있는", "상체가 있",
        "어깨넓", "어깨가 넓", "어깨가넓",
        "가슴크", "가슴이 좀", "가슴이좀", "가슴이 있", "가슴이있", "가슴이 크",
        "가슴이 커", "가슴이커", "가슴 있는", "가슴있는",
        "상체비만", "상체가 발달", "상체발달",
        "어깨가 커", "어깨가커",
    ]
    upper_small_kws = ["상체가 작", "상체작", "어깨좁", "어깨가 좁", "어깨가좁", "상체가 좁"]
    if any(k in q for k in upper_big_kws):
        result["upper_body_hint"] = "상체큰편"
    elif any(k in q for k in upper_small_kws):
        result["upper_body_hint"] = "상체작은편"

    # ── 하체
    lower_big_kws = [
        "하체가 크", "하체크", "허벅지", "하체통통", "하체가 통통",
        "허벅지가", "엉덩이가 크", "엉덩이가크", "골반이 넓", "골반넓",
        "하체가 있", "하체있는", "하체비만", "다리가 굵",
    ]
    if any(k in q for k in lower_big_kws):
        result["lower_body_hint"] = "하체통통"

    # ── 복부
    belly_kws = ["배가 나", "복부", "뱃살", "배가나", "배살", "배가 있", "배가있", "배나온", "배가 좀"]
    if any(k in q for k in belly_kws):
        result["belly_hint"] = "복부커버필요"

    return result


def detect_size_from_text(user_text: str) -> Optional[str]:
    for token in ["55반", "66반", "77반", "44", "55", "66", "77", "88", "99"]:
        if token in user_text:
            return token
    return None


def detect_situation_from_text(user_text: str) -> List[str]:
    """다양한 말투의 상황 키워드 감지"""
    found = []
    # 기본 키워드 매칭
    for kw in SITUATION_KEYWORDS:
        if kw in user_text:
            found.append(kw)
    # 추가 패턴
    extra_map = {
        "학교": ["선생님 만나", "학부모", "입학식", "졸업식", "학교 행사", "학교에"],
        "출근": ["직장", "회사", "사무실", "오피스", "업무", "미팅", "비즈니스"],
        "모임": ["하객", "결혼식", "돌잔치", "동창회", "동문회", "동기", "선후배", "격식"],
        "여행": ["여행", "나들이", "나들이", "캠핑", "피크닉"],
        "데이트": ["남자친구", "남편", "소개팅", "썸"],
        "친구": ["친구들", "친구만나", "친구랑"],
    }
    for situation, patterns in extra_map.items():
        if situation not in found:
            if any(p in user_text for p in patterns):
                found.append(situation)
    return found


def is_size_question(user_text: str) -> bool:
    """다양한 말투의 사이즈 질문 감지 — 추천 요청은 추천이 우선"""
    if is_recommendation_question(user_text):
        return False
    q = user_text.replace(" ", "")

    # 직접적 사이즈/핏 키워드
    size_kws = [
        "사이즈", "맞을까", "맞을까요", "맞아", "맞아요", "맞나요", "맞나",
        "핏", "작을까", "작을까요", "클까", "클까요",
        "여유", "여유있게", "여유있어", "여유있나요", "여유로워",
        "타이트", "빡빡", "빡빡할", "빡빡해요",
        "내사이즈", "나한테맞", "나에게맞", "제사이즈",
        "입을수있", "살수있", "나올까", "나오나요", "나오는지",
        "안맞겠지", "안맞을까", "안맞나요", "안맞아",
        "될까", "될까요", "되나요", "돼요", "돼",
        "가능해요", "가능한가요", "가능할까", "가능한지",
        "살수있어", "살수있나요",
        "입어도될", "입어도돼", "입어도되나",
        "사이즈있어", "사이즈있나", "사이즈나와",
    ]
    if any(k in q for k in size_kws):
        return True

    # 사이즈 숫자 포함
    if detect_size_from_text(user_text):
        return True

    # 자연어 패턴
    patterns = [
        r"이\s*[옷거상품].{0,15}(나한테|나에게|내가|제가|맞|어때|어울|될까|가능|살|입)",
        r"(나한테|나에게|내가|제가|저한테).{0,15}(맞|될|가능|입을|돼|살)",
        r"(내|제)\s*(사이즈|몸|체형|키|몸무게).{0,10}(맞|될|가능|있|나)",
        r"(이거|이옷|이상품|그거|그옷).{0,10}(될까|맞|가능|살|입|어때)",
        r"(큰\s*사이즈|빅\s*사이즈|라지).{0,10}(있|나와|있나|있어)",
        r"(내\s*몸|제\s*몸).{0,10}(맞|될|가능)",
    ]
    for pat in patterns:
        if re.search(pat, user_text):
            return True

    # 체형 힌트 + 사이즈 의도
    body_hints = extract_user_body_from_text(user_text)
    if body_hints and any(k in q for k in ["맞", "어때", "어울", "될까", "입을", "안맞", "역시", "가능", "살"]):
        return True

    return False
def is_name_question(user_text: str) -> bool:
    q = user_text.replace(" ", "")
    return any(k in q for k in ["이옷이름", "상품명", "상품이름", "이름뭐", "이옷이뭐야", "품명"])


def is_color_question(user_text: str) -> bool:
    return any(k in user_text for k in ["컬러", "색상", "무슨 색", "어떤 색"] + COLOR_CANDIDATES)


def is_recommendation_question(user_text: str) -> bool:
    """다양한 말투의 추천/코디 요청 감지"""
    q = user_text.replace(" ", "")

    # ── 명시적 추천/코디 키워드
    reco_kws = [
        "추천", "추천해", "추천해줘", "추천해주세요", "추천좀", "추천부탁",
        "골라줘", "골라줘요", "골라주세요", "골라봐줘",
        "어울리는", "어울릴", "어울려", "어울리게",
        "같이입", "같이입을", "함께입",
        "코디", "코디해줘", "코디도와줘", "코디부탁",
        "매치", "매치해줘",
        "비슷한옷", "비슷한상품", "비슷한거",
        "다른옷", "다른상품", "다른거", "다른걸로",
        "다른자켓", "다른아우터", "다른바지", "다른점퍼", "다른맨투맨",
        "다른셔츠", "다른블라우스", "다른가디건", "다른니트", "다른스커트",
        "없어요", "없나요", "없어", "없을까",
        "보여줘", "보여주세요", "봐줘", "봐주세요",
        "뭐입", "뭘입", "무엇을입",
    ]
    if any(k in q for k in reco_kws):
        return True

    # ── 상황 기반 추천 패턴
    situation_patterns = [
        r"(학교|출근|모임|행사|파티|결혼식|데이트|친구|여행|산책|소풍).{0,20}(입|코디|추천|뭐|어떤|골라)",
        r"(입고\s*갈|입을\s*만한|걸칠\s*만한|걸칠|입어야).{0,15}(거|것|옷|상품)",
        r"(뭐\s*입|어떻게\s*입|어떤\s*옷|어떤\s*거).{0,10}(좋|나을|어울|될|살)",
        r"(더\s*큰|빅|라지|큰\s*사이즈).{0,10}(없|있|나와|찾)",
        r"(88|99|77반).{0,10}(되는|나오는|있는).{0,10}(자켓|아우터|옷|상품|점퍼|바지)",
    ]
    for pat in situation_patterns:
        if re.search(pat, user_text):
            return True

    # 나한테 맞는 X 있어? 패턴
    match_patterns = [
        r"(나한테|나에게|내가|제가).{0,10}(맞는|어울리는|맞을).{0,15}(있|없|찾|줘|추천)",
        r"(맞는|어울리는|어울릴).{0,10}(자켓|아우터|옷|바지|상품|점퍼|맨투맨|셔츠).{0,10}(있|없|찾|줘)",
        r"(내|제)\s*(사이즈|몸).{0,10}(맞는|되는|가능한).{0,10}(있|없|찾)",
    ]
    for pat in match_patterns:
        if re.search(pat, user_text):
            return True

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
    if any(k in q for k in ["출고", "당일출고", "언제와", "배송언제", "며칠"]):
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
    # 현재 상품 기준으로 반대편 추천
    current_sub = clean_text(current_product.get("sub_category", ""))
    current_cat = clean_text(current_product.get("category", ""))
    if current_sub in BOTTOM_SUB_CATS:
        return "블라우스"
    if current_cat in TOP_MAIN_CATS or current_sub in TOP_SUB_CATS:
        return "팬츠"
    return ""


def build_product_reason(rowd: Dict, user_text: str, body_hints: Dict) -> List[str]:
    reasons: List[str] = []
    blob = row_blob(rowd)
    name = clean_text(rowd.get("product_name", ""))
    situations = detect_situation_from_text(user_text)
    if any(s in situations for s in ["학교", "방문"]):
        reasons.append("단정한 자리에 어울리는 분위기예요" if not any(k in blob for k in ["단정", "클래식", "슬랙스"]) else "학교 방문룩으로 단정하게 받쳐주기 좋아요")
    elif "출근" in situations:
        if any(k in blob for k in ["단정", "클래식", "슬랙스"]):
            reasons.append("출근룩으로 깔끔하게 이어주기 좋아요")
    elif any(s in situations for s in ["모임", "행사"]):
        reasons.append("모임 자리에 자연스럽게 어울려요")
    elif "데이트" in situations:
        reasons.append("데이트 코디로 자연스럽게 어울려요")
    cover = clean_text(rowd.get("body_cover_features", ""))
    if body_hints.get("belly_hint") and any(k in cover for k in ["복부", "뱃살"]):
        reasons.append("복부 커버에 도움이 되는 편이에요")
    elif body_hints.get("lower_body_hint") and any(k in cover for k in ["힙", "허벅지"]):
        reasons.append("하체 라인 부담이 적은 편이에요")
    if not reasons:
        if "슬랙스" in name:
            reasons.append("라인이 정돈돼 보여서 상의를 깔끔하게 살려줘요")
        elif any(k in name for k in ["데님", "청바지"]):
            reasons.append("너무 힘주지 않은 분위기로 편하게 매치하기 좋아요")
        elif any(k in name for k in ["자켓", "재킷"]):
            reasons.append("전체 실루엣이 정돈되어 보이는 편이에요")
        else:
            reasons.append("지금 보시는 옷이랑 코디가 자연스럽게 이어져요")
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
    for idx, words in {0: ["1번", "첫번째", "첫째"], 1: ["2번", "두번째", "둘째"], 2: ["3번", "세번째", "셋째"]}.items():
        if any(w in q for w in words):
            return idx
    if any(w in q for w in ["방금추천", "추천해준", "그거", "그상품", "그옷", "그바지", "그자켓"]):
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
    if body_hints.get("upper_body_hint") == "상체큰편" and size_cat == "top":
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


def build_followup_recommendation_answer(user_text: str) -> Optional[str]:
    reco = get_followup_recommendation(user_text)
    if not reco:
        return None
    reco_context, reco_db = recommendation_to_context(reco)
    idx = (get_recommendation_reference_index(user_text) or 0) + 1
    name = reco_context["product_name"]

    if is_name_question(user_text) or any(k in user_text for k in ["어떤 옷", "어떤 바지", "설명", "알려줘", "뭐야"]):
        reasons = [clean_text(x) for x in (reco.get("reasons") or []) if clean_text(x)]
        reason_line = " ".join(reasons[:2]) if reasons else "고객님 스타일에 맞게 골라드린 상품이에요."
        return "{}번으로 추천드린 상품은 {}이에요 :)\n{}\n사이즈나 코디가 궁금하시면 말씀해주세요.".format(idx, name, reason_line)

    if is_size_question(user_text):
        body = build_body_context()
        size_cat = get_product_size_category(reco_db, reco_context)
        user_size = (clean_text(body.get("bottom_size", "")) if size_cat == "bottom"
                     else clean_text(body.get("top_size", ""))) or detect_size_from_text(user_text) or ""
        if not user_size:
            size_label = "하의" if size_cat == "bottom" else "상의"
            return "{}번으로 추천드린 {} 기준으로 사이즈 보려면 고객님 {} 사이즈를 먼저 알려주세요 :)".format(idx, name, size_label)
        size_eval = evaluate_size_support(user_size, reco_context, reco_db)
        reason = clean_text(size_eval.get("reason", ""))
        max_size = size_eval.get("max_size", "")
        if size_eval["supported"] is False:
            return "{}번으로 추천드린 {}은 고객님 {} 기준으로는 맞는 사이즈가 없어요.\n{}\n다른 상품으로 다시 찾아드릴까요?".format(
                idx, name, user_size, reason or "최대 {}까지 나오는 상품이에요.".format(max_size))
        if size_eval["supported"] == "edge":
            return "{}번으로 추천드린 {}은 고객님 {} 기준이면 가능은 하지만 딱 경계 사이즈예요.\n{}\n편하게 입는 걸 좋아하시면 실측표를 같이 보시는 게 안전해요.".format(
                idx, name, user_size, reason or "상단 사이즈에 가까워서 체감이 약간 타이트할 수 있어요.")
        if size_eval["supported"] is True:
            return "{}번으로 추천드린 {}은 고객님 {} 기준으로 사이즈 범위 안에 들어와요 :)\n{}\n부담 없이 입는 쪽으로 비교적 안정적인 편이에요.".format(
                idx, name, user_size, reason)
        db_range = clean_text((reco_db or {}).get("size_range", ""))
        return "{}번으로 추천드린 {}은 현재 {}쪽이에요 :)\n정확한 핏은 실측표를 같이 보시는 쪽이 제일 안전해요.".format(
            idx, name, db_range or "사이즈 정보 확인 필요")

    if is_color_question(user_text):
        ans = build_color_answer(reco_context, reco_db)
        return "{}번으로 추천드린 {} 기준으로 보면, {}".format(idx, name, ans) if ans else "{}번으로 추천드린 {}은 현재 컬러 정보가 정확히 확인되지 않아요.".format(idx, name)
    return None


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
    return {
        "current_product_name": current.get("product_name") or "지금 보시는 상품",
        "current_category": current.get("category", ""),
        "current_sub_category": current.get("sub_category", ""),
        "confirmed_colors": colors[:6],
        # 사이즈 정보: 입력 없으면 null 명시 → LLM이 추측하지 않도록
        "size_ok": size_eval.get("supported") if top_size else None,
        "size_reason": trim_text(size_eval.get("reason", ""), 180) if top_size else "",
        "user_has_size_input": bool(top_size),
        "db_size_range": trim_text(db_size_range, 80),
        "body_context": body,
        "situations": situations,
        "page_fit": trim_text(product_context.get("fit", ""), 220),
        "page_material": trim_text(product_context.get("material", ""), 180),
        "allowed_candidates": [],
        "policy_db": POLICY_DB,
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
    # 사이즈/이름/컬러/정책은 결정론적 로직이 처리하므로 LLM 불필요
    if is_name_question(user_text) or is_color_question(user_text):
        return False
    if get_fast_policy_answer(user_text) is not None:
        return False
    # 사이즈 질문은 결정론적 처리 후에도 LLM이 보완할 수 있도록 허용
    # (단, 추천 질문은 이미 reco_ans로 처리되므로 중복 LLM 호출 방지)
    if is_recommendation_question(user_text):
        return False  # 추천은 build_recommendation_answer에서 이미 처리
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
        reco_ans = build_recommendation_answer(user_text, product_context, db_product) if is_recommendation_question(user_text) else None
        name_ans = build_name_answer(product_context, db_product) if is_name_question(user_text) else None
        policy_ans = get_fast_policy_answer(user_text)
        size_ans = build_size_answer(user_text, product_context, db_product)
        color_ans = build_color_answer(product_context, db_product) if is_color_question(user_text) else None

        direct_answers = [followup_ans, reco_ans, name_ans, policy_ans, size_ans, color_ans]
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
            "상품 상세페이지에서 채팅창을 열면 그 상품 기준으로 더 정확하게 상담해드릴 수 있어요 :)"
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
