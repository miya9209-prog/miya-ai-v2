import base64
import io
import json
import mimetypes
import re
import time
from typing import Any, Dict, List

import streamlit as st
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt
from openai import OpenAI, RateLimitError

st.set_page_config(page_title="PAGE BUILDER", layout="wide")

# -----------------------------
# Session state
# -----------------------------
if "reset_nonce" not in st.session_state:
    st.session_state.reset_nonce = 0
if "naming_result" not in st.session_state:
    st.session_state.naming_result = ""
if "naming_input_value" not in st.session_state:
    st.session_state.naming_input_value = ""
if "generated_result" not in st.session_state:
    st.session_state.generated_result = ""
if "generated_docx" not in st.session_state:
    st.session_state.generated_docx = b""
if "generated_file_stem" not in st.session_state:
    st.session_state.generated_file_stem = "page_builder"

st.markdown(
    """
<style>
div[data-testid="stButton"] > button { min-height: 42px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("MISHARP PAGE BUILDER")
st.caption("구매전환율 상승을 위한 상세페이지 기획 + 상품 원고 생성기")

api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.warning("OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Cloud Secrets 또는 .streamlit/secrets.toml을 확인해 주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

FIXED_HTML_HEAD = """<meta http-equiv=\"X-UA-Compatible\" content=\"IE=edge,chrome=1\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<link href=\"http://fonts.googleapis.com/css?family=Roboto\" rel=\"stylesheet\" type=\"text/css\">
<link href=\"http://netdna.bootstrapcdn.com/font-awesome/4.3.0/css/font-awesome.min.css\" rel=\"stylesheet\" type=\"text/css\">
<link href=\"/SRC2/cssmtmenu/style.css\" rel=\"stylesheet\" type=\"text/css\">
<link href=\"//spoqa.github.io/spoqa-han-sans/css/SpoqaHanSans-kr.css\" rel=\"stylesheet\" type=\"text/css\">
<link href=\"//misharp.co.kr/subtap.css\" rel=\"stylesheet\" type=\"text/css\">"""

NAME_PROMPT = """
너는 4050 여성 패션 쇼핑몰 미샵의 상품 네이밍 전문가다.
- 상품 주요 특징을 반영해 상품명을 20개 제안한다.
- 각 상품명은 공백 포함 최대 18자 이내.
- 반드시 단어와 단어 사이를 자연스럽게 띄어쓴다.
- AI 검색, 키워드 검색 모두 고려한다.
- 디테일/형태/원단/핏 등을 반영한 단어 + 카테고리명을 포함한다.
- 필요하면 세련되고 여성스러운 단어를 앞에 붙여도 된다.
- 번호, 설명, 코드펜스 없이 한 줄에 하나씩 20개만 출력한다.
"""

# -----------------------------
# Helpers
# -----------------------------
def chat_with_retry(*, model: str, messages: List[Dict[str, Any]], temperature: float = 0.3, max_retries: int = 2):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        except RateLimitError as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            time.sleep(1 * (attempt + 1))
    raise last_exc


def file_to_content_item(uploaded_file):
    mime = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0] or "image/jpeg"
    data = uploaded_file.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def extract_lines_with_digits(text: str) -> List[str]:
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line and re.search(r"\d", line):
            out.append(line)
    return out


def combine_measurements(top_text: str, bottom_text: str, dress_text: str) -> List[str]:
    lines = []
    for block in [top_text, bottom_text, dress_text]:
        lines.extend(extract_lines_with_digits(block))
    return lines


def count_colors(color_text: str) -> int:
    if not color_text.strip():
        return 0
    text = color_text.replace(" / ", "\n").replace("/", "\n").replace(",", "\n")
    parts = [re.sub(r"^\s*\d+\s*", "", p).strip() for p in text.splitlines()]
    return len([p for p in parts if p])


def apply_color_count_to_name(product_name: str, color_text: str) -> str:
    count = count_colors(color_text)
    suffix = f"({count} color)" if count > 0 else "(color)"
    name = (product_name or "").strip()
    name = re.sub(r"\(\s*color\s*\)", suffix, name, flags=re.I)
    if re.search(r"\(\s*\d+\s*color\s*\)", name, flags=re.I):
        return name
    if "( color)" in name:
        return name.replace("( color)", f" {suffix}")
    return name


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def clean_line(line: str) -> str:
    line = (line or "").strip()
    line = re.sub(r"^[\-•▪⦁\s]+", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()




def _natural_wrap(text: str, min_len: int, max_len: int) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    words = text.split(' ')
    lines = []
    cur = ''
    for w in words:
        cand = f"{cur} {w}".strip()
        if len(cand) <= max_len:
            cur = cand
            continue
        if cur:
            # if current is too short and next word can fit with slight overflow, keep together once
            if len(cur) < min_len and len(cand) <= max_len + 4:
                lines.append(cand)
                cur = ''
            else:
                lines.append(cur)
                cur = w
        else:
            lines.append(w)
            cur = ''
    if cur:
        lines.append(cur)
    return lines


def wrap_text_source(text: str) -> str:
    return '<br>\n'.join(_natural_wrap(text, 18, 30))
def wrap_md(text: str) -> str:
    return '<br>\n'.join(_natural_wrap(text, 16, 24))
def to_noun(line: str) -> str:
    s = re.sub(r'\s+', ' ', (line or '').strip())
    s = s.rstrip('. ')
    replacements = [
        ('추천드립니다', ''), ('추천합니다', ''), ('권해드립니다', ''), ('권합니다', ''),
        ('만족을 드립니다', ''), ('만족을 드릴 선택입니다', ''), ('이상적입니다', ''),
        ('잘 어울립니다', ''), ('알맞습니다', ''), ('적합합니다', ''), ('좋습니다', ''),
        ('권해드려요', ''), ('추천드려요', ''),
    ]
    for a,b in replacements:
        s = s.replace(a,b)
    s = s.replace('원하시는 분께도', '원하시는 분')
    s = s.replace('원하시는 분께', '원하시는 분')
    s = s.replace('찾으시는 분께', '찾으시는 분')
    s = s.replace('고객님께', '분')
    s = s.replace('고객님', '분')
    s = s.replace('분께도', '분')
    s = s.replace('분께', '분')
    s = s.replace('께도', '')
    s = s.replace('께', '')
    s = re.sub(r'\s+', ' ', s).strip(' .')
    if not s.endswith('분'):
        if s.endswith('원하시는'):
            s += ' 분'
        elif s.endswith('좋아하시는'):
            s += ' 분'
        elif s.endswith('찾는'):
            s += ' 분'
        else:
            s += ' 분'
    return s


def clean_md(text: str) -> str:
    text = re.sub(r'<strong style="font-weight:700 !important;">\[구매 전 꼭 확인해 주세요\]</strong><br>\n(?:.*?<br>\n)+?<br>\n', '', text, flags=re.S)
    return text
def ensure_sentence(line: str, ending: str = ".") -> str:
    line = clean_line(line)
    if not line:
        return ""
    if re.search(r"[.!?…]$", line):
        return line
    return line + ending


def normalize_list(items: List[str], count: int, quote: bool = False) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        item = clean_line(item)
        item = item.replace('"', "").strip()
        if not item:
            continue
        key = re.sub(r"\s+", "", item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= count:
            break
    if quote:
        out = [f'"{x}"' for x in out]
    return out


def normalize_multiline_text(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace("\r", "\n")
    candidates = []
    for part in text.split("\n"):
        part = clean_line(part)
        if part:
            candidates.append(part)
    return candidates


def format_measurement_lines(lines: List[str]) -> str:
    if not lines:
        return "실측사이즈 정보를 입력해 주세요."
    formatted = []
    for line in lines:
        line = re.sub(r"\s+단위:cm$", "", line).strip()
        line = re.sub(r"\s+L\s+", "<br>L ", line)
        line = re.sub(r"\s+M\s+", "<br>M ", line)
        line = re.sub(r"\s+S\s+", "<br>S ", line)
        line = re.sub(r"\s+XL\s+", "<br>XL ", line)
        line = re.sub(r"\s+", " ", line)
        formatted.append(line)
    return "<br>".join(formatted) + " (단위: cm)"


def result_to_docx_bytes(result_text: str) -> bytes:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Dotum"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "돋움")
    style.font.size = Pt(10)
    style.paragraph_format.space_before = Pt(0)
    style.paragraph_format.space_after = Pt(0)
    style.paragraph_format.line_spacing = 1.5

    for line in result_text.splitlines():
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.5
        run = p.add_run(line)
        run.font.name = "Dotum"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "돋움")
        run.font.size = Pt(10)

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()


def reset_all():
    st.session_state.reset_nonce += 1
    st.session_state.naming_result = ""
    st.session_state.naming_input_value = ""
    st.session_state.generated_result = ""
    st.session_state.generated_docx = b""
    st.session_state.generated_file_stem = "page_builder"


def build_generation_prompt(data: Dict[str, str], additional_request: str) -> str:
    sample_format = """
출력폼 구조 핵심:
- 상단 헤더: 상품명 / 컬러 / 사이즈 / 소재 / 소재설명 / 제조국
- 포인트 원고(포토샵 작업): 현재는 비워 둔다.
- 텍스트 소스: 추천 4줄 / 착용후기 4줄 / FAQ 4개 / 쇼핑참고 3줄
- MD원고: [이 상품을 초이스한 이유입니다.] / [원단과 두께 체감에 대하여] / [체형과 핏, 사이즈 선택 가이드] / [이렇게 입는 날이 많아집니다] / [구매 전 꼭 확인해 주세요] + 마무리 3줄
- 사이즈 팁: 55 / 66 / 66반 / 77 각 1문장
"""
    return f"""
너는 10년 이상 4050 여성 패션몰 미샵 상세페이지를 써 온 최고 수준의 한국어 커머스 카피라이터다.
단순 요약이 아니라, 실제 고객이 망설이는 이유를 먼저 해소하고 구매를 돕는 고급 원고를 작성한다.
SEO/AEO/GEO를 고려해 상품명, 카테고리, 소재, 핏, 활용 장면, 고객 니즈를 자연스럽게 녹인다.

중요 원칙
1) 입력 문구를 그대로 복붙하지 말고 반드시 자연스럽고 매력적인 문장으로 재작성한다.
2) 없는 디테일은 절대 만들지 않는다. 타이/스카프/포켓/단추/절개/비침/스트랩 등은 입력 또는 이미지 근거가 있을 때만 언급한다.
3) FAQ는 고객 입장에서 실제로 물을 질문으로 쓴다. 단순한 "77까지 맞나요" 같은 기계적 질문 금지.
4) 추천/후기/MD원고/사이즈팁 모두 TPO, 고객 pain point, 착용 장면, 체형 고민을 반영한다.
5) 문장은 너무 짧게 끊지 말고, 한 줄 단위로 보기 좋게 정리한다.
6) 결과는 반드시 JSON만 출력한다. 코드펜스 금지.

{sample_format}

입력 데이터
- 상품명: {data['display_name']}
- 컬러: {data['color']}
- 사이즈: {data['size']}
- 소재: {data['material']}
- 소재설명 참고메모: {data['material_desc_raw']}
- 디테일 특징: {data['detail_tip']}
- 핏/실루엣: {data['fit']}
- 주요 어필 포인트: {data['appeal_points']}
- 기타 특징: {data['etc']}
- 타겟: {data['target']}
- 세탁방법: {data['washing']}
- 실측사이즈: {' / '.join(data['measurement_lines'])}
- 추가/수정 요청사항: {additional_request or '없음'}

JSON 스키마
{{
  "material_desc_lines": ["문장", "문장", "문장"],
  "recommend_lines": ["문장", "문장", "문장", "문장"],
  "review_lines": ["문장", "문장", "문장", "문장"],
  "faqs": [
    {{"q": "질문", "a": "답변"}},
    {{"q": "질문", "a": "답변"}},
    {{"q": "질문", "a": "답변"}},
    {{"q": "질문", "a": "답변"}}
  ],
  "shopping_lines": ["문장", "문장", "문장"],
  "md_sections": {{
    "choice": ["문장", "문장", "문장", "문장"],
    "fabric": ["문장", "문장", "문장", "문장"],
    "fit": ["문장", "문장", "문장", "문장"],
    "occasion": ["문장", "문장", "문장", "문장"],
    "purchase_note": ["문장", "문장", "문장", "문장"],
    "ending": ["문장", "문장", "문장"]
  }},
  "size_tips": {{
    "55": "한 문장",
    "66": "한 문장",
    "66half": "한 문장",
    "77": "한 문장"
  }}
}}

작성 기준
- material_desc_lines: 3줄 권장. 소재 장점과 표면감, 계절감, 관리 포인트를 자연스럽게.
- recommend_lines: 고객 유형 중심으로 4줄.
- review_lines: 실제 미샵 스탭이 말할 법한 자연스러운 후기로 4줄.
- faqs: 각 질문은 구체적으로. 예) 가슴이 있는 77 체형, 밝은 컬러 비침, 하루 종일 구김, TPO 활용, 관리법 등.
- shopping_lines: 꼭 확인해야 할 실용 정보 3줄.
- md_sections.choice: 왜 이 상품을 골라야 하는지 설득형.
- md_sections.fabric: 소재와 두께감을 설명형으로.
- md_sections.fit: 체형, 실루엣, 사이즈 선택에 도움 되는 내용.
- md_sections.occasion: 언제 어떻게 입는지 장면 중심으로.
- md_sections.purchase_note: 비침/세탁/디테일/사이즈 확인 등 구매 전 체크사항.
- md_sections.ending: 감성 과장 없이 미샵 톤으로 3줄.
- size_tips: 각 체형별로 실제 입었을 때 느낌을 한 문장으로.
"""


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("JSON 응답을 찾지 못했습니다.")
    return json.loads(m.group(0))


def smart_faq_defaults(data: Dict[str, str]) -> List[Dict[str, str]]:
    size = clean_line(data.get("size") or "FREE 사이즈로 77까지 추천드립니다.")
    product = data.get("display_name") or "상품"
    color = data.get("color") or ""
    detail = normalize_space(data.get("detail_tip") or "")
    fit = normalize_space(data.get("fit") or "")
    material_desc = normalize_space(data.get("material_desc_raw") or "")
    bright = "아이보리" in color or "화이트" in color or "밝" in material_desc
    has_detachable = bool(re.search(r"탈부착|분리", detail))
    faq = [
        {"q": "Q. 가슴이 있는 77 체형인데 답답하지 않게 입을 수 있을까요?", "a": f"A. {size} 기준으로 여유 있게 착용하실 수 있도록 안내드리며, 실측사이즈를 함께 보시면 더 정확한 선택에 도움이 됩니다."},
        {"q": "Q. 하루 종일 입으면 구김이 심하게 남는 편인가요?", "a": "A. 소재 특성상 비교적 깔끔한 인상을 유지하기 좋은 편이라 출근룩이나 모임룩으로도 부담이 적습니다."},
        {"q": "Q. 밝은 컬러라 비침이 많이 느껴질까요?", "a": "A. 밝은 컬러 계열은 약간의 비침이 있을 수 있어 스킨톤 이너와 함께 착용하시면 훨씬 안정감 있게 입으실 수 있습니다." if bright else "A. 컬러와 소재 특성상 과한 부담 없이 입기 좋지만, 밝은 이너보다 톤을 맞춰 주시면 더 깔끔하게 연출됩니다."},
        {"q": "Q. 데일리로 입기 쉬운 스타일인가요, 아니면 격식 있는 자리에 더 잘 어울릴까요?", "a": f"A. {product}은 데님, 슬랙스, 스커트 등과 두루 잘 어울려 데일리부터 오피스룩, 모임룩까지 자연스럽게 활용하시기 좋습니다."},
    ]
    if has_detachable:
        faq[3] = {"q": "Q. 디테일이 과해 보이지 않을까요? 탈부착 연출도 가능한가요?", "a": "A. 포인트는 은은하게 살아 있으면서도 전체 분위기는 정돈되어 보여 부담이 적고, 디테일 연출 폭도 넓어 활용도가 높습니다."}
    if fit:
        faq[0]["a"] = f"A. {size} 기준으로 안내드리며, {fit} 특성이 있어 체형을 비교적 편안하게 감싸주는 편입니다. 실측사이즈를 함께 보시면 더 정확합니다."
    return faq


def fallback_structured(data: Dict[str, str]) -> Dict[str, Any]:
    product = data.get("display_name") or "상품"
    size = clean_line(data.get("size") or "FREE 사이즈로 77까지 추천드립니다.")
    material_desc_raw = data.get("material_desc_raw") or ""
    material_lines = normalize_multiline_text(material_desc_raw)[:3]
    if not material_lines:
        material_lines = [
            "부드러운 터치감으로 피부에 닿는 느낌이 편안한 소재입니다.",
            "표면감이 차분하게 정리되어 데일리로 활용하기 좋습니다.",
            "관리 부담이 크지 않아 손이 자주 가는 아이템입니다.",
        ]
    else:
        material_lines = [ensure_sentence(x) for x in material_lines]

    fit_text = normalize_space(data.get("fit") or "")
    detail_text = normalize_space(data.get("detail_tip") or "")
    appeal_text = normalize_space(data.get("appeal_points") or "")
    etc_text = normalize_space(data.get("etc") or "")

    recommend = [
        "출근룩부터 모임룩까지 단정하게 입을 아이템을 찾으시는 분",
        "상체 라인을 부담 없이 정리해 주는 편안한 핏을 선호하시는 분",
        "소재감이 주는 고급스러운 분위기를 중요하게 보시는 분",
        "데님, 슬랙스, 스커트와 두루 잘 어울리는 상의를 원하시는 분",
    ]
    reviews = [
        "입었을 때 전체 실루엣이 차분하게 정리돼서 손이 자주 가요.",
        "촉감이 부담스럽지 않아 하루 종일 입어도 편안한 느낌이에요.",
        "격식 있는 자리에도 과하지 않게 잘 어울려 활용도가 높아요.",
        "구김 부담이 크지 않아 바쁜 날에도 깔끔하게 입기 좋았어요.",
    ]
    shopping = [
        f"{size}",
        "밝은 컬러 계열은 스킨톤 이너와 함께 착용하시면 더 안정감 있게 연출하실 수 있습니다.",
        "실측사이즈를 함께 확인하시면 원하는 핏으로 선택하시기 더 좋습니다.",
    ]
    if detail_text:
        reviews[2] = f"{detail_text} 포인트가 과하지 않게 살아 있어 단독으로 입어도 충분히 멋스러워요."
    if fit_text:
        recommend[1] = f"{fit_text}처럼 체형을 자연스럽게 감싸주는 실루엣을 선호하시는 분"
    if appeal_text:
        recommend[2] = f"{appeal_text} 같은 실용 포인트를 중요하게 보시는 분"

    md_choice = [
        f"{product}은 과하게 꾸민 느낌 없이도 차분하고 세련된 인상을 만들어 줍니다.",
        "기본에 가까운 디자인일수록 소재와 핏의 차이가 크게 드러나는데, 이 아이템은 그 균형감이 특히 좋습니다.",
        fit_text + " 장점이 자연스럽게 살아 있어 부담 없이 손이 갑니다." if fit_text else "체형을 부드럽게 감싸주는 실루엣이 안정감 있게 느껴집니다.",
        etc_text + " 장점까지 더해져 소장 만족도를 높여줍니다." if etc_text else "데일리와 격식을 오가는 장면에서 활용도가 높아 추천드리기 좋습니다.",
    ]
    md_fabric = material_lines + ["두께감 또한 과하게 무겁지 않아 계절감에 맞춰 손쉽게 매치하시기 좋습니다."]
    md_fit = [
        f"{size} 기준으로 안내드리며, 전체적으로 답답하지 않게 착용하시기 좋은 편입니다.",
        fit_text + " 장점이 자연스럽게 드러나 체형 고민을 덜어줍니다." if fit_text else "어깨선과 품이 과하게 붙지 않아 상체 라인이 비교적 편안해 보입니다.",
        detail_text + " 요소가 있다면 전체 라인을 한층 더 정돈돼 보이게 도와줍니다." if detail_text else "실루엣이 과하게 흐트러지지 않아 단정한 분위기를 유지하기 좋습니다.",
        "실측사이즈를 함께 확인하시면 원하는 핏으로 선택하시기 더 수월합니다.",
    ]
    md_occasion = [
        "출근룩처럼 단정함이 필요한 날에도 무리 없이 입기 좋습니다.",
        "모임이나 식사 자리처럼 차려입은 느낌이 필요한 순간에도 자연스럽게 어울립니다.",
        "데님, 슬랙스, 스커트와 두루 매치하기 쉬워 코디 폭이 넓습니다.",
        "아우터 안에 받쳐 입거나 단독으로 연출해도 분위기가 흐트러지지 않습니다.",
    ]
    md_purchase = [
        f"{size}",
        "밝은 컬러 계열은 스킨톤 이너를 함께 입어주시면 더 깔끔하게 연출됩니다.",
        clean_line(data.get("washing") or "드라이클리닝, 단독 울세탁, 손세탁 권장. 건조기 사용 금지") + ".",
        "실측사이즈와 평소 선호하시는 핏을 함께 비교해 보시길 권해드립니다.",
    ]
    md_ending = [
        "기본 아이템일수록 오래 입기 좋은 밸런스가 중요합니다.",
        f"{product}은 그런 기준에 잘 맞는 데일리 상의로 추천드릴 만합니다.",
        "한 벌만으로도 차분하고 세련된 분위기를 완성해 보세요.",
    ]

    return {
        "material_desc_lines": material_lines,
        "recommend_lines": recommend,
        "review_lines": reviews,
        "faqs": smart_faq_defaults(data),
        "shopping_lines": shopping,
        "md_sections": {
            "choice": md_choice[:4],
            "fabric": md_fabric[:4],
            "fit": md_fit[:4],
            "occasion": md_occasion[:4],
            "purchase_note": md_purchase[:4],
            "ending": md_ending[:3],
        },
        "size_tips": {
            "55": "전체적으로 여유 있는 느낌으로 떨어져 단독 착용 시에도 부담 없이 활용하시기 좋습니다.",
            "66": "품과 실루엣이 가장 안정감 있게 정리되어 데일리부터 모임룩까지 자연스럽게 이어집니다.",
            "66half": "상체를 비교적 편안하게 감싸 주는 편이라 체형 고민을 덜고 입기 좋습니다.",
            "77": "답답하게 조이지 않고 여유 있게 착용 가능한 편으로 실측 확인 후 선택하시면 만족도가 높습니다.",
        },
    }


def generate_structured_copy(data: Dict[str, str], additional_request: str, uploaded_images) -> Dict[str, Any]:
    prompt = build_generation_prompt(data, additional_request)
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in uploaded_images[:5] if uploaded_images else []:
        user_content.append(file_to_content_item(img))

    response = chat_with_retry(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "사용자가 입력한 추가/수정 요청사항은 최우선으로 반드시 반영해야 한다."},
            {"role": "system", "content": "반드시 JSON만 출력한다. 없는 디테일은 추정하지 않는다. 입력 문구를 그대로 반복하지 말고 고객 니즈 중심의 매력적인 한국어 문장으로 재작성한다."},
            {"role": "user", "content": user_content},
        ],
        temperature=0.55,
        max_retries=2,
    )
    return extract_json(response.choices[0].message.content)


def safe_lines(value: Any, count: int, fallback: List[str], quote: bool = False) -> List[str]:
    items = value if isinstance(value, list) else []
    if quote:
        normalized = normalize_list(items, count, quote=False)
        if len(normalized) < count:
            normalized += [x for x in fallback if x not in normalized]
        normalized = normalized[:count]
        return [f'"{x.replace(chr(34), "").strip()}"' for x in normalized]
    normalized = normalize_list(items, count)
    if len(normalized) < count:
        for item in fallback:
            c = clean_line(item)
            if c and c not in normalized:
                normalized.append(c)
            if len(normalized) >= count:
                break
    return normalized[:count]


def safe_faqs(value: Any, fallback: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    items = value if isinstance(value, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = clean_line(item.get("q", ""))
        a = clean_line(item.get("a", ""))
        if not q or not a:
            continue
        if not q.startswith("Q."):
            q = "Q. " + q
        if not a.startswith("A."):
            a = "A. " + a
        out.append({"q": q, "a": a})
        if len(out) >= 4:
            break
    for item in fallback:
        if len(out) >= 4:
            break
        q = item["q"]
        if all(q != x["q"] for x in out):
            out.append(item)
    return out[:4]


def normalize_generated(result: Dict[str, Any], data: Dict[str, str]) -> Dict[str, Any]:
    fallback = fallback_structured(data)
    material_desc_lines = safe_lines(result.get("material_desc_lines"), 3, fallback["material_desc_lines"])
    recommend_lines = safe_lines(result.get("recommend_lines"), 4, fallback["recommend_lines"])
    review_lines = safe_lines(result.get("review_lines"), 4, fallback["review_lines"], quote=True)
    faqs = safe_faqs(result.get("faqs"), fallback["faqs"])
    shopping_lines = safe_lines(result.get("shopping_lines"), 3, fallback["shopping_lines"])

    md_raw = result.get("md_sections") if isinstance(result.get("md_sections"), dict) else {}
    md_sections = {
        "choice": safe_lines(md_raw.get("choice"), 4, fallback["md_sections"]["choice"]),
        "fabric": safe_lines(md_raw.get("fabric"), 4, fallback["md_sections"]["fabric"]),
        "fit": safe_lines(md_raw.get("fit"), 4, fallback["md_sections"]["fit"]),
        "occasion": safe_lines(md_raw.get("occasion"), 4, fallback["md_sections"]["occasion"]),
        "purchase_note": safe_lines(md_raw.get("purchase_note"), 4, fallback["md_sections"]["purchase_note"]),
        "ending": safe_lines(md_raw.get("ending"), 3, fallback["md_sections"]["ending"]),
    }

    tips_raw = result.get("size_tips") if isinstance(result.get("size_tips"), dict) else {}
    size_tips = {
        "55": ensure_sentence(tips_raw.get("55", fallback["size_tips"]["55"])),
        "66": ensure_sentence(tips_raw.get("66", fallback["size_tips"]["66"])),
        "66half": ensure_sentence(tips_raw.get("66half", fallback["size_tips"]["66half"])),
        "77": ensure_sentence(tips_raw.get("77", fallback["size_tips"]["77"])),
    }

    return {
        "material_desc_lines": [ensure_sentence(x) for x in material_desc_lines],
        "recommend_lines": [ensure_sentence(x) for x in recommend_lines],
        "review_lines": review_lines,
        "faqs": faqs,
        "shopping_lines": [ensure_sentence(x) for x in shopping_lines],
        "md_sections": {k: [ensure_sentence(x) for x in v] for k, v in md_sections.items()},
        "size_tips": size_tips,
    }


def build_subtap_html(data: Dict[str, str], material_desc_lines: List[str]) -> str:
    material_items = [x.strip() for x in (data["material"] or "").split("+") if x.strip()]
    material_line = " + ".join(material_items) if material_items else "소재 정보 입력 필요"
    if "(건조기사용금지)" not in material_line:
        material_line = f"{material_line} (건조기사용금지)"
    washing = (data["washing"] or "").strip() or "드라이클리닝, 단독 울세탁, 손세탁 권장. 건조기 사용 금지"
    size_tip = (data["size"] or "").strip() or "FREE 사이즈로 77까지 추천드립니다."
    measurement_html = format_measurement_lines(data["measurement_lines"])
    material_desc_html = "<br>\n\t\t\t\t\t\t\t\t\t".join(material_desc_lines) + "<br>"

    return f"""<div id=\"Subtap\">
\t<div id=\"header2\" role=\"banner\">
\t\t<nav class=\"nav\" role=\"navigation\">
\t\t\t<ul class=\"nav__list\">
\t\t\t\t<li>
\t\t\t\t\t<input id=\"group-1\" type=\"checkbox\" hidden=\"\">
\t\t\t\t\t<label for=\"group-1\" style=\"border-top-color: rgb(204, 204, 204); border-top-width: 1px; border-top-style: solid;\">
\t\t\t\t\t\t<p class=\"fa fa-angle-right\"></p>소재 정보</label>
\t\t\t\t\t<ul class=\"group-list\">
\t\t\t\t\t\t<li>
\t\t\t\t\t\t\t<a href=\"#\">
\t\t\t\t\t\t\t\t<h3>소재 : {material_line}</h3>
\t\t\t\t\t\t\t\t<p>
\t\t\t\t\t\t\t\t\t{material_desc_html}
\t\t\t\t\t\t\t\t</p>
\t\t\t\t\t\t\t\t<h3>세탁방법</h3>
\t\t\t\t\t\t\t\t<p>{washing}</p>
\t\t\t\t\t\t\t</a>
\t\t\t\t\t\t</li>
\t\t\t\t\t</ul>
\t\t\t\t</li>
\t\t\t\t<li>
\t\t\t\t\t<input id=\"group-2\" type=\"checkbox\" hidden=\"\">
\t\t\t\t\t<label for=\"group-2\">
\t\t\t\t\t\t<p class=\"fa fa-angle-right\"></p>사이즈 정보</label>
\t\t\t\t\t<ul class=\"group-list gray\">
\t\t\t\t\t\t<li>
\t\t\t\t\t\t\t<a href=\"#\">
\t\t\t\t\t\t\t\t<h3>사이즈 TIP</h3>
\t\t\t\t\t\t\t\t<p>{size_tip}</p>
\t\t\t\t\t\t\t\t<h3>길이 TIP</h3>
\t\t\t\t\t\t\t\t<p>162-167cm에서는 모델핏을 참고해 주시고,
<br> 다리 길이나 체형에 따라 다르지만,
<br> 160cm이하에서는 모델의 핏보다 조금 길게
<br> 연출됩니다.</p>
\t\t\t\t\t\t\t</a>
\t\t\t\t\t\t</li>
\t\t\t\t\t</ul>
\t\t\t\t</li>
\t\t\t\t<li>
\t\t\t\t\t<input id=\"group-3\" type=\"checkbox\" hidden=\"\">
\t\t\t\t\t<label for=\"group-3\">
\t\t\t\t\t\t<p class=\"fa fa-angle-right\"></p>실측 사이즈</label>
\t\t\t\t\t<ul class=\"group-list\">
\t\t\t\t\t\t<li>
\t\t\t\t\t\t\t<a href=\"#\">
\t\t\t\t\t\t\t\t<p>{measurement_html}</p>
\t\t\t\t\t\t\t</a>
\t\t\t\t\t\t</li>
\t\t\t\t\t</ul>
\t\t\t\t</li>
\t\t\t\t<li>
\t\t\t\t\t<input id=\"group-5\" type=\"checkbox\" hidden=\"\">
\t\t\t\t\t<label for=\"group-5\"><span class=\"fa fa-angle-right\"></span>
\t\t\t\t\t\t<a href=\"#crema-product-fit-1\" style=\"padding: 0px; box-shadow:none; background:#f7f7f7;\">실측사이즈 재는방법</a></label>
\t\t\t\t</li>
\t\t\t</ul>
\t\t</nav>
\t</div>
</div>"""


def render_text_source(structured: Dict[str, Any]) -> str:
    rec_lines = ''.join([f"▪ {wrap_text_source(to_noun(x))}<br>\n" for x in structured['recommend_lines']])
    review_lines = ''.join([f'"{wrap_text_source(clean_line(x))}"<br>\n' for x in structured['review_lines']])
    faq_lines = []
    for idx, faq in enumerate(structured['faqs']):
        q = wrap_text_source(clean_line(faq['q']))
        a = wrap_text_source(clean_line(faq['a']))
        faq_lines.append(f"Q. {q}<br>\n")
        faq_lines.append(f"A. {a}<br>\n")
        if idx < len(structured['faqs']) - 1:
            faq_lines.append("<br>\n")
    shopping_lines = ''.join([f"▪ {wrap_text_source(clean_line(x))}<br>\n" for x in structured['shopping_lines']])

    return (
        '<div style="text-align:center;">\n'
        '<h3 style="margin-bottom:0;">\n'
        '✓ 이런 분께 추천해요!</h3>\n'
        '<br>\n'
        '<p><span style="font-size:14px; line-height:1.8;">\n'
        f'{rec_lines}'
        '</span></p></div>\n'
        '<br><br><br><br>\n\n'
        '<div style="text-align:center;">\n'
        '<h3 style="margin-bottom:0;">\n'
        '✓ 미리 입어 본 착용후기 (모델/스텝/MD리뷰)</h3>\n'
        '<br>\n'
        '<p><span style="font-size:14px; line-height:1.8;">\n'
        f'{review_lines}'
        '</span></p></div>\n'
        '<br><br><br>\n\n'
        '<div style="text-align:center;">\n'
        '<h3 style="margin-bottom:0;">\n'
        '✓ (FAQ) 이 상품, 이게 궁금해요!</h3>\n'
        '<br>\n'
        '<p><span style="font-size:14px; line-height:1.4;">\n'
        f'{"".join(faq_lines)}'
        '</span></p></div>\n'
        '<br><br><br><br>\n\n'
        '<div style="text-align:center;">\n'
        '<h3 style="margin-bottom:0;">\n'
        '✓ 쇼핑에 꼭 참고하세요</h3>\n'
        '<br>\n'
        '<p><span style="font-size:14px; line-height:1.8;">\n'
        f'{shopping_lines}'
        '</span></p></div>\n'
        '<br><br><br>'
    )

def render_subsc_html(data: Dict[str, str], structured: Dict[str, Any]) -> str:
    md = structured['md_sections']

    def join_lines(lines: List[str]) -> str:
        return ''.join([f"{wrap_md(clean_line(x))}<br>\n" for x in lines if clean_line(x)])

    html = (
        '<div id="subsc">\n'
        f'<h3>{data["display_name"]}</h3>\n'
        '<p>\n'
        '<strong style="font-weight:700 !important;">[이 상품을 초이스한 이유입니다.]</strong><br>\n'
        f'{join_lines(md["choice"])}'
        '<br>\n'
        '<strong style="font-weight:700 !important;">[원단과 두께 체감에 대하여]</strong><br>\n'
        f'{join_lines(md["fabric"])}'
        '<br>\n'
        '<strong style="font-weight:700 !important;">[체형과 핏, 사이즈 선택 가이드]</strong><br>\n'
        f'{join_lines(md["fit"])}'
        '<br>\n'
        '<strong style="font-weight:700 !important;">[이렇게 입는 날이 많아집니다]</strong><br>\n'
        f'{join_lines(md["occasion"])}'
        '<br>\n'
        '</p></div>'
    )
    return clean_md(html)

def assemble_final_output(data: Dict[str, str], structured: Dict[str, Any]) -> str:
    material_items = [x.strip() for x in (data['material'] or '').split('+') if x.strip()]
    material_line = ' + '.join(material_items) if material_items else data['material']
    if '(건조기사용금지)' not in material_line:
        material_line = f'{material_line} (건조기사용금지)'

    text_source = render_text_source(structured)
    subsc_html = render_subsc_html(data, structured)
    subtap_html = build_subtap_html(data, structured['material_desc_lines'])
    source_block = FIXED_HTML_HEAD + '\n\n' + subsc_html + '\n\n' + subtap_html

    lines: List[str] = []
    lines.append(f"상품명 : {data['display_name']}")
    lines.append('')
    lines.append(f"컬러 : {data['color']}")
    lines.append(f"사이즈 : {data['size']}")
    lines.append(f"소재 : {material_line}")
    lines.append('소재설명 :')
    for x in structured['material_desc_lines']:
        lines.append(f'- {x}')
    lines.append(f"제조국 : {data['country']}")
    lines.append('')
    lines.append('-----------------')
    lines.append('포인트 원고(포토샵 작업)')
    lines.append('-----------------')
    lines.extend([''] * 11)
    lines.append('---------------------------------')
    lines.append('텍스트 소스')
    lines.append('---------------------------------')
    lines.append('')
    lines.append(text_source)
    lines.append('')
    lines.append('----------------------------------')
    lines.append('MD원고(상품 설명 소스)')
    lines.append('----------------------------------')
    lines.append(source_block)
    lines.append('')
    lines.append('-----------------')
    lines.append('사이즈 팁')
    lines.append('-----------------')
    lines.append('')
    lines.append('ㅇ55 (90) 160cm 48kg')
    lines.append(structured['size_tips']['55'])
    lines.append('')
    lines.append('ㅇ66 (95) 165cm 54kg')
    lines.append(structured['size_tips']['66'])
    lines.append('')
    lines.append('ㅇ66반 (95) 164cm 58kg')
    lines.append(structured['size_tips']['66half'])
    lines.append('')
    lines.append('ㅇ77 (100) 163cm 61kg')
    lines.append(structured['size_tips']['77'])
    return '\n'.join(lines).strip()


# -----------------------------
# UI
# -----------------------------
st.markdown("---")
st.subheader("상품 네이밍")
ncol1, ncol2 = st.columns([5, 1], vertical_alignment="bottom")
with ncol1:
    naming_input = st.text_area(
        "상품 주요특징 입력",
        height=120,
        placeholder="예: 여리핏, 부드러운 엠보 텍스처, 상체 군살 커버, 루즈핏 맨투맨",
        key="naming_input_value",
    )
with ncol2:
    if st.button("네이밍 생성", use_container_width=True):
        if naming_input.strip():
            with st.spinner("상품명을 생성 중입니다..."):
                try:
                    response = chat_with_retry(
                        model="gpt-4.1",
                        messages=[
                            {"role": "system", "content": NAME_PROMPT},
                            {"role": "user", "content": naming_input},
                        ],
                        temperature=0.5,
                        max_retries=2,
                    )
                    st.session_state.naming_result = response.choices[0].message.content.strip()
                    st.rerun()
                except RateLimitError:
                    st.error("현재 OpenAI 요청이 일시적으로 몰려 네이밍 생성을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요.")
                except Exception as e:
                    st.error(f"네이밍 생성 중 오류가 발생했습니다: {e}")
        else:
            st.warning("상품 주요특징을 입력해 주세요.")

st.text_area("상품 네이밍 제안 20개", value=st.session_state.naming_result, height=250)
st.markdown("---")

h1, h2 = st.columns([2.2, 1.0], vertical_alignment="bottom")
with h1:
    st.subheader("상품정보 입력")
with h2:
    st.button("초기화", use_container_width=True, on_click=reset_all)

nonce = st.session_state.reset_nonce
left, right = st.columns(2)

with left:
    product_name = st.text_input("상품명", value="( color)", key=f"product_name_{nonce}")
    color = st.text_input("컬러", placeholder="예: 1 먹 / 2 블랙", key=f"color_{nonce}")
    size = st.text_area("사이즈", height=90, value="FREE 사이즈로 77까지 추천드립니다.", key=f"size_{nonce}")
    material = st.text_input("소재", placeholder="예: 폴리에스터70 + 레이온27 + 스판3", key=f"material_{nonce}")
    material_desc = st.text_area("소재설명", height=110, placeholder="예: 후들후들 가볍고 부드러운 무광의 소재\n예: 비침이나 구김에 강하고 유연한 핏 연출", key=f"material_desc_{nonce}")
    country = st.text_input("제조국", key=f"country_{nonce}")
    top_measure = st.text_area("상의 실측사이즈", height=120, value="어깨단면 / 가슴둘레 / 암홀둘레 / 소매길이 / 소매둘레 / 총장 / 총장(앞) / 총장(뒤)  단위:cm", key=f"top_measure_{nonce}")
    bottom_measure = st.text_area("하의 실측사이즈", height=110, value="F 허리둘레 / 엉덩이둘레 / 허벅지둘레 / 밑단둘레 / 총장 / 밑위 길이  단위:cm\nL 허리둘레 / 엉덩이둘레 / 허벅지둘레 / 밑단둘레 / 총장 / 밑위 길이  단위:cm", key=f"bottom_measure_{nonce}")
    dress_measure = st.text_area("원피스 실측사이즈", height=130, value="어깨단면 / 가슴둘레 / 허리둘레 / 엉덩이둘레 / 암홀둘레 / 소매길이 / 어깨소매길이 / 총장(앞) / 총장(뒤)  단위:cm", key=f"dress_measure_{nonce}")

with right:
    detail_tip = st.text_input("디테일 특징 (예:디자인, 절개라인, 부자재, 스펙상 특징 등)", key=f"detail_tip_{nonce}")
    fit = st.text_input("핏/실루엣 (예:정핏,레귤러핏,오버핏 등/체형커버, 다리길어보이는 등의 특장점)", key=f"fit_{nonce}")
    appeal_points = st.text_area("주요 어필 포인트 (예:고객 문제해결 포인트,원단 구김-탄력-내구성,체형커버,계절성,기능성,코디활용도 등)", height=150, key=f"appeal_points_{nonce}")
    etc = st.text_area("기타 특징 (브랜드퀄리티,백화점납품상품,가격경쟁력,가성비,전문거래처 등)", height=120, key=f"etc_{nonce}")
    target = st.text_input("타겟", value="4050 여성", key=f"target_{nonce}")
    washing = st.text_input("세탁방법", value="드라이클리닝, 단독 울세탁, 손세탁 권장. 건조기 사용 금지", key=f"washing_{nonce}")
    additional_request = st.text_area("추가/수정 요청사항(출력물 확인 후 수정사항 입력)", height=120, key=f"additional_request_{nonce}")

st.subheader("이미지 업로드")
uploaded_images = st.file_uploader("이미지", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key=f"uploaded_images_{nonce}")

if st.button("생성하기", type="primary", use_container_width=True, key=f"generate_{nonce}"):
    display_name = apply_color_count_to_name(product_name, color)
    measurement_lines = combine_measurements(top_measure, bottom_measure, dress_measure)
    data = {
        "product_name": product_name,
        "display_name": display_name,
        "color": color,
        "size": size,
        "material": material,
        "material_desc_raw": material_desc,
        "country": country,
        "measurement_lines": measurement_lines,
        "detail_tip": detail_tip,
        "fit": fit,
        "appeal_points": appeal_points,
        "etc": etc,
        "target": target,
        "washing": washing,
    }

    with st.spinner("출력물을 생성 중입니다..."):
        try:
            structured_raw = generate_structured_copy(data, additional_request, uploaded_images)
            structured = normalize_generated(structured_raw, data)
            result = assemble_final_output(data, structured)
        except RateLimitError:
            st.error("현재 OpenAI 요청이 일시적으로 몰려 원고 생성을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요.")
            st.stop()
        except Exception as e:
            st.error(f"원고 생성 중 오류가 발생했습니다: {e}")
            st.stop()

    st.session_state.generated_result = result
    st.session_state.generated_docx = result_to_docx_bytes(result)
    st.session_state.generated_file_stem = (display_name or "page_builder").replace(" ", "_")

if st.session_state.generated_result:
    st.text_area("결과", st.session_state.generated_result, height=1200)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("TXT 다운로드", data=st.session_state.generated_result, file_name=f"{st.session_state.generated_file_stem}_output.txt", mime="text/plain", use_container_width=True)
    with c2:
        st.download_button("HWP 다운로드", data=st.session_state.generated_docx, file_name=f"{st.session_state.generated_file_stem}_output.hwp", mime="application/x-hwp", use_container_width=True)

st.markdown("---")
st.markdown("© made by MISHARP, MIYAWA. All rights reserved.")