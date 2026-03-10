import html
import os
import streamlit as st

from product_parser import parse_product, try_extract_product_url_from_message
from size_engine import recommend_size
from ai_engine import ask_miya
from customer_context import build_customer_context


st.set_page_config(
    page_title="미야언니",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY가 필요합니다.")
    st.stop()

if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

qp = st.query_params

current_url = qp.get("url", "") or ""
product_no = qp.get("pn", "") or ""
product_name_q = qp.get("pname", "") or ""

customer_ctx = build_customer_context(qp)


def ensure_state():
    defaults = {
        "messages": [],
        "last_context_key": "",
        "body_height": "",
        "body_weight": "",
        "body_top": "",
        "body_bottom": "",
        "product_context": None,
        "mode": "general",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_state()


def build_context_key(url: str, pn: str, pname: str) -> str:
    return f"{url}|{pn}|{pname}"


def load_product_context(url: str, pname: str):
    if not url:
        return None
    try:
        return parse_product(url, pname)
    except Exception:
        return None


incoming_key = build_context_key(current_url, product_no, product_name_q)
if incoming_key != st.session_state.last_context_key:
    st.session_state.last_context_key = incoming_key
    st.session_state.messages = []
    st.session_state.product_context = load_product_context(current_url, product_name_q) if current_url else None
    st.session_state.mode = "product" if st.session_state.product_context else "general"


def body_context():
    return {
        "height_cm": st.session_state.body_height.strip(),
        "weight_kg": st.session_state.body_weight.strip(),
        "top_size": st.session_state.body_top.strip(),
        "bottom_size": st.session_state.body_bottom.strip(),
    }


def build_app_context():
    product_ctx = st.session_state.product_context
    body_ctx = body_context()

    size_result = None
    if product_ctx:
        size_result = recommend_size(
            height_cm=body_ctx["height_cm"],
            weight_kg=body_ctx["weight_kg"],
            top_size=body_ctx["top_size"],
            product_category=product_ctx.get("category", ""),
            size_options=product_ctx.get("size_options", []),
        )

    return {
        "mode": st.session_state.mode,
        "current_product": product_ctx,
        "body_context": body_ctx,
        "customer_context": customer_ctx,
        "recommended_size": size_result,
    }


def maybe_switch_product_from_message(user_text: str):
    detected = try_extract_product_url_from_message(user_text)
    if detected:
        new_ctx = load_product_context(detected, "")
        if new_ctx:
            st.session_state.product_context = new_ctx
            st.session_state.mode = "product"
            st.session_state.messages = []
            st.session_state.last_context_key = build_context_key(
                detected,
                new_ctx.get("product_no", ""),
                new_ctx.get("product_name", "")
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{new_ctx.get('product_name', '지금 보시는 상품')} 기준으로 다시 같이 봐드릴게요 :)"
            })


# ---------- UI ----------
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
  --miya-bot-bg:#071b4e;
  --miya-user-bg:#dff0ec;
  --miya-user-text:#1f3b36;
}

/* 라이트 기본 */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  --miya-text-main:#303443;
  --miya-text-sub:#5f6471;
  --miya-text-muted:#7a7f8c;
  --miya-divider:#ccccd2;
  --miya-name-bot:#5f6471;
  --miya-name-user:#0f6a63;
}

/* 다크 모드 텍스트 */
@media (prefers-color-scheme: dark){
  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
    --miya-text-main:#f3f5f8;
    --miya-text-sub:#cfd6e0;
    --miya-text-muted:#b5bfcb;
    --miya-divider:rgba(255,255,255,0.22);
    --miya-name-bot:#d9e0ea;
    --miya-name-user:#66d7c6;
  }
}

div[data-testid="column"]{
  min-width:0 !important;
}

div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label{
  color:var(--miya-text-main) !important;
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
  margin-top:3px !important;
  margin-bottom:3px !important;
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

/* 입력 placeholder 가독성 */
input::placeholder{
  color:var(--miya-text-muted) !important;
  opacity:1 !important;
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
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center; margin:0 0 16px 0;">
      <div style="font-size:31px; font-weight:800; line-height:1.1; letter-spacing:-0.02em; color:var(--miya-text-main);">
        미샵 쇼핑친구 <span style="color:#0f6a63;">미야언니</span>
      </div>
      <div style="margin-top:6px; font-size:13.5px; line-height:1.35; color:var(--miya-text-sub);">
        24시간 언제나 미샵님들 쇼핑 판단에 도움드리는 스마트한 쇼핑친구
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="margin-top:2px; margin-bottom:4px;">
      <div style="font-size:13px; font-weight:700; line-height:1.2; color:var(--miya-text-main); margin-bottom:4px;">
        사이즈 입력<span style="font-size:11px; font-weight:500; color:var(--miya-text-muted);">(더 구체적인 상담 가능)</span>
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

size_options_ui = ["", "44", "55", "55반", "66", "66반", "77", "77반", "88"]

row2 = st.columns(2, gap="small")
with row2[0]:
    current_top = st.session_state.body_top if st.session_state.body_top in size_options_ui else ""
    st.session_state.body_top = st.selectbox(
        "상의",
        options=size_options_ui,
        index=size_options_ui.index(current_top),
        key="body_top_input"
    )
with row2[1]:
    current_bottom = st.session_state.body_bottom if st.session_state.body_bottom in size_options_ui else ""
    st.session_state.body_bottom = st.selectbox(
        "하의",
        options=size_options_ui,
        index=size_options_ui.index(current_bottom),
        key="body_bottom_input"
    )

st.markdown("</div></div>", unsafe_allow_html=True)

app_ctx = build_app_context()

if app_ctx["mode"] == "product" and app_ctx["current_product"]:
    product_label = app_ctx["current_product"]["product_name"]
else:
    product_label = "일반 상담"

st.markdown(
    f'<div style="margin-top:2px; margin-bottom:2px; font-size:10.8px; color:var(--miya-text-muted);">현재 상담 기준: {html.escape(product_label)}</div>',
    unsafe_allow_html=True
)

if app_ctx["recommended_size"] and app_ctx["recommended_size"]["recommended"]:
    st.markdown(
        f'<div style="margin-top:0; margin-bottom:2px; font-size:10.8px; color:#0f6a63;">추천 사이즈: {html.escape(str(app_ctx["recommended_size"]["recommended"]))}</div>',
        unsafe_allow_html=True
    )

if not st.session_state.messages:
    if st.session_state.mode == "product" and st.session_state.product_context:
        product_name = st.session_state.product_context.get("product_name", "지금 보시는 상품")
        welcome = (
            f"안녕하세요? 옷 같이 봐드리는 미야언니예요 :) \n"
            f"'{product_name}' 기준으로 제가 같이 봐드릴게요. \n"
            f"사이즈, 코디, 배송, 교환 중 뭐부터 이야기해볼까요?"
        )
    else:
        welcome = (
            "안녕하세요? 옷 같이 봐드리는 미야언니예요 :) \n"
            "궁금한 상품 링크나 상품명을 보내주시면 바로 같이 봐드릴게요. \n"
            "사이즈, 코디, 배송, 교환 중 뭐부터 이야기해볼까요?"
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
                '<div style="display:block; font-size:12px; font-weight:700; line-height:1.15; color:var(--miya-name-user); text-align:right; margin:0 6px 1px 0;">고객님</div>'
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
                '<div style="display:block; font-size:12px; font-weight:700; line-height:1.15; color:var(--miya-name-bot); margin:0 0 1px 6px;">미야언니</div>'
                f'<div style="padding:10px 14px 10px 10px; border-radius:18px; border-bottom-left-radius:6px; font-size:15px; line-height:1.5; white-space:pre-wrap; word-break:keep-all; background:#071b4e; color:#ffffff; border:1px solid rgba(255,255,255,.08);">{safe_text}</div>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True
        )

user_input = st.chat_input("메시지를 입력하세요…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    maybe_switch_product_from_message(user_input)

    app_ctx = build_app_context()
    llm_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]

    answer = ask_miya(llm_messages, app_ctx)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
