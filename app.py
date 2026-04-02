
import streamlit as st
import os
import datetime

st.set_page_config(page_title="미야언니", layout="centered")

# ------------------ 상태 ------------------
def ensure_state():
    defaults = {
        "messages": [],
        "body_height": "",
        "body_weight": "",
        "body_top": "",
        "body_bottom": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_state()

# ------------------ 체형 ------------------
def build_body_context():
    return {
        "height_cm": str(st.session_state.get("body_height", "")),
        "weight_kg": str(st.session_state.get("body_weight", "")),
        "top_size": str(st.session_state.get("body_top", "")),
        "bottom_size": str(st.session_state.get("body_bottom", "")),
    }

def build_body_context_text(body_ctx):
    if not any(body_ctx.values()):
        return "입력된 체형 정보 없음"
    return (
        f"키: {body_ctx.get('height_cm') or '-'}cm, "
        f"체중: {body_ctx.get('weight_kg') or '-'}kg, "
        f"상의: {body_ctx.get('top_size') or '-'}, "
        f"하의: {body_ctx.get('bottom_size') or '-'}"
    )

# ------------------ 로그 ------------------
def save_log(user, answer):
    os.makedirs("logs", exist_ok=True)
    filename = f"logs/chat_{datetime.date.today()}.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[USER] {user}\n[AI] {answer}\n\n")

# ------------------ 간단 응답 ------------------
def process_user_message(user_text):
    if "맞을까" in user_text or "사이즈" in user_text:
        return "지금 체형 기준으로 보면 살짝 타이트하게 느껴질 수 있어요 🙂 조금 여유 있는 쪽이 더 편하실 거예요."
    if "추천" in user_text:
        return "이런 쪽 한번 같이 보시면 좋아요 🙂 여유핏 자켓 / 깔끔핏 자켓 / 단정한 스타일 추천드려요."
    return "같이 봐드릴게요 🙂 어떤 부분이 궁금하세요?"

# ------------------ UI ------------------
st.title("미샵 쇼핑친구 미야언니")

col1, col2 = st.columns(2)
with col1:
    st.text_input("키", key="body_height")
    st.selectbox("상의", ["", "55", "66", "77", "77반"], key="body_top")
with col2:
    st.text_input("체중", key="body_weight")
    st.selectbox("하의", ["", "55", "66", "77", "77반"], key="body_bottom")

st.caption("현재 입력 정보: " + build_body_context_text(build_body_context()))

# 채팅 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 입력
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    answer = process_user_message(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_log(user_input, answer)
    st.rerun()
