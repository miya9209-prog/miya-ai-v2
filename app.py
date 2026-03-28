import streamlit as st
from openai import OpenAI
import hashlib

client = OpenAI()

SIZE_ORDER = ["44", "55", "66", "66반", "77", "88"]

def is_duplicate(text):
    h = hashlib.md5(text.strip().encode()).hexdigest()
    if st.session_state.get("last_q") == h:
        return True
    st.session_state["last_q"] = h
    return False

def normalize(size):
    return size.replace(" ", "").strip()

def is_over(user, max_size):
    try:
        return SIZE_ORDER.index(normalize(user)) > SIZE_ORDER.index(normalize(max_size))
    except:
        return False

def size_answer(user_size, max_size):
    if is_over(user_size, max_size):
        return f"""이 상품은 {max_size}까지 추천되는 디자인이라
고객님 상의 {user_size} 기준으로는 타이트하게 느껴질 가능성이 있어요.

편하게 입으시려면
조금 더 여유 있는 상품을 보시는 걸 추천드려요 :)"""

    return f"""현재 입력 기준으로는 {user_size} 기준으로 무리 없는 쪽으로 보여요 :)

이 상품은 {max_size}까지 추천되는 디자인이라
편안한 핏으로 입으실 가능성이 높아요.

원하시면 핏감도 더 자세히 봐드릴게요 :)"""

def is_size_question(text):
    return any(k in text for k in ["사이즈", "맞을까", "핏"])

def ask_llm(text):
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "미샵 쇼핑 상담사처럼 정확하고 짧게 답변해."},
            {"role": "user", "content": text}
        ],
        max_tokens=200
    )
    return resp.choices[0].message.content

def process(text, user_size, max_size):
    if is_duplicate(text):
        return st.session_state.get("last_answer", "잠시 후 다시 도와드릴게요 :)")

    if is_size_question(text):
        answer = size_answer(user_size, max_size)
        st.session_state["last_answer"] = answer
        return answer

    answer = ask_llm(text)
    st.session_state["last_answer"] = answer
    return answer

st.title("미샵 쇼핑친구 미야언니")

user_input = st.text_input("메시지를 입력하세요")

user_size = "77"
max_size = "66반"

if user_input:
    answer = process(user_input, user_size, max_size)
    st.write(answer)
