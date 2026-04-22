from engine.fit_effect_engine import fit_effect_answer

def detect_intent(text):
    if any(x in text for x in ["커버","체형","다리 짧","다리가 짧","비율"]):
        return "fit_effect"
    if any(x in text for x in ["66","77"]) and "좋" in text:
        return "size_choice"
    if any(x in text for x in ["블랙","화이트","컬러"]):
        return "color"
    if "추천" in text:
        if "셔츠" in text: return "shirt"
        if "자켓" in text: return "jacket"
        return "recommend"
    return "fit"

def handle_message(text):
    intent = detect_intent(text)

    if intent=="fit_effect":
        r=fit_effect_answer(text)
        if r: return r

    if intent=="size_choice":
        if "힙" in text:
            return "66반에 힙이 있는 편이면 77이 더 편해요."
        return "66/77 중 여유감이면 77 추천이에요."

    if intent=="color":
        return "출근룩이면 블랙이 가장 무난하고 활용도 높아요."

    if intent=="shirt":
        return "셔츠 추천\n1. 드라이 실키 셔츠\n2. 베이직 셔츠\n3. 루즈핏 셔츠"

    if intent=="jacket":
        return "자켓 추천\n1. 멀린 무드 자켓\n2. 라이트 자켓\n3. 클래식 자켓"

    return "지금 질문 기준으로 바로 이어서 봐드릴게요."
