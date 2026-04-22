def handle_message(text, analysis, state):
    intent = analysis.get("intent")
    if intent == "size_choice":
        if "66반" in text and ("힙" in text or "엉덩이" in text):
            return "66반에 힙이 있는 편이면 77이 더 편하고 안정적이에요."
        return "66과 77 중에서는 여유감 원하시면 77이 더 편해요."
    if intent == "color":
        return "출근룩이면 블랙이 가장 안정적이고 활용도가 높아요."
    if intent == "recommend":
        if analysis.get("category") == "shirt":
            return "셔츠 추천드리면\n1. 드라이 실키 셔츠\n2. 베이직 코튼 셔츠\n3. 루즈핏 셔츠"
        if analysis.get("category") == "jacket":
            return "자켓 추천드리면\n1. 멀린 무드 자켓\n2. 라이트 페미닌 자켓\n3. 클래식 자켓"
        return "추천을 다시 골라드릴게요."
    return "지금 질문 기준으로 바로 이어서 봐드릴게요."
