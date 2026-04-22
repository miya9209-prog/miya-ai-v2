def fit_effect_answer(text):
    if "다리" in text and "짧" in text:
        return "배기핏이라 하체 라인을 부드럽게 정리해줘서 커버에는 도움이 되는 편이에요. 다만 기장이 길면 비율이 짧아 보일 수 있어요."
    if "커버" in text:
        return "핀턱과 여유 있는 실루엣이라 체형 커버에는 도움이 되는 타입이에요."
    return None
