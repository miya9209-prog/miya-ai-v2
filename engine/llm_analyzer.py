def analyze(text):
    t = text
    if any(x in t for x in ["66", "77"]) and any(x in t for x in ["좋을까", "선택", "뭐가"]):
        return {"intent":"size_choice"}
    if any(x in t for x in ["블랙","화이트","아이보리","베이지","네이비","컬러","색"]):
        return {"intent":"color"}
    if "추천" in t:
        if "셔츠" in t:
            return {"intent":"recommend","category":"shirt"}
        if "자켓" in t:
            return {"intent":"recommend","category":"jacket"}
        return {"intent":"recommend"}
    if any(x in t for x in ["맞", "어울", "핏", "타이트"]):
        return {"intent":"fit"}
    return {"intent":"fit"}
