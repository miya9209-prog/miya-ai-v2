import re
from typing import List, Optional


def normalize_size_options(size_options: List[str]) -> List[str]:
    cleaned = []
    for s in size_options:
        s = s.strip()
        if not s:
            continue
        if "필수" in s or "선택" in s:
            continue
        cleaned.append(s)
    return cleaned


def detect_free_size(size_options: List[str]) -> Optional[str]:
    for s in size_options:
        up = s.upper()
        if "FREE" in up or up.startswith("F(") or up == "F":
            return s
    return None


def contains_alpha_sizes(size_options: List[str]) -> bool:
    joined = " ".join(size_options).upper()
    return any(x in joined for x in [" XS", " S", " M", " L", " XL", "XXL"]) or any(
        re.search(rf"(^|[^A-Z]){k}([^A-Z]|$)", joined) for k in ["S", "M", "L", "XL", "XXL"]
    )


def contains_korean_sizes(size_options: List[str]) -> bool:
    joined = " ".join(size_options)
    return any(x in joined for x in ["44", "55", "55반", "66", "66반", "77", "77반", "88"])


def pick_from_alpha(weight: float, options: List[str]) -> str:
    upper_map = {o.upper(): o for o in options}
    ordered = [x for x in ["XS", "S", "M", "L", "XL", "XXL"] if x in upper_map]

    if not ordered:
        return options[0]

    if weight <= 50 and "S" in upper_map:
        return upper_map["S"]
    if weight <= 58 and "M" in upper_map:
        return upper_map["M"]
    if weight <= 66 and "L" in upper_map:
        return upper_map["L"]
    if "XL" in upper_map:
        return upper_map["XL"]
    return upper_map[ordered[-1]]


def pick_from_korean(weight: float, options: List[str]) -> str:
    order = ["44", "55", "55반", "66", "66반", "77", "77반", "88"]
    available = [x for x in order if any(x == o or x in o for o in options)]

    if not available:
        return options[0]

    if weight <= 47:
        target = "44"
    elif weight <= 53:
        target = "55"
    elif weight <= 56:
        target = "55반"
    elif weight <= 61:
        target = "66"
    elif weight <= 65:
        target = "66반"
    elif weight <= 70:
        target = "77"
    elif weight <= 74:
        target = "77반"
    else:
        target = "88"

    for x in available:
        if x == target:
            return x

    # 가장 가까운 상위 우선, 없으면 마지막
    try:
        t_idx = order.index(target)
        for x in available:
            if order.index(x) >= t_idx:
                return x
    except ValueError:
        pass

    return available[-1]


def recommend_size(
    height_cm: Optional[str],
    weight_kg: Optional[str],
    top_size: Optional[str],
    product_category: str,
    size_options: List[str]
) -> dict:
    options = normalize_size_options(size_options)

    if not options:
        return {
            "recommended": None,
            "reason": "상품 옵션 정보가 없어 추천 사이즈를 확정하기 어렵습니다."
        }

    free_size = detect_free_size(options)
    if free_size:
        reason = "이 상품은 free/F 계열 옵션이라 기본적으로 그 옵션 안에서 핏을 보시면 됩니다."
        if "66반" in free_size or "55~66" in free_size:
            reason = f"이 상품은 {free_size} 옵션 기준으로 입는 상품입니다."
        return {"recommended": free_size, "reason": reason}

    try:
        weight = float(weight_kg) if weight_kg not in ("", None) else None
    except ValueError:
        weight = None

    if weight is None:
        if top_size and top_size.strip():
            return {
                "recommended": top_size,
                "reason": "입력하신 평소 상의/하의 사이즈를 기준으로 먼저 보시는 게 가장 안전합니다."
            }
        return {
            "recommended": options[0],
            "reason": "체형 정보가 부족해서 가장 기본 옵션부터 보시는 게 좋습니다."
        }

    if contains_alpha_sizes(options):
        rec = pick_from_alpha(weight, options)
        return {
            "recommended": rec,
            "reason": "현재 체중 기준으로 가장 무난한 알파벳 옵션을 우선 추천드렸습니다."
        }

    if contains_korean_sizes(options):
        rec = pick_from_korean(weight, options)
        return {
            "recommended": rec,
            "reason": "현재 체중 기준으로 가장 가까운 숫자 옵션을 기준으로 추천드렸습니다."
        }

    return {
        "recommended": options[0],
        "reason": "옵션 체계가 일반적이지 않아 첫 번째 유효 옵션을 기준으로 추천드렸습니다."
    }
