import json
import os
from typing import Dict


MOCK_PATH = os.path.join("data", "customer_mock.json")


def load_mock_customer(customer_id: str) -> Dict:
    if not customer_id:
        return {}
    if not os.path.exists(MOCK_PATH):
        return {}
    try:
        with open(MOCK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(customer_id, {})
    except Exception:
        return {}


def build_customer_context(query_params) -> Dict:
    """
    실제 Cafe24 API 연결 전까지는
    query param + mock 데이터 구조로 유지.
    UI는 깨지지 않고, 나중에 API만 갈아끼우면 됨.
    """
    customer_id = query_params.get("cid", "") or ""
    customer_name = query_params.get("cname", "") or ""
    member_group = query_params.get("cgroup", "") or ""
    is_logged_in = str(query_params.get("logged_in", "")).lower() in {"1", "true", "yes"}

    mock = load_mock_customer(customer_id)

    return {
        "customer_id": customer_id,
        "customer_name": customer_name or mock.get("customer_name", ""),
        "member_group": member_group or mock.get("member_group", ""),
        "is_logged_in": is_logged_in or bool(mock),
        "saved_top_size": mock.get("saved_top_size", ""),
        "saved_bottom_size": mock.get("saved_bottom_size", ""),
        "last_purchase_size": mock.get("last_purchase_size", ""),
        "recent_product_names": mock.get("recent_product_names", []),
    }
