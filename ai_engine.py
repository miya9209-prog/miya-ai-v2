import os
import json
from openai import OpenAI
from prompt import SYSTEM_PROMPT


def build_policy():
    return {
        "shipping": {
            "courier": "CJ 대한통운",
            "shipping_fee": 3000,
            "free_shipping_over": 70000,
            "delivery_time": "결제 완료 후 2~4일 (영업일 기준)",
            "same_day_dispatch_rule": "오후 2시 이전 주문은 당일 출고",
        },
        "exchange_return": {
            "exchange_possible": "사이즈 교환 가능 / 동일상품 교환 가능 / 타상품 교환 가능",
            "period": "상품 수령 후 7일 이내",
            "exchange_fee": 6000,
            "defect_wrong": "불량/오배송은 미샵 부담입니다.",
        }
    }


def ask_miya(messages, app_context):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    context_payload = {
        "policy": build_policy(),
        "app_context": app_context
    }

    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "다음 JSON을 기준으로만 답변하세요:\n" + json.dumps(context_payload, ensure_ascii=False)},
    ]

    chat_messages.extend(messages)

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=chat_messages,
        temperature=0.55,
        max_tokens=450
    )
    return resp.choices[0].message.content.strip()
