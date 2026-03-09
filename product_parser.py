import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional


HEADERS = {"User-Agent": "Mozilla/5.0"}


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def get_product_no_from_url(url: str) -> str:
    m = re.search(r"product_no=(\d+)", url)
    return m.group(1) if m else ""


def _extract_name(soup: BeautifulSoup, fallback_name: str = "") -> str:
    selectors = [
        "#span_product_name",
        "#span_product_name_mobile",
        ".infoArea #span_product_name",
        ".infoArea .headingArea h2",
        ".infoArea .headingArea h3",
        ".headingArea h2",
        ".headingArea h3",
        "meta[property='og:title']",
        "title",
    ]

    for sel in selectors:
        el = soup.select_one(sel)
        if not el:
            continue

        if el.name == "meta":
            txt = clean_text(el.get("content", ""))
        else:
            txt = clean_text(el.get_text(" ", strip=True))

        if txt and txt not in {"미샵", "MISHARP"}:
            txt = re.sub(r"\s*-\s*미샵.*$", "", txt, flags=re.I).strip()
            return txt

    return fallback_name or "지금 보시는 상품"


def _extract_options_from_selects(soup: BeautifulSoup) -> Dict[str, List[str]]:
    selects = soup.select("select")
    all_options = []

    for sel in selects:
        opts = []
        for opt in sel.select("option"):
            text = clean_text(opt.get_text(" ", strip=True))
            if not text:
                continue
            if any(bad in text for bad in ["필수 옵션", "옵션 선택", "선택해주세요", "품절", "----"]):
                continue
            opts.append(text)
        if opts:
            all_options.append(opts)

    colors = []
    sizes = []

    # 흔한 패턴: 첫 번째 유효 select=컬러, 두 번째=사이즈
    if len(all_options) >= 1:
        colors = all_options[0]
    if len(all_options) >= 2:
        sizes = all_options[1]

    return {"colors": _uniq(colors), "sizes": _uniq(sizes)}


def _uniq(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_meta_summary(soup: BeautifulSoup) -> str:
    texts = []
    for sel in [
        ".prdDesc",
        ".detailArea",
        ".infoArea",
        "#prdDetail",
        ".cont",
        ".detail",
    ]:
        el = soup.select_one(sel)
        if el:
            texts.append(clean_text(el.get_text(" ", strip=True)))
    merged = " ".join(texts)
    return merged[:1800]


def guess_category(name: str, text: str) -> str:
    corpus = f"{name} {text}"
    mapping = {
        "슬랙스": ["슬랙스", "팬츠", "바지"],
        "블라우스": ["블라우스"],
        "셔츠": ["셔츠"],
        "티셔츠": ["티셔츠", "탑"],
        "니트": ["니트", "가디건", "맨투맨"],
        "자켓": ["자켓", "재킷"],
        "원피스": ["원피스"],
        "데님": ["데님", "청바지"],
        "코트": ["코트"],
    }
    for cat, keywords in mapping.items():
        if any(k in corpus for k in keywords):
            return cat
    return "기타"


def parse_product(url: str, passed_name: str = "") -> Dict:
    r = requests.get(url, headers=HEADERS, timeout=12)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    product_name = _extract_name(soup, passed_name)
    options = _extract_options_from_selects(soup)
    summary = _extract_meta_summary(soup)
    category = guess_category(product_name, summary)

    return {
        "product_no": get_product_no_from_url(url),
        "url": url,
        "product_name": product_name,
        "category": category,
        "color_options": options["colors"],
        "size_options": options["sizes"],
        "summary": summary,
    }


def try_extract_product_url_from_message(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"https?://[^\s]+product/detail\.html\?[^\s]+", text)
    return m.group(0) if m else None
