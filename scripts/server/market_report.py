from __future__ import annotations

import base64
import html
import json
import os
import re
import shutil
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scripts.lib.schema import utc_now_iso
from scripts.server.tcgpro_report import image_generator

ROOT = Path(__file__).resolve().parents[2]
REPORT_OUTPUT_DIR = Path(os.environ.get("CARD_SCAN_REPORT_OUTPUT_DIR", "/tmp/card_scan_reports"))
SNKR_HISTORY_TIMEOUT = float(os.environ.get("CARD_SCAN_SNKR_HISTORY_TIMEOUT", "12"))
DEFAULT_JPY_RATE = float(os.environ.get("CARD_SCAN_MARKET_JPY_RATE", "150"))


def safe_report_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "report"


def report_file_url(base_url: str | None, report_id: str, path: Path) -> str | None:
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}/reports/{safe_report_filename(report_id)}/{path.name}"


def parse_snkr_product_id(recognition: dict[str, Any]) -> str:
    results = recognition.get("results")
    if not isinstance(results, list) or not results:
        return ""
    top = results[0] if isinstance(results[0], dict) else {}
    snkr = top.get("snkr") if isinstance(top.get("snkr"), dict) else {}
    value = snkr.get("product_id") or top.get("snkr_product_id")
    return str(value).strip() if value not in (None, "") else ""


def top_result(recognition: dict[str, Any]) -> dict[str, Any]:
    results = recognition.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return results[0]
    return {}


def best_display_name(result: dict[str, Any]) -> str:
    for key in ("name_ja", "name", "name_en", "card_id", "canonical_id"):
        value = result.get(key)
        if value:
            return str(value)
    return "Unknown Card"


def fetch_snkr_trading_histories(product_id: str, per_page: int = 100, page: int = 1) -> dict[str, Any]:
    url = f"https://snkrdunk.com/en/v1/streetwears/{product_id}/trading-histories?perPage={per_page}&page={page}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 card-scan-market-report/0.1",
        },
        method="GET",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=SNKR_HISTORY_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return {
            "status": "http_error",
            "http_status": exc.code,
            "body_preview": body,
            "url": url,
            "seconds": time.perf_counter() - started,
        }
    except (TimeoutError, URLError, OSError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "error": str(exc),
            "url": url,
            "seconds": time.perf_counter() - started,
        }
    histories = payload.get("histories") or payload.get("data") or []
    return {
        "status": "ok",
        "url": url,
        "seconds": time.perf_counter() - started,
        "histories": histories if isinstance(histories, list) else [],
    }


def parse_dt(value: Any) -> datetime:
    text = str(value or "")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.utcnow()


def date_label(value: Any) -> str:
    return parse_dt(value).strftime("%Y/%m/%d")


def usd(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def jpy_from_usd(value: Any, jpy_rate: float) -> int:
    return int(round(usd(value) * jpy_rate))


def iqr_filter(histories: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    bucket = [item for item in histories if item.get("condition") == condition]
    values = [usd(item.get("price")) for item in bucket]
    if len(values) >= 4:
        q1, _median_raw, q3 = statistics.quantiles(sorted(values), n=4, method="inclusive")
    elif values:
        q1 = min(values)
        q3 = max(values)
    else:
        q1 = 0.0
        q3 = 0.0
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item in bucket:
        price = usd(item.get("price"))
        if lower <= price <= upper:
            kept.append(item)
        else:
            dropped.append(item)
    kept_values = [usd(item.get("price")) for item in kept]
    average_usd = sum(kept_values) / len(kept_values) if kept_values else 0.0
    return {
        "condition": condition,
        "records": bucket,
        "kept_records": kept,
        "dropped_records": dropped,
        "summary": {
            "condition": condition,
            "total_count": len(bucket),
            "kept_count": len(kept),
            "dropped_count": len(dropped),
            "outlier_count": len(dropped),
            "q1_usd": q1,
            "q3_usd": q3,
            "iqr_usd": iqr,
            "lower_bound_usd": lower,
            "upper_bound_usd": upper,
            "latest_usd": kept_values[0] if kept_values else 0.0,
            "latest_price_usd": kept_values[0] if kept_values else 0.0,
            "latest_traded_at": kept[0].get("tradedAt") if kept else None,
            "average_usd": average_usd,
            "iqr_average_usd": average_usd,
            "median_usd": statistics.median(kept_values) if kept_values else 0.0,
            "min_usd": min(kept_values) if kept_values else 0.0,
            "max_usd": max(kept_values) if kept_values else 0.0,
            "dropped_prices_usd": [usd(item.get("price")) for item in dropped],
        },
    }


def snkr_records_for_poster(records: list[dict[str, Any]], grade: str, jpy_rate: float) -> list[dict[str, Any]]:
    return [
        {
            "date": date_label(item.get("tradedAt")),
            "grade": grade,
            "price": jpy_from_usd(item.get("price"), jpy_rate),
            "source": "SNKRDUNK",
        }
        for item in records
    ]


def usd_records_for_poster(records: list[dict[str, Any]], grade: str) -> list[dict[str, Any]]:
    return [
        {
            "date": date_label(item.get("tradedAt")),
            "grade": grade,
            "price": usd(item.get("price")),
            "source": "SNKRDUNK",
        }
        for item in records
    ]


def money(value: Any, digits: int = 2) -> str:
    return f"${usd(value):.{digits}f}"


def money0(value: Any) -> str:
    return f"${usd(value):.0f}"


def jpy_display(usd_value: Any, jpy_rate: float) -> str:
    return f"¥{jpy_from_usd(usd_value, jpy_rate):,} (~${usd(usd_value):.0f} USD)"


def build_markdown_report(
    recognition: dict[str, Any],
    product_id: str,
    snkr_url: str,
    raw_bucket: dict[str, Any],
    psa_bucket: dict[str, Any],
    jpy_rate: float,
) -> str:
    top = top_result(recognition)
    name = best_display_name(top)
    name_en = top.get("name_en") or top.get("name") or ""
    set_id = top.get("set_id") or "-"
    number = top.get("card_code") or "-"
    language = top.get("language") or "-"
    variant = top.get("variant") or top.get("edition") or "-"
    raw = raw_bucket["summary"]
    psa = psa_bucket["summary"]

    lines: list[str] = []
    lines.append("⚠️ 免責聲明：請確認相對應連結中的卡片是否與您上傳的卡片一致，機器人有時可能誤判。")
    lines.append("")
    lines.append("# MARKET REPORT GENERATED")
    lines.append("")
    identity = f"{name}"
    if name_en and name_en != name:
        identity += f" ({name_en})"
    lines.append(f"⚡ {identity} #{number}")
    lines.append(f"🏷️ 系列：{set_id}")
    lines.append(f"🔢 編號：{number}")
    lines.append(f"🌐 語言：{language}")
    lines.append(f"🧩 版本：{variant}")
    lines.append(f"🧷 SNKR product_id：{product_id}")
    lines.append("---")
    lines.append("")
    lines.append("## Direct Price Summary")
    lines.append("")
    lines.append("| Bucket | Latest | IQR Avg | Median | Min | Max | Kept | Removed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Raw / A | {money0(raw['latest_usd'])} | {money(raw['average_usd'])} | "
        f"{money0(raw['median_usd'])} | {money0(raw['min_usd'])} | {money0(raw['max_usd'])} | "
        f"{raw['kept_count']}/{raw['total_count']} | {raw['dropped_count']} |"
    )
    lines.append(
        f"| PSA 10 | {money0(psa['latest_usd'])} | {money(psa['average_usd'])} | "
        f"{money0(psa['median_usd'])} | {money0(psa['min_usd'])} | {money0(psa['max_usd'])} | "
        f"{psa['kept_count']}/{psa['total_count']} | {psa['dropped_count']} |"
    )
    lines.append("")
    lines.append(
        "IQR filter: keep records from `Q1 - 1.5*IQR` through `Q3 + 1.5*IQR` per bucket."
    )
    lines.append("---")
    lines.append("")
    lines.append("## Raw / A Transactions")
    for item in raw_bucket["kept_records"][:10]:
        lines.append(f"📅 {date_label(item.get('tradedAt'))}      💰 {money(item.get('price'))} USD      📝 狀態：Raw / A")
    if not raw_bucket["kept_records"]:
        lines.append("Raw / A: 無成交紀錄")
    lines.append("")
    lines.append("## PSA 10 Transactions")
    for item in psa_bucket["kept_records"][:10]:
        lines.append(f"📅 {date_label(item.get('tradedAt'))}      💰 {jpy_display(item.get('price'), jpy_rate)}      📝 狀態：PSA 10")
    if not psa_bucket["kept_records"]:
        lines.append("PSA 10: 無成交紀錄")
    lines.append("---")
    lines.append("")
    lines.append("## Source Links")
    lines.append(f"🔗 [查看 SNKRDUNK]({snkr_url})")
    lines.append(f"🔗 [查看 SNKRDUNK 銷售歷史]({snkr_url}/sales-histories)")
    return "\n".join(lines)


def card_data_from_recognition(
    recognition: dict[str, Any],
    raw_summary: dict[str, Any],
    psa_summary: dict[str, Any],
) -> dict[str, Any]:
    top = top_result(recognition)
    name = top.get("name") or top.get("name_en") or top.get("name_ja") or "Unknown Card"
    return {
        "name": name,
        "c_name": top.get("name_ja") or name,
        "jp_name": top.get("name_ja") or "",
        "number": top.get("card_code") or "Unknown",
        "set_code": top.get("set_id") or "Unknown",
        "set_name": top.get("set_id") or "Unknown",
        "grade": "PSA 10",
        "category": top.get("rarity") or top.get("variant") or "TCG",
        "language": top.get("language") or "",
        "img_url": top.get("image_url") or top.get("display_image_url") or "",
        "release_info": top.get("set_id") or "",
        "illustrator": top.get("illustrator") or "Unknown",
        "market_heat": f"Medium: Raw/A latest US {money0(raw_summary['latest_usd'])}; PSA 10 latest US {money0(psa_summary['latest_usd'])}.",
        "collection_value": f"Medium: Raw/A IQR avg US {money(raw_summary['average_usd'])}; PSA 10 IQR avg US {money(psa_summary['average_usd'])}.",
        "competitive_freq": "Low: use market condition and grade data for valuation.",
        "features": (
            f"Raw/A IQR: latest US {money0(raw_summary['latest_usd'])}, avg US {money(raw_summary['average_usd'])}, "
            f"{raw_summary['kept_count']}/{raw_summary['total_count']} records kept\n"
            f"PSA 10 IQR: latest US {money0(psa_summary['latest_usd'])}, avg US {money(psa_summary['average_usd'])}, "
            f"{psa_summary['kept_count']}/{psa_summary['total_count']} records kept"
        ),
        "ui_lang": "zh",
        "gemrate_stats": {},
    }


def inline_template_logo(html_doc: str, template_dir: Path) -> str:
    logo_path = template_dir / "logo.png"
    if not logo_path.exists():
        return html_doc
    try:
        logo_bytes = logo_path.read_bytes()
        logo_bytes = image_generator._strip_white_border_background_png(logo_bytes)
        logo_src = "data:image/png;base64," + base64.b64encode(logo_bytes).decode("utf-8")
        return html_doc.replace('src="logo.png"', f'src="{logo_src}"').replace("src='logo.png'", f"src='{logo_src}'")
    except Exception:
        return html_doc


def stat_card(label: str, value: str, sub: str = "") -> str:
    sub_html = (
        f'<span class="text-text-muted text-[11px] font-bold uppercase tracking-widest mt-1">{html.escape(sub)}</span>'
        if sub
        else ""
    )
    return f"""
        <div class="flex flex-col gap-1 p-5 rounded-xl bg-white/85 border border-white/90 shadow-[0_8px_20px_rgba(15,23,42,0.04)]">
            <span class="text-text-muted text-xs font-bold uppercase tracking-widest">{html.escape(label)}</span>
            <div class="text-4xl font-black text-text-main tracking-tight mt-1">{html.escape(value)}</div>{sub_html}
            <div class="w-full h-1 bg-gradient-to-r from-gray-300 to-transparent mt-3 rounded-full"></div>
        </div>"""


async def render_market_data_poster(
    out_path: Path,
    card_name: str,
    set_code: str,
    raw_records: list[dict[str, Any]],
    psa_records: list[dict[str, Any]],
    raw_summary: dict[str, Any],
    psa_summary: dict[str, Any],
    jpy_rate: float,
) -> None:
    template_dir = Path(image_generator.BASE_DIR) / "templates" / "v3"
    template_path = template_dir / "ai_studio_code.html"
    html_doc = template_path.read_text(encoding="utf-8")
    html_doc = inline_template_logo(html_doc, template_dir)
    html_doc = image_generator._localize_template_static(html_doc, "zh")
    for old, new in {
        "PriceCharting Trend": "Raw Price Trend",
        "Confirmed sales (USD)": "SNKRDUNK Raw/A sales (USD, IQR-filtered)",
        "SNKRDUNK Trend": "PSA 10 Trend",
        "Real-time marketplace (JPY &rarr; USD)": "SNKRDUNK PSA 10 sales (JPY &rarr; USD, IQR-filtered)",
        "Combined Transaction Report": "Raw vs PSA 10 Transaction Report",
        "Global Aggregated Market Stats": "IQR-Filtered Market Stats",
    }.items():
        html_doc = html_doc.replace(old, new)

    chart_line = "#1f6f8b"
    chart_class = "block w-full h-full object-fill"
    raw_chart = image_generator.create_premium_matplotlib_chart_b64(
        raw_records,
        color_line=chart_line,
        target_grade="Ungraded",
        is_jpy=False,
        theme="light",
        jpy_rate=jpy_rate,
    )
    psa_chart = image_generator.create_premium_matplotlib_chart_b64(
        psa_records,
        color_line=chart_line,
        target_grade="PSA 10",
        is_jpy=True,
        theme="light",
        jpy_rate=jpy_rate,
    )
    pc_charts_html = f'<div class="w-full h-[220px] mt-2 mb-1 flex items-end justify-center relative overflow-hidden"><img src="{raw_chart}" class="{chart_class}" /></div>'
    snkr_charts_html = f'<div class="w-full h-[220px] mt-2 mb-1 flex items-end justify-center overflow-hidden"><img src="{psa_chart}" class="{chart_class}" /></div>'

    table_outer_border = "border-slate-300/70"
    table_head_border = "border-slate-300/70"
    table_head_text = "text-slate-600"
    table_body_divider = "divide-slate-200/80"
    raw_rows = image_generator.generate_table_rows(raw_records, is_jpy=False, target_grade=None, theme="light", ui_lang="zh", max_rows=6, jpy_rate=jpy_rate)
    psa_rows = image_generator.generate_table_rows(psa_records, is_jpy=True, target_grade="PSA 10", theme="light", ui_lang="zh", max_rows=6, jpy_rate=jpy_rate)
    pc_table_html = f"""
                <div class="flex-1 glass-panel rounded-xl overflow-hidden p-3 border {table_outer_border}">
                    <table class="w-full text-left border-collapse">
                        <thead><tr class="border-b {table_head_border} text-[10px] font-black uppercase tracking-widest {table_head_text}">
                            <th class="p-3">Date (日期)</th><th class="p-3">Grade (狀態)</th><th class="p-3 text-right">Price (金額)</th>
                        </tr></thead>
                        <tbody class="text-sm divide-y {table_body_divider}">{raw_rows}</tbody>
                    </table>
                </div>"""
    snkr_table_html = f"""
                <div class="flex-1 glass-panel rounded-xl overflow-hidden p-3 border {table_outer_border}">
                    <table class="w-full text-left border-collapse">
                        <thead><tr class="border-b {table_head_border} text-[10px] font-black uppercase tracking-widest {table_head_text}">
                            <th class="p-3">Time (時間)</th><th class="p-3">Grade (狀態)</th><th class="p-3 text-right">Price (金額)</th>
                        </tr></thead>
                        <tbody class="text-sm divide-y {table_body_divider}">{psa_rows}</tbody>
                    </table>
                </div>"""
    filter_panel_html = f"""
        <h3 class="text-sm font-black uppercase tracking-[0.3em] text-text-muted mb-6 flex items-center gap-3">
            <span class="w-2 h-2 rounded-full bg-premium-gold shadow-[0_0_8px_rgba(212,175,55,0.8)] animate-pulse"></span>
            IQR Outlier Filter / 四分位過濾
        </h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            {stat_card('Raw kept', f"{raw_summary['kept_count']}/{raw_summary['total_count']}", f"removed {raw_summary['dropped_count']}")}
            {stat_card('Raw bounds', f"${raw_summary['lower_bound_usd']:.0f}-${raw_summary['upper_bound_usd']:.0f}", 'USD')}
            {stat_card('PSA 10 kept', f"{psa_summary['kept_count']}/{psa_summary['total_count']}", f"removed {psa_summary['dropped_count']}")}
            {stat_card('PSA 10 bounds', f"${psa_summary['lower_bound_usd']:.0f}-${psa_summary['upper_bound_usd']:.0f}", 'USD')}
        </div>"""

    replacements = {
        "{{ card_name }}": card_name,
        "{{ card_set }}": set_code,
        "{{ grade }}": "PSA 10",
        "{{ stat_1_title }}": "Raw/A Avg",
        "{{ stat_1_val }}": money(raw_summary["average_usd"]),
        "{{ stat_2_title }}": "Raw/A Latest",
        "{{ stat_2_val }}": money0(raw_summary["latest_usd"]),
        "{{ stat_3_title }}": "PSA 10 Avg",
        "{{ stat_3_val }}": money(psa_summary["average_usd"]),
        "{{ stat_4_title }}": "PSA 10 Latest",
        "{{ stat_4_val }}": money0(psa_summary["latest_usd"]),
        "{{ pc_charts_html }}": pc_charts_html,
        "{{ pc_table_html }}": pc_table_html,
        "{{ snkr_charts_html }}": snkr_charts_html,
        "{{ snkr_table_html }}": snkr_table_html,
        "{{ psa_stats_panel_html }}": filter_panel_html,
    }
    for key, value in replacements.items():
        core = key.replace("{{", "").replace("}}", "").strip()
        html_doc = re.sub(r"\{\{\s*" + re.escape(core) + r"\s*\}\}", str(value).replace("\\", r"\\"), html_doc)
    await image_generator._render_single_html_poster(str(html_doc), str(out_path), width=1200, height=900, device_scale_factor=2)


def encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


async def build_market_report(
    recognition: dict[str, Any],
    *,
    base_url: str | None = None,
    include_posters: bool = True,
    include_poster_base64: bool = False,
    jpy_rate: float = DEFAULT_JPY_RATE,
) -> dict[str, Any]:
    report_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:10]}"
    output_dir = REPORT_OUTPUT_DIR / report_id
    output_dir.mkdir(parents=True, exist_ok=True)

    product_id = parse_snkr_product_id(recognition)
    top = top_result(recognition)
    snkr = top.get("snkr") if isinstance(top.get("snkr"), dict) else {}
    snkr_url = snkr.get("url") or (f"https://snkrdunk.com/apparels/{product_id}" if product_id else "")
    if not product_id:
        markdown = "# MARKET REPORT GENERATED\n\nSNKR product_id missing; cannot fetch Raw/A or PSA 10 market prices."
        markdown_path = output_dir / "market_report.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        return {
            "status": "missing_product_id",
            "report_id": report_id,
            "recognition": recognition,
            "markdown": markdown,
            "files": {
                "markdown": {
                    "path": str(markdown_path),
                    "url": report_file_url(base_url, report_id, markdown_path),
                }
            },
        }

    history_payload = fetch_snkr_trading_histories(product_id)
    histories = history_payload.get("histories") if history_payload.get("status") == "ok" else []
    histories = histories if isinstance(histories, list) else []
    raw_bucket = iqr_filter(histories, "A")
    psa_bucket = iqr_filter(histories, "PSA 10")
    markdown = build_markdown_report(recognition, product_id, snkr_url, raw_bucket, psa_bucket, jpy_rate)
    markdown_path = output_dir / "market_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    raw_records = usd_records_for_poster(raw_bucket["kept_records"], "Raw / A")
    psa_records = snkr_records_for_poster(psa_bucket["kept_records"], "PSA 10", jpy_rate)
    files: dict[str, Any] = {
        "markdown": {
            "path": str(markdown_path),
            "url": report_file_url(base_url, report_id, markdown_path),
        }
    }
    poster_error = None
    if include_posters:
        try:
            card_data = card_data_from_recognition(recognition, raw_bucket["summary"], psa_bucket["summary"])
            profile_generated, _unused_data = await image_generator.generate_report(
                card_data,
                psa_records + snkr_records_for_poster(raw_bucket["kept_records"], "A", jpy_rate),
                [],
                out_dir=str(output_dir),
                template_version="v3",
                ui_lang="zh",
                jpy_rate=jpy_rate,
            )
            profile_path = output_dir / "poster_profile.png"
            shutil.copyfile(profile_generated, profile_path)
            data_path = output_dir / "poster_market_data.png"
            await render_market_data_poster(
                data_path,
                card_name=card_data["c_name"] or card_data["name"],
                set_code=card_data["set_code"],
                raw_records=raw_records,
                psa_records=psa_records,
                raw_summary=raw_bucket["summary"],
                psa_summary=psa_bucket["summary"],
                jpy_rate=jpy_rate,
            )
            for key, path in {"profile": profile_path, "market_data": data_path}.items():
                files[key] = {
                    "path": str(path),
                    "url": report_file_url(base_url, report_id, path),
                }
                if include_poster_base64:
                    files[key]["png_base64"] = encode_file_base64(path)
        except Exception as exc:  # noqa: BLE001
            poster_error = str(exc)

    return {
        "status": "ok",
        "report_id": report_id,
        "generated_at": utc_now_iso(),
        "recognition": recognition,
        "snkr": {
            "product_id": product_id,
            "url": snkr_url,
            "history_status": history_payload.get("status"),
            "history_url": history_payload.get("url"),
            "history_seconds": history_payload.get("seconds"),
        },
        "prices": {
            "source": "SNKRDUNK trading histories",
            "iqr_method": "Q1 - 1.5*IQR <= price <= Q3 + 1.5*IQR",
            "raw_A": raw_bucket["summary"],
            "psa_10": psa_bucket["summary"],
        },
        "markdown": markdown,
        "files": files,
        "poster_error": poster_error,
    }
