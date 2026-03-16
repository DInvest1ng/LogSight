from __future__ import annotations

from typing import Any

from .constants import APACHE_RE, BRACKET_DATE_RE, ISO_PREFIX_RE, MONTH_TO_NUM, SYSLOG_RE


def parse_raw_line(raw_line: str) -> dict[str, str]:
    base = {
        "Date": "-",
        "Admin": "-",
        "Month": "-",
        "Day": "-",
        "Time": "-",
        "AdminAddr": "-",
        "Content": raw_line.strip() if raw_line else "-",
    }
    line = raw_line.strip()
    if not line:
        return base

    match = APACHE_RE.match(line)
    if match:
        month = _normalize_month(match.group("Month"))
        day = _normalize_day(match.group("Day"))
        year = match.group("Year")
        base.update(
            {
                "Date": _build_date(year, month, day),
                "Admin": _sanitize_field(match.group("Admin")),
                "Month": month,
                "Day": day,
                "Time": _sanitize_field(match.group("Time")),
                "AdminAddr": _sanitize_field(match.group("AdminAddr")),
                "Content": _sanitize_field(match.group("Content")),
            }
        )
        return base

    match = BRACKET_DATE_RE.match(line)
    if match:
        month = _normalize_month(match.group("Month"))
        day = _normalize_day(match.group("Day"))
        year = match.group("Year")
        base.update(
            {
                "Date": _build_date(year, month, day),
                "Month": month,
                "Day": day,
                "Time": _sanitize_field(match.group("Time")),
                "Content": _sanitize_field(match.group("Content")),
            }
        )
        return base

    match = SYSLOG_RE.match(line)
    if match:
        month = _normalize_month(match.group("Month"))
        day = _normalize_day(match.group("Day"))
        base.update(
            {
                "Month": month,
                "Day": day,
                "Time": _sanitize_field(match.group("Time")),
                "Admin": _sanitize_field(match.group("Admin")),
                "Content": _sanitize_field(match.group("Content")),
            }
        )
        return base

    match = ISO_PREFIX_RE.match(line)
    if match:
        base.update(
            {
                "Date": _sanitize_field(match.group("Date")),
                "Time": _sanitize_field(match.group("Time")),
                "Content": _sanitize_field(match.group("Content")),
            }
        )
        return base

    return base


def extract_content(parsed_line: dict[str, str]) -> str:
    content = parsed_line.get("Content", "-")
    if not content or content == "-":
        return "-"
    return content.strip()


def _sanitize_field(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _normalize_month(month_value: str) -> str:
    month_clean = _sanitize_field(month_value)
    return MONTH_TO_NUM.get(month_clean[:3].title(), month_clean)


def _normalize_day(day_value: str) -> str:
    day_clean = _sanitize_field(day_value)
    if day_clean.isdigit():
        return day_clean.zfill(2)
    return day_clean


def _build_date(year: str, month: str, day: str) -> str:
    if year.isdigit() and month.isdigit() and day.isdigit():
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return "-"
