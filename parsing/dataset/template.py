from __future__ import annotations

import hashlib
import re


def normalize_to_template(content: str) -> str:
    template = (content or "-").strip()
    if not template or template == "-":
        return "-"

    template = re.sub(r"\[(\d+)\]", r"[<*>]", template)
    template = re.sub(
        r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|CONNECT|TRACE)\s+\S+\s+HTTP/\d+(?:\.\d+)?",
        lambda match: f"{match.group(1)} <*> HTTP/<*>",
        template,
    )
    template = re.sub(r"https?://[^\s\"']+", "<*>", template)
    template = re.sub(r"\bHTTP/\d+(?:\.\d+)?\b", "HTTP/<*>", template)
    template = re.sub(
        r"\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b",
        "<*>",
        template,
    )
    template = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<*>", template)
    template = re.sub(
        r"(?i)\b(session(?:id)?|csrf(?:token)?|nonce|request(?:[_-]?id)?|req(?:[_-]?id)?|token|jwt)\b\s*[:=]\s*([^\s,;\"']+)",
        lambda match: f"{match.group(1)}=<*>",
        template,
    )
    template = re.sub(r"([?&][A-Za-z0-9_.-]+)=([^&\s\"']+)", r"\1=<*>", template)
    template = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<*>", template)
    template = re.sub(r"(?i)\b(?:[0-9a-f]{1,4}:){3,}[0-9a-f:]*\b", "<*>", template)
    template = re.sub(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b", "<*>", template)
    template = re.sub(r"\blocalhost\b", "<*>", template, flags=re.IGNORECASE)
    template = re.sub(r"(?<!\w)(?:/[A-Za-z0-9._~:@%+-]+)+/?", "<*>", template)
    template = re.sub(r"\b[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]*\b", "<*>", template)
    template = re.sub(r"(?i)\b(port)\s+\d{1,5}\b", lambda match: f"{match.group(1)} <*>", template)
    template = re.sub(r":\d{2,5}\b", ":<*>", template)
    template = re.sub(r"\b[a-fA-F0-9]{16,}\b", "<*>", template)
    template = re.sub(r"\b[A-Za-z0-9_-]{20,}\b", "<*>", template)
    template = re.sub(r"\bssh\d+\b", "ssh<*>", template)
    template = re.sub(r"\b\d+\b", "<*>", template)
    template = re.sub(r"<\*>(?:\s*<\*>)+", "<*>", template)
    template = re.sub(r"\s+", " ", template).strip()
    return template or "-"


def make_event_id(template: str) -> str:
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:8]


def training_pre_normalize(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return "-"

    text = re.sub(r'("ts"\s*:\s*")([^"]*)(")', r"\1<*>\3", text, flags=re.IGNORECASE)
    text = re.sub(r'("ip"\s*:\s*")([^"]*)(")', r"\1<*>\3", text, flags=re.IGNORECASE)
    text = re.sub(r'("ua"\s*:\s*")([^"]*)(")', r"\1<*>\3", text, flags=re.IGNORECASE)
    text = re.sub(r'("referer"\s*:\s*")([^"]*)(")', r"\1<*>\3", text, flags=re.IGNORECASE)
    text = re.sub(
        r'"(?:Mozilla|python-requests|curl|Wget|Go-http-client|okhttp|sqlmap|nikto|Nmap)[^"]*"',
        '"<*>"',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?\b",
        "<*>",
        text,
    )
    return text


def finalize_training_template(template: str) -> str:
    out = template
    out = re.sub(r"<\*>-<\*>-\d{2}T\d{2}:<\*>:<\*>(?:\+<\*>:<\*>)?", "<*>", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out or "-"
