from __future__ import annotations

import re


ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".log"}
RAW_TEXT_KEYS = (
    "log",
    "message",
    "text",
    "raw",
    "line",
    "content",
    "event",
    "record",
)
LABEL_KEYS = ("label", "anomaly", "is_anomaly", "target", "class", "y")
MONTH_TO_NUM = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}
DATASET_TITLE_OVERRIDES = {
    "vulhab": "Vulhab",
    "vulhub": "Vulhab",
    "dvwa": "DVWA",
}
TARGET_ALIASES = {
    "vulhab": {"vulhab", "vulhub"},
}

STRUCTURED_COLUMNS = [
    "LineId",
    "Label",
    "Id",
    "Date",
    "Admin",
    "Month",
    "Day",
    "Time",
    "AdminAddr",
    "Content",
    "EventId",
    "EventTemplate",
]
TEMPLATE_COLUMNS = ["EventId", "EventTemplate", "Occurrences"]

APACHE_RE = re.compile(
    r"^(?P<AdminAddr>\S+)\s+\S+\s+(?P<Admin>\S+)\s+\[(?P<Day>\d{1,2})/"
    r"(?P<Month>[A-Za-z]{3})/(?P<Year>\d{4}):"
    r"(?P<Time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+[+\-]\d{4}\]\s*(?P<Content>.*)$"
)
BRACKET_DATE_RE = re.compile(
    r"^\[(?P<Day>\d{1,2})/(?P<Month>[A-Za-z]{3})/(?P<Year>\d{4}):"
    r"(?P<Time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+[+\-]\d{4}\]\s*(?P<Content>.*)$"
)
SYSLOG_RE = re.compile(
    r"^(?P<Month>[A-Za-z]{3})\s+(?P<Day>\d{1,2})\s+"
    r"(?P<Time>\d{2}:\d{2}:\d{2})\s+(?P<Admin>\S+)\s*(?P<Content>.*)$"
)
ISO_PREFIX_RE = re.compile(
    r"^(?P<Date>\d{4}-\d{2}-\d{2})[T\s]+"
    r"(?P<Time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*(?P<Content>.*)$"
)
