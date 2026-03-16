from __future__ import annotations

from typing import Any, Iterable, List, Optional

try:
    from .drain import LogParser, Logcluster, Node
    _DRAIN_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:
    LogParser = None  # type: ignore[assignment]
    Logcluster = Any  # type: ignore[assignment]
    Node = Any  # type: ignore[assignment]
    _DRAIN_IMPORT_ERROR = exc


if LogParser is not None:

    class DrainParser(LogParser):
        """Lightweight alias for the Drain LogParser used in inference."""

else:

    class DrainParser:  # type: ignore[no-redef]
        """Placeholder class when optional Drain dependencies are unavailable."""


TBIRD_LOG_FORMAT = "<Date> <Time> <Pid> <Level> <Component>: <Content>"
TBIRD_REGEX: List[str] = []
TBIRD_DRAIN_DEFAULTS = {
    "depth": 3,
    "st": 0.3,
    "maxChild": 1000,
    "keep_para": True,
}


def build_tbird_drain_parser(
    log_format: Optional[str] = None,
    regex: Optional[Iterable[str]] = None,
    depth: int = TBIRD_DRAIN_DEFAULTS["depth"],
    st: float = TBIRD_DRAIN_DEFAULTS["st"],
    maxChild: int = TBIRD_DRAIN_DEFAULTS["maxChild"],
    keep_para: bool = TBIRD_DRAIN_DEFAULTS["keep_para"],
) -> DrainParser:
    """Construct a Drain parser aligned with TBird defaults."""
    if _DRAIN_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Drain parser dependencies are not installed. "
            "Install optional dependencies (including pandas) to use build_tbird_drain_parser()."
        ) from _DRAIN_IMPORT_ERROR

    rex = list(regex) if regex is not None else []
    return DrainParser(
        log_format=log_format or TBIRD_LOG_FORMAT,
        depth=depth,
        st=st,
        maxChild=maxChild,
        rex=rex,
        keep_para=keep_para,
    )


__all__ = [
    "DrainParser",
    "Logcluster",
    "Node",
    "build_tbird_drain_parser",
    "TBIRD_LOG_FORMAT",
    "TBIRD_REGEX",
    "TBIRD_DRAIN_DEFAULTS",
]
