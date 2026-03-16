from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from training.bert_pytorch.model import BERT, BERTLog
from training.logbert_inference_client import LogBERTInferenceClient
from parsing import TBIRD_DRAIN_DEFAULTS, TBIRD_LOG_FORMAT, TBIRD_REGEX

from .utils import safe_torch_load


@dataclass(frozen=True)
class LogBERTPredictorConfig:
    vocab_size: int = 30522
    hidden: int = 768
    n_layers: int = 12
    attn_heads: int = 12
    seq_len: int = 512
    device: str = "auto"
    drain_log_format: Optional[str] = TBIRD_LOG_FORMAT
    drain_regex: Optional[List[str]] = None
    drain_depth: Optional[int] = TBIRD_DRAIN_DEFAULTS["depth"]
    drain_st: Optional[float] = TBIRD_DRAIN_DEFAULTS["st"]
    drain_max_child: Optional[int] = TBIRD_DRAIN_DEFAULTS["maxChild"]
    num_candidates: int = 15
    anomaly_threshold: float = 0.5
    batch_size: int = 512


class LogBERTPredictor:
    """Ready-to-use LogBERT predictor for raw log lines."""

    def __init__(
        self,
        state_path: Path | str,
        vocab_path: Path | str,
        center_path: Optional[Path | str] = None,
        config: Optional[LogBERTPredictorConfig] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.config = config or LogBERTPredictorConfig()

        if model is None:
            model = self._build_model(self.config)

        self.client = LogBERTInferenceClient(
            model=model,
            state_path=Path(state_path),
            vocab_path=Path(vocab_path),
            drain_log_format=self.config.drain_log_format,
            drain_regex=self._resolve_regex(self.config.drain_regex),
            seq_len=self.config.seq_len,
            device=self.config.device,
            drain_depth=self.config.drain_depth,
            drain_st=self.config.drain_st,
            drain_max_child=self.config.drain_max_child,
            num_candidates=self.config.num_candidates,
            anomaly_threshold=self.config.anomaly_threshold,
            batch_size=self.config.batch_size,
        )

        self.center: Optional[Dict[str, Any] | torch.Tensor] = None
        if center_path is not None:
            device = self._resolve_device(self.config.device)
            self.center = safe_torch_load(str(center_path), device)

    def predict(self, raw_logs: Iterable[str], group_by_minute: bool = True) -> List[Dict[str, Any]]:
        """Return anomaly scores and flags for each log line."""
        lines = list(raw_logs)
        if not lines:
            return []
        if not group_by_minute:
            return self.client.predict(lines)
        results: List[Dict[str, Any]] = []
        for group in self._group_by_minute(lines):
            results.extend(self.client.predict(group))
        return results

    @staticmethod
    def _build_model(config: LogBERTPredictorConfig) -> BERTLog:
        bert = BERT(
            vocab_size=config.vocab_size,
            hidden=config.hidden,
            n_layers=config.n_layers,
            attn_heads=config.attn_heads,
        )
        return BERTLog(bert=bert, vocab_size=config.vocab_size)

    @staticmethod
    def _resolve_regex(regex: Optional[List[str]]) -> Optional[List[str]]:
        if regex is not None:
            return regex
        if TBIRD_REGEX:
            return list(TBIRD_REGEX)
        return None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _group_by_minute(lines: Sequence[str]) -> List[List[str]]:
        groups: List[List[str]] = []
        current: List[str] = []
        current_key: Optional[str] = None
        for line in lines:
            minute_key = LogBERTPredictor._extract_minute_key(line)
            if minute_key is None:
                if current:
                    groups.append(current)
                    current = []
                groups.append([line])
                current_key = None
                continue
            if current_key is None or minute_key == current_key:
                current.append(line)
                current_key = minute_key
                continue
            groups.append(current)
            current = [line]
            current_key = minute_key
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _extract_minute_key(line: str) -> Optional[str]:
        match = re.search(r"(\d{4}[.-]\d{2}[.-]\d{2})[ T](\d{2}):(\d{2})", line)
        if not match:
            return None
        date, hour, minute = match.group(1), match.group(2), match.group(3)
        return f"{date} {hour}:{minute}"


__all__ = ["LogBERTPredictor", "LogBERTPredictorConfig"]
