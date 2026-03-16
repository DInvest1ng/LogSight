"""Production-ready inference client for LogBERT on TBird.

The client expects that the LogBERT architecture and the trained weights
already exist locally. It handles log parsing (Drain), tokenization with
the training vocabulary, batching, device placement, and anomaly scoring
based on masked-language-model logits.
"""
from __future__ import annotations

import hashlib
import importlib
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from parsing import (
    DrainParser,
    Logcluster,
    Node,
    build_tbird_drain_parser,
    TBIRD_LOG_FORMAT,
    TBIRD_REGEX,
    TBIRD_DRAIN_DEFAULTS,
)  # type: ignore


class LogBERTInferenceClient:
    """Inference-only pipeline for a pretrained LogBERT model.

    Parameters
    ----------
    model : torch.nn.Module
        Instantiated LogBERT model with the same architecture used in training.
    state_path : str or Path
        Path to the checkpoint (e.g., ``logbert/output/tbird/bert/best_bert.pth``).
    vocab_path : str or Path
        Path to the pickled vocabulary produced during training (e.g.,
        ``logbert/output/tbird/vocab.pkl``).
    drain_log_format : str, optional
        The log format string used during training. Defaults to the TBird
        value from ``TBird/data_process.py`` in the original repo.
    drain_regex : list[str], optional
        Regex list used to mask dynamic fields during parsing; must match the
        training setup. Defaults to the TBird regex list.
    seq_len : int, optional
        Fixed sequence length expected by the model. Default is 512 for TBird.
    device : {"cpu", "cuda", "auto"}, optional
        Target device. "auto" picks CUDA when available. Default "cpu".
    drain_depth : int, optional
        Drain tree depth; defaults to TBird (3).
    drain_st : float, optional
        Drain similarity threshold; defaults to TBird (0.3).
    drain_max_child : int, optional
        Max children per node for Drain; defaults to TBird (1000).
    num_candidates : int, optional
        Top-K threshold for anomaly flagging (same as training hyperparameter).
    anomaly_threshold : float, optional
        Flag anomaly if ``1 - p(true_id)`` exceeds this value. Default 0.5.
    pad_token : str, optional
        Token string used for padding in the vocabulary. Default "<pad>".
    unk_token : str, optional
        Token string used for unknown templates. Default "<unk>".
    batch_size : int, optional
        Inference batch size (number of sequences). Default 512 as in TBird.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        state_path: Path | str,
        vocab_path: Path | str,
        drain_log_format: Optional[str] = None,
        drain_regex: Optional[List[str]] = None,
        seq_len: int = 512,
        device: str = "cpu",
        drain_depth: Optional[int] = None,
        drain_st: Optional[float] = None,
        drain_max_child: Optional[int] = None,
        num_candidates: int = 15,
        anomaly_threshold: float = 0.5,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        batch_size: int = 512,
    ) -> None:
        self.device = self._resolve_device(device)
        self.seq_len = seq_len
        self.num_candidates = num_candidates
        self.anomaly_threshold = anomaly_threshold
        self.batch_size = batch_size

        self.model = model.to(self.device)
        self._install_legacy_module_aliases()
        try:
            state = torch.load(state_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location=self.device)
        except Exception:
            state = torch.load(state_path, map_location=self.device, weights_only=False)
        if isinstance(state, torch.nn.Module):
            self.model = state.to(self.device)
            self._patch_loaded_model(self.model)
        else:
            if hasattr(state, "state_dict"):
                state = state.state_dict()
            if isinstance(state, dict) and any(k in state for k in ("state_dict", "model_state_dict")):
                state = state.get("state_dict", state.get("model_state_dict", state))
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.token_to_id = self._load_vocab(vocab_path)
        self.pad_id = self._get_required_token_id(pad_token)
        self.unk_id = self._get_required_token_id(unk_token, allow_missing=True)

        self.parser = self._init_drain_parser(
            drain_log_format,
            drain_regex,
            drain_depth,
            drain_st,
            drain_max_child,
        )
        self._drain_root = Node()
        self._parse_cache: Dict[str, Tuple[str, str]] = {}

    @staticmethod
    def _install_legacy_module_aliases() -> None:
        """
        Keep backward compatibility with checkpoints pickled under old module
        paths like ``bert_pytorch.model.log_model.BERTLog``.
        """
        alias_map = {
            "bert_pytorch": "training.bert_pytorch",
            "bert_pytorch.model": "training.bert_pytorch.model",
            "bert_pytorch.model.bert": "training.bert_pytorch.model.bert",
            "bert_pytorch.model.log_model": "training.bert_pytorch.model.log_model",
            "bert_pytorch.model.language_model": "training.bert_pytorch.model.language_model",
            "bert_pytorch.model.transformer": "training.bert_pytorch.model.transformer",
            "bert_pytorch.model.attention": "training.bert_pytorch.model.attention",
            "bert_pytorch.model.attention.multi_head": "training.bert_pytorch.model.attention.multi_head",
            "bert_pytorch.model.attention.single": "training.bert_pytorch.model.attention.single",
            "bert_pytorch.model.embedding": "training.bert_pytorch.model.embedding",
            "bert_pytorch.model.embedding.bert": "training.bert_pytorch.model.embedding.bert",
            "bert_pytorch.model.embedding.position": "training.bert_pytorch.model.embedding.position",
            "bert_pytorch.model.embedding.segment": "training.bert_pytorch.model.embedding.segment",
            "bert_pytorch.model.embedding.time_embed": "training.bert_pytorch.model.embedding.time_embed",
            "bert_pytorch.model.embedding.token": "training.bert_pytorch.model.embedding.token",
            "bert_pytorch.model.utils": "training.bert_pytorch.model.utils",
            "bert_pytorch.model.utils.feed_forward": "training.bert_pytorch.model.utils.feed_forward",
            "bert_pytorch.model.utils.gelu": "training.bert_pytorch.model.utils.gelu",
            "bert_pytorch.model.utils.layer_norm": "training.bert_pytorch.model.utils.layer_norm",
            "bert_pytorch.model.utils.sublayer": "training.bert_pytorch.model.utils.sublayer",
            "bert_pytorch.dataset": "training.bert_pytorch.dataset",
            "bert_pytorch.trainer": "training.bert_pytorch.trainer",
        }
        for legacy_name, new_name in alias_map.items():
            if legacy_name in sys.modules:
                continue
            sys.modules[legacy_name] = importlib.import_module(new_name)

    def predict(self, raw_logs: List[str], batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run batched inference on raw log lines.

        Returns one dictionary per input log line with the original text,
        resolved template ID, anomaly score, and a boolean flag.
        """

        if not raw_logs:
            return []

        batch_size = batch_size or self.batch_size

        # 1) Parse & map to token IDs
        template_ids: List[str] = []
        token_ids: List[int] = []
        for line in raw_logs:
            template_id, template_str = self._parse_to_template(line)
            template_ids.append(template_id)
            token_ids.append(self._template_to_token(template_id, template_str))

        # 2) Build fixed-length sequences with padding
        sequences, starts = self._build_sequences(token_ids)

        # 3) Allocate result containers aligned to original lines
        scores: List[float] = [0.0 for _ in raw_logs]
        flags: List[bool] = [False for _ in raw_logs]

        # 4) Model forward pass in batches
        batches = [
            (starts[i : i + batch_size], sequences[i : i + batch_size])
            for i in range(0, len(sequences), batch_size)
        ]

        with torch.inference_mode():
            for start_chunk, seq_chunk in batches:
                input_ids = torch.stack(seq_chunk).to(self.device)
                segment_labels = torch.zeros_like(input_ids)  # single segment
                logits = self._forward_logits(input_ids, segment_labels)
                probs = torch.softmax(logits, dim=-1)
                topk = torch.topk(probs, k=self.num_candidates, dim=-1).indices

                for row, seq_start in enumerate(start_chunk):
                    for pos in range(self.seq_len):
                        global_idx = seq_start + pos
                        if global_idx >= len(raw_logs):
                            break
                        true_id = input_ids[row, pos].item()
                        # Skip padded positions
                        if true_id == self.pad_id:
                            continue
                        p_true = probs[row, pos, true_id].item()
                        score = 1.0 - p_true
                        is_anom = true_id not in topk[row, pos].tolist() or score > self.anomaly_threshold
                        scores[global_idx] = score
                        flags[global_idx] = is_anom

        # 5) Compose output aligned to input order
        results = []
        for line, template_id, score, flag in zip(raw_logs, template_ids, scores, flags):
            results.append(
                {
                    "log": line,
                    "template_id": template_id,
                    "anomaly_score": score,
                    "is_anomaly": bool(flag),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_drain_parser(
        self,
        log_format: Optional[str],
        regex: Optional[List[str]],
        depth: Optional[int],
        st: Optional[float],
        max_child: Optional[int],
    ) -> DrainParser:
        """
        Build a Drain parser with TBird defaults unless explicitly overridden.
        """
        params = {
            "depth": depth if depth is not None else TBIRD_DRAIN_DEFAULTS["depth"],
            "st": st if st is not None else TBIRD_DRAIN_DEFAULTS["st"],
            "maxChild": max_child if max_child is not None else TBIRD_DRAIN_DEFAULTS["maxChild"],
            "keep_para": TBIRD_DRAIN_DEFAULTS["keep_para"],
        }
        return build_tbird_drain_parser(
            log_format=log_format or TBIRD_LOG_FORMAT,
            regex=regex or TBIRD_REGEX,
            **params,
        )

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_vocab(self, vocab_path: Path | str) -> Dict[str, int]:
        path = Path(vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        with path.open("rb") as f:
            vocab_obj = pickle.load(f)
        if isinstance(vocab_obj, dict):
            for key in ("token2idx", "event2idx", "vocab"):
                if key in vocab_obj and isinstance(vocab_obj[key], dict):
                    return vocab_obj[key]
        if isinstance(vocab_obj, dict):
            return {str(k): int(v) for k, v in vocab_obj.items()}
        if hasattr(vocab_obj, "stoi") and isinstance(vocab_obj.stoi, dict):
            return {str(k): int(v) for k, v in vocab_obj.stoi.items()}
        raise ValueError("Unsupported vocab format; expected a dict-like mapping")

    def _get_required_token_id(self, token: str, allow_missing: bool = False) -> int:
        if token in self.token_to_id:
            return int(self.token_to_id[token])
        if allow_missing:
            # fall back to padding idx if available; else -1
            return int(self.token_to_id.get("<pad>", -1))
        raise KeyError(f"Token '{token}' is missing from the vocabulary")

    def _parse_to_template(self, line: str) -> Tuple[str, str]:
        if line in self._parse_cache:
            return self._parse_cache[line]

        tokens = self.parser.preprocess(line).strip().split()
        match = self.parser.treeSearch(self._drain_root, tokens)
        if match is None:
            cluster = Logcluster(logTemplate=tokens, logIDL=[0])
            self.parser.addSeqToPrefixTree(self._drain_root, cluster)
            template_tokens = cluster.logTemplate
        else:
            template_tokens = match.logTemplate
        template_str = " ".join(template_tokens)
        template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[:8]
        self._parse_cache[line] = (template_id, template_str)
        return template_id, template_str

    def _template_to_token(self, template_id: str, template_str: str) -> int:
        if template_id in self.token_to_id:
            return int(self.token_to_id[template_id])
        if template_str in self.token_to_id:
            return int(self.token_to_id[template_str])
        if self.unk_id >= 0:
            return self.unk_id
        raise KeyError(f"Template '{template_str}' not found in vocabulary and no <unk> provided")

    def _build_sequences(self, token_ids: List[int]) -> Tuple[List[torch.Tensor], List[int]]:
        sequences: List[torch.Tensor] = []
        starts: List[int] = []
        for start in range(0, len(token_ids), self.seq_len):
            seq = token_ids[start : start + self.seq_len]
            if len(seq) < self.seq_len:
                seq = seq + [self.pad_id] * (self.seq_len - len(seq))
            sequences.append(torch.tensor(seq, dtype=torch.long))
            starts.append(start)
        return sequences, starts

    def _forward_logits(self, input_ids: torch.Tensor, segment_labels: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids, segment_labels)
        if isinstance(outputs, dict) and "logkey_output" in outputs:
            logits = outputs["logkey_output"]
        elif isinstance(outputs, (list, tuple)):
            logits = outputs[0]
        else:
            logits = outputs
        if logits.dim() != 3:
            raise ValueError("Model output must be [batch, seq_len, vocab_size] logits")
        return logits

    def _patch_loaded_model(self, model: torch.nn.Module) -> None:
        """Attach missing submodules for pickled checkpoints that omit non-parameter modules."""
        try:
            import torch.nn as nn
            from training.bert_pytorch.model.attention import MultiHeadedAttention
            from training.bert_pytorch.model.language_model import MaskedLanguageModel
            from training.bert_pytorch.model.log_model import MaskedLogModel
            from training.bert_pytorch.model.utils.feed_forward import PositionwiseFeedForward
            from training.bert_pytorch.model.utils.gelu import GELU
            from training.bert_pytorch.model.utils.sublayer import SublayerConnection
        except Exception:
            return

        for m in model.modules():
            if isinstance(m, PositionwiseFeedForward):
                if not hasattr(m, "activation"):
                    m.activation = GELU()
                if not hasattr(m, "dropout"):
                    m.dropout = nn.Dropout(0.1)
            if isinstance(m, MultiHeadedAttention):
                if not hasattr(m, "dropout"):
                    m.dropout = nn.Dropout(0.1)
            if isinstance(m, SublayerConnection):
                if not hasattr(m, "dropout"):
                    m.dropout = nn.Dropout(0.1)
            if isinstance(m, (MaskedLanguageModel, MaskedLogModel)):
                if not hasattr(m, "softmax"):
                    m.softmax = nn.LogSoftmax(dim=-1)


__all__ = ["LogBERTInferenceClient"]
