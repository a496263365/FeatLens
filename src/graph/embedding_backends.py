import os
from typing import List, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F

from search.search_models.unixcoder import UniXcoder

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]


class BaseCodeEmbeddingBackend:
    """
    Unified interface for code search embedding backends.

    Both query and code embeddings are L2-normalized so that
    dot product equals cosine similarity.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_query(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def encode_code(self, code_list: List[str], batch_size: int = 32) -> torch.Tensor:
        raise NotImplementedError


class UniXcoderBackend(BaseCodeEmbeddingBackend):
    """
    Embedding backend based on UniXcoder.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        max_length: int = 512,
    ) -> None:
        super().__init__(device=device)
        if model_name is None:
            model_name = "microsoft/unixcoder-base"
        self.max_length = max_length

        try:
            self.model = UniXcoder(model_name)
        except Exception as e:  # pragma: no cover - defensive
            raise ImportError(
                "Unable to import UniXcoder. Ensure UniXcoder is installed / in PYTHONPATH."
            ) from e

        self.model.to(self.device)
        self.model.eval()

    def encode_query(self, text: str) -> torch.Tensor:
        tokens_ids = self.model.tokenize(
            [text], max_length=self.max_length, mode="<encoder-only>"
        )
        source_ids = torch.tensor(tokens_ids).to(self.device)
        with torch.no_grad():
            _, nl_embedding = self.model(source_ids)
            nl_embedding = F.normalize(nl_embedding, p=2, dim=1)
        return nl_embedding  # shape (1, dim) on self.device

    def encode_code(self, code_list: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(code_list), batch_size):
            batch = code_list[i : i + batch_size]
            batch_token_ids = []
            for code in batch:
                if not isinstance(code, str):
                    code = ""
                ids = self.model.tokenize(
                    [code], max_length=self.max_length, mode="<encoder-only>"
                )
                batch_token_ids.append(ids[0])

            if not batch_token_ids:
                continue

            max_len_in_batch = max(len(x) for x in batch_token_ids)
            padded_ids = []
            for x in batch_token_ids:
                padded_ids.append(
                    x + [self.model.config.pad_token_id] * (max_len_in_batch - len(x))
                )

            source_ids = torch.tensor(padded_ids).to(self.device)
            with torch.no_grad():
                _, code_embedding = self.model(source_ids)
                normed = F.normalize(code_embedding, p=2, dim=1)
                all_embs.append(normed)

        if len(all_embs) == 0:
            return torch.empty((0, 0), device=self.device)
        return torch.cat(all_embs, dim=0)  # (N, dim) on self.device


class BGECodeV1Backend(BaseCodeEmbeddingBackend):
    """
    Embedding backend based on BAAI/bge-code-v1 via SentenceTransformer.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-code-v1",
        device: Optional[torch.device] = None,
        instruction: Optional[str] = None,
        max_seq_length: int = 4096,
    ) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence_transformers is required for BGECodeV1Backend. "
                "Please install sentence-transformers."
            )
        super().__init__(device=device)
        self.instruction = 'Given a requirement description in text, retrieve code snippets that are relevant to the requirement.'
        self.max_seq_length = max_seq_length

        model_kwargs = {}
        if self.device.type == "cuda":
            # Use float16 on GPU for faster inference & lower memory.
            model_kwargs["torch_dtype"] = torch.float16

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=str(self.device),
            model_kwargs=model_kwargs,
        )
        # Cap sequence length so long code is truncated and CUDA OOM is avoided.
        self.model.max_seq_length = max_seq_length


    def _truncate_to_max_tokens(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncate text by character count so encode() does not trigger CUDA OOM.
        Uses ~3 chars/token (conservative for code) to avoid extra tokenizer calls and keep speed.
        """
        if not text or (max_tokens is not None and max_tokens <= 0):
            return text
        raw_limit = max_tokens if max_tokens is not None else self.max_seq_length
        limit = max(1, raw_limit - 2)  # leave room for [CLS]/[SEP]
        max_chars = limit * 3  # conservative: code ~3–4 chars per token
        return text[:max_chars] if len(text) > max_chars else text


    def _build_query_text(self, text: str) -> str:
        if self.instruction:
            return f"<instruct>{self.instruction}\n<query>{text}"
        return text

    def encode_query(self, text: str) -> torch.Tensor:
        query_text = self._build_query_text(text)
        emb = self.model.encode(
            [query_text],
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb.to(self.device)  # (1, dim)

    def encode_code(self, code_list: List[str], batch_size: int = 32) -> torch.Tensor:
        if not code_list:
            return torch.empty((0, 0), device=self.device)

        # Ensure all elements are strings and truncate long code to avoid CUDA OOM.
        norm_codes = []
        for c in code_list:
            if not isinstance(c, str):
                c = "" if c is None else str(c)
            norm_codes.append(self._truncate_to_max_tokens(c))

        embs = self.model.encode(
            norm_codes,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        if embs.dim() == 1:
            embs = embs.unsqueeze(0)
        return embs.to(self.device)  # (N, dim)


def create_embedding_backend(
    kind: str = "unixcoder",
    device: Optional[torch.device] = None,
    **kwargs,
) -> BaseCodeEmbeddingBackend:
    """
    Factory method to create an embedding backend.

    kind:
        - "unixcoder": UniXcoder-based backend (default)
        - "bge-code" / "bge_code" / "bge": BGE code v1 backend
    """
    normalized = (kind or "unixcoder").strip().lower()

    if normalized in {"unixcoder", "unix"}:
        return UniXcoderBackend(device=device, **kwargs)

    if normalized in {"bge-code", "bge_code", "bge"}:
        return BGECodeV1Backend(device=device, **kwargs)

    raise ValueError(f"Unknown embedding backend kind: {kind}")

