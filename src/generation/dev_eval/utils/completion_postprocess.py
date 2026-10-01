import re


def extract_code_from_markdown(text: str) -> str:
    if not text:
        return ""
    pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n([\s\S]*?)\n?```", re.MULTILINE)
    matches = pattern.findall(text)
    if matches:
        candidate = max(matches, key=lambda s: len(s.strip()))
        return candidate.strip("\n")
    return text.strip("\n")


# Print content for debugging
def preview_text(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, total_chars={len(text)})"


def remove_triple_quoted_block_containing_anchor(text: str, anchor: str) -> str:
    if not text or not anchor:
        return text
    out = text
    while True:
        anchor_idx = out.find(anchor)
        if anchor_idx == -1:
            break
        start_sq = out.rfind("'''", 0, anchor_idx)
        start_dq = out.rfind('"""', 0, anchor_idx)
        start = max(start_sq, start_dq)
        if start == -1:
            break
        delimiter = out[start : start + 3]
        end = out.find(delimiter, anchor_idx)
        if end == -1:
            break
        out = out[:start] + out[end + 3 :]
    return out


# Extract the completed code from the model response and remove possible function signatures or multiline comments
def keep_only_completion(
    completion_text: str,
    *,
    signature: str,
    requirement_comment: str,
    requirement_text: str,
) -> str:
    if not completion_text:
        return ""

    text = completion_text.replace("\r\n", "\n").replace("\r", "\n")

    # If the completion directly contains the requirement comment, remove it
    req_comment_norm = (
        (requirement_comment or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    )
    if req_comment_norm:
        text = text.replace(req_comment_norm, "")

    # Remove possible multiline comments
    req_text_norm = (requirement_text or "").strip()
    if req_text_norm:
        anchor = req_text_norm[:30]
        text = remove_triple_quoted_block_containing_anchor(text, anchor)

    # Remove possible function signatures
    sig_norm = (signature or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if sig_norm:
        sig_idx = text.rfind(sig_norm)
        if sig_idx != -1:
            text = text[sig_idx + len(sig_norm) :]

    return text.lstrip("\n")

