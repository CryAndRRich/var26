"""Tiện ích gọi LLM chat (dùng chung cho assertion/llm.py và mapping/rerank.py).

Lý do tồn tại: `tokenizer.apply_chat_template(..., return_tensors='pt')` trả về
KIỂU KHÁC NHAU tuỳ phiên bản transformers — có bản trả `torch.Tensor` (ids), bản
mới trả `BatchEncoding`/dict (có cả attention_mask). Bọc lại một chỗ để mọi
notebook chạy được bất kể phiên bản Kaggle cài sẵn.
"""
from __future__ import annotations


def chat_generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 16) -> str:
    """Sinh text cho 1 lượt chat, trả PHẦN MỚI (đã bỏ prompt). Greedy decode."""
    import torch

    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt")

    device = getattr(model, "device", None) or "cuda"
    if isinstance(enc, dict) or hasattr(enc, "input_ids"):   # BatchEncoding / dict
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if attn is not None else None
    else:                                                     # Tensor ids
        ids = enc.to(device)
        attn = None

    kw = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if attn is not None:
        kw["attention_mask"] = attn
    if getattr(tokenizer, "pad_token_id", None) is not None:
        kw["pad_token_id"] = tokenizer.pad_token_id

    with torch.no_grad():
        out = model.generate(ids, **kw)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
