"""Tiện ích gọi LLM chat (dùng chung cho assertion/llm.py và mapping/rerank.py).

Lý do tồn tại:
1. `tokenizer.apply_chat_template(..., return_tensors='pt')` trả về KIỂU KHÁC NHAU tuỳ
   phiên bản transformers — bản trả `torch.Tensor` (ids), bản mới trả `BatchEncoding`/dict.
2. Greedy decode nhưng generation_config của model (vd Qwen) có sẵn temperature/top_p/top_k
   -> transformers cảnh báo "generation flags are not valid and may be ignored". Dọn 1 lần.
3. Generate THEO BATCH: nhanh hơn nhiều lần so với batch=1 khi phải chấm hàng nghìn mention.
"""
from __future__ import annotations


def silence_sampling_flags(model) -> None:
    """Bỏ temperature/top_p/top_k khỏi generation_config (ta luôn greedy).

    Tránh warning 'The following generation flags are not valid and may be ignored'.
    Gọi 1 lần sau khi nạp model là đủ (hàm này idempotent)."""
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    for k in ("temperature", "top_p", "top_k"):
        if getattr(cfg, k, None) is not None:
            setattr(cfg, k, None)


def _device_of(model):
    return getattr(model, "device", None) or "cuda"


def chat_generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 16) -> str:
    """Sinh text cho 1 lượt chat, trả PHẦN MỚI (đã bỏ prompt). Greedy decode."""
    import torch

    silence_sampling_flags(model)
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt")

    device = _device_of(model)
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


def chat_generate_batch(model, tokenizer, batch_messages: list[list[dict]],
                        max_new_tokens: int = 16) -> list[str]:
    """Generate cho NHIỀU hội thoại một lượt -> list text mới (cùng thứ tự).

    Decoder-only nên PHẢI pad BÊN TRÁI, nếu không phần sinh ra bị lệch/rác.
    Trả về list cùng độ dài batch_messages.
    """
    import torch

    if not batch_messages:
        return []
    silence_sampling_flags(model)

    texts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
             for m in batch_messages]

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"          # bắt buộc cho generate theo batch
    try:
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tokenizer.padding_side = old_side

    device = _device_of(model)
    enc = {k: v.to(device) for k, v in enc.items()}
    n_prompt = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(out[:, n_prompt:], skip_special_tokens=True)
