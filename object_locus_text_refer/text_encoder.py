from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


MODEL_ID = "openai/clip-vit-base-patch32"


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_clip_text(cache_dir=None, provenance_path=None, revision=None):
    """Resolve immutable HF revision on first load, then persist and reuse it."""
    from transformers import CLIPTextModel, CLIPTokenizer
    provenance_file = Path(provenance_path) if provenance_path else None
    prior = json.loads(provenance_file.read_text()) if provenance_file and provenance_file.exists() else None
    pinned = revision or (prior or {}).get("revision")
    cache_snapshot = (Path(cache_dir) / "models--openai--clip-vit-base-patch32" / "snapshots" / pinned
                      if cache_dir and pinned else None)
    offline = bool(cache_snapshot and cache_snapshot.is_dir())
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir, revision=pinned,
                                               local_files_only=offline)
    model = CLIPTextModel.from_pretrained(MODEL_ID, cache_dir=cache_dir, revision=pinned,
                                          local_files_only=offline)
    resolved = getattr(model.config, "_commit_hash", None) or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if not resolved and cache_dir:
        ref = Path(cache_dir) / "models--openai--clip-vit-base-patch32" / "refs" / "main"
        if ref.exists(): resolved = ref.read_text().strip()
    if not resolved:
        raise RuntimeError("Hugging Face did not expose the resolved immutable commit revision")
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    model.requires_grad_(False)
    source = "huggingface_cache" if getattr(model, "name_or_path", "").startswith(str(cache_dir)) else "huggingface_hub"
    files = []
    for path in getattr(model, "hf_device_map", {}) if False else []: pass
    # Record cached files for tokenizer and model from their resolved snapshots.
    for obj in (model, tokenizer):
        name = getattr(obj, "name_or_path", None)
        if name and Path(name).exists(): files.append(Path(name))
    local_entries = []
    snapshot = Path(cache_dir) / "models--openai--clip-vit-base-patch32" / "snapshots" / resolved if cache_dir else None
    if snapshot and snapshot.is_dir():
        for path in sorted(p for p in snapshot.iterdir() if p.is_file()):
            local_entries.append({"path": str(path), "sha256": _sha(path)})
    provenance = {"model_id": MODEL_ID, "revision": resolved, "source": source,
                  "local_files": local_entries, "tokenizer_max_length": 77,
                  "hidden_dim": 512, "dtype": "float32", "text_encoder_frozen": True}
    if provenance_file:
        provenance_file.parent.mkdir(parents=True, exist_ok=True)
        provenance_file.write_text(json.dumps(provenance, indent=2) + "\n")
    return tokenizer, model, provenance


@torch.no_grad()
def encode_text(tokenizer, model, texts, device):
    tokens = tokenizer(texts, padding="max_length", truncation=True, max_length=77,
                       return_tensors="pt", return_attention_mask=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    features = output.last_hidden_state.float()
    return input_ids, attention_mask, features
