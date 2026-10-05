from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


class SIU3RReferDataset:
    """SIU3R preprocessed descriptions; instance IDs remain scene-local."""
    def __init__(self, refer_json, pair_json=None, split="train", load_arrays=None, seed=0):
        self.refer_path = Path(refer_json)
        self.records = json.loads(self.refer_path.read_text())
        self.pairs = json.loads(Path(pair_json).read_text()) if pair_json else None
        self.split = split
        self.load_arrays = load_arrays or self._load_context_arrays
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._access_counts = {}
        if split == "val" and self.pairs is None:
            raise ValueError("validation requires official val_refer_pair.json")
        self.items = self._expand_validation() if split == "val" else self._expand_train()

    def _expand_train(self):
        # Scene/object/text retain source iteration order; text choice is local RNG per item access.
        rows = []
        for scene, data in self.records.items():
            for object_id, obj in data["objects"].items():
                texts = obj.get("text", [])
                if isinstance(texts, str): texts = [texts]
                if texts: rows.append((scene, int(object_id), -1))
        return rows

    def _expand_validation(self):
        rows = []
        for pair in self.pairs:
            scene = pair["scene_name"]
            refs = self.records[scene]
            pair_objects = pair["context_objects"]
            object_ids = pair_objects if isinstance(pair_objects, list) else [pair_objects]
            # Current SIU3R val_refer_pair stores one expression in pair["texts"];
            # compatible variants may store per-object lists in the object records.
            for object_id in object_ids:
                obj = refs["objects"].get(str(object_id))
                if obj is None: continue
                texts = pair.get("texts", obj.get("text", []))
                if isinstance(texts, str): texts = [texts]
                for index, text in enumerate(texts):
                    rows.append((scene, tuple(pair["context_views_id"]), int(object_id), index, text))
        return rows

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        if self.split == "val": scene, frame_ids, object_id, text_index, pair_text = row
        else: scene, object_id, text_index = row; frame_ids = None
        obj = self.records[scene]["objects"][str(object_id)]
        texts = obj.get("text", [])
        if isinstance(texts, str): texts = [texts]
        if self.split == "train":
            # Per-access deterministic local stream; never touches process/global RNG.
            access = self._access_counts.get(int(index), 0)
            self._access_counts[int(index)] = access + 1
            rng = random.Random((self.seed << 40) ^ (int(index) << 16) ^ access ^ len(texts))
            text_index = rng.randrange(len(texts)) if texts else -1
        text = pair_text if self.split == "val" else (texts[text_index] if text_index >= 0 else "")
        arrays = self.load_arrays(scene, frame_ids)
        if self.split == "train":
            frame_ids = arrays.get("context_frame_ids")
        packed = arrays["packed_panoptic"]
        valid = arrays.get("valid_mask", packed >= 0).astype(bool)
        target = valid & ((packed % 1000) == int(object_id))
        return {"scene": scene, "context_frame_ids": frame_ids,
                "object_id": int(object_id), "text": text, "text_index": int(text_index),
                "context_target_mask": target, "context_valid_mask": valid,
                "arrays": arrays}

    @staticmethod
    def _load_context_arrays(scene, frame_ids):
        raise RuntimeError("provide load_arrays(scene, frame_ids) matching the frozen provider")
