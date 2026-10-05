from __future__ import annotations

import json
from pathlib import Path
import torch


class SIU3RReferDataset:
    """SIU3R preprocessed descriptions; instance IDs remain scene-local."""
    def __init__(self, refer_json, pair_json=None, split="train", load_arrays=None, seed=0):
        self.refer_path = Path(refer_json)
        self.records = json.loads(self.refer_path.read_text())
        self.pairs = json.loads(Path(pair_json).read_text()) if pair_json else None
        self.split = split
        self.load_arrays = load_arrays or self._load_context_arrays
        if split != "val":
            raise ValueError("expression dataset is validation-only; training samples must start from provider context")
        if self.pairs is None:
            raise ValueError("validation requires official val_refer_pair.json")
        self.items = self._expand_validation()

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
        scene, frame_ids, object_id, text_index, pair_text = self.items[index]
        obj = self.records[scene]["objects"][str(object_id)]
        texts = obj.get("text", [])
        if isinstance(texts, str): texts = [texts]
        text = pair_text
        arrays = self.load_arrays(scene, frame_ids)
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


class NoVisibleReferent(ValueError):
    """A legal context pair has no described, annotated visible thing object."""


def normalize_scene_object_ids(values):
    return {int(value) for value in values}


def nonempty_texts(obj):
    texts = obj.get("text", [])
    if isinstance(texts, str):
        texts = [texts]
    return [(index, text) for index, text in enumerate(texts)
            if isinstance(text, str) and text.strip()]


def sample_context_referent(refer_data, scene, batch, rng):
    """Select only after a real provider context and its valid GT are available."""
    if batch["semantic_label_all"].shape[0] != 1:
        raise ValueError("text refer data interface fixes batch size to one")
    frames = [int(value) for value in batch["frame_ids"][0, :2].detach().cpu().tolist()]
    scene_data = refer_data[scene]
    frame_objects = set()
    for frame in frames:
        frame_objects |= normalize_scene_object_ids(scene_data["frame2object"].get(str(frame), []))
    objects_by_id = {int(key): value for key, value in scene_data["objects"].items()}
    described = {oid for oid, obj in objects_by_id.items() if nonempty_texts(obj)}
    sem = batch["semantic_label_all"][0, :2].long()
    ins = batch["instance_label_all"][0, :2].long()
    valid = (sem >= 0) & (sem <= 19) & ((sem < 2) | (ins > 0))
    visible = set(torch.unique(ins[valid & (sem >= 2) & (ins > 0)]).detach().cpu().tolist())
    candidates = sorted(frame_objects & described & visible)
    if not candidates:
        raise NoVisibleReferent(
            f"scene={scene} context_frames={frames} has no object in frame2object ∩ described objects ∩ valid visible thing IDs"
        )
    object_id = int(rng.choice(candidates))
    texts = nonempty_texts(objects_by_id[object_id])
    text_index, text = rng.choice(texts)
    gt = valid & (ins == object_id)
    return {
        "scene": str(scene),
        "context_frame_ids": frames,
        "object_id": object_id,
        "text": text,
        "text_index": int(text_index),
        "candidate_object_ids": candidates,
        "context_target_mask": gt,
        "context_valid_mask": valid,
        "batch": batch,
    }


def choose_context_candidate(candidates, rng):
    """Choose scene only after caller has a legal provider context candidate set."""
    ordered = sorted(set(candidates))
    if not ordered:
        raise NoVisibleReferent("no train scenes have a legal provider context")
    return rng.choice(ordered)
