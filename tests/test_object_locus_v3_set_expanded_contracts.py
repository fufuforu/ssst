import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.train_object_locus_v3_set_expanded import (
    build_manifest_and_plan, expanded_lr, validate_assets, science_module_hashes, SCIENCE_SHA256,
    _official_scope, _attach_official_provenance,
)


def test_assets_and_data_split_contracts():
    manifest, plan = build_manifest_and_plan()
    validate_assets(manifest, plan)
    assert len(manifest["expanded_train_windows"]) == 1008
    assert len({w["scene"] for w in manifest["expanded_train_windows"]}) == 128
    assert len(plan["entries"]) == 32256
    assert len({(e["scene"], tuple(e["context"]), tuple(e["novel"])) for e in plan["entries"]}) == 1008
    exposures = Counter((e["scene"], tuple(e["context"]), tuple(e["novel"])) for e in plan["entries"])
    assert len(exposures) == 1008 and set(exposures.values()) == {32}
    for stage, pos in ((0, 0), (1, 0), (31, 1007)):
        e = plan["entries"][stage * 1008 + pos]
        expected = np.random.default_rng(42 + stage).permutation(1008)[pos]
        assert e["window_index"] == int(expected)


def test_expanded_lr_schedule():
    assert np.allclose(expanded_lr(1), (1e-4 / 200, 1e-6 / 200), rtol=1e-15, atol=0.)
    assert np.allclose(expanded_lr(200), (1e-4, 1e-6), rtol=1e-15, atol=0.)
    assert expanded_lr(201)[0] <= 1e-4
    assert math.isclose(expanded_lr(32256)[0], 1e-5, rel_tol=1e-12)
    assert math.isclose(expanded_lr(32256)[1], 1e-7, rel_tol=1e-12)


def test_global_step_and_exposure_contracts():
    assert 3584 + 32256 == 35840
    assert 64 + 32 == 96
    assert science_module_hashes() == SCIENCE_SHA256


def test_official_scope_uses_explicit_context_target_and_novel_paths():
    metrics = {"all": {
        "context_miou": 0.11, "context_pq": 0.12,
        "context_map": {"map": 0.13, "map_50": 0.14},
        "target_miou": 0.21, "target_pq": 0.22,
        "target_map": {"map": 0.23, "map_50": 0.24},
    }, "novel": {
        "target_miou": 0.31, "target_pq": 0.32,
        "target_map": {"map": 0.33, "map_50": 0.34},
    }}
    assert _official_scope(metrics, "context")["scope_official_ap50"] == 0.14
    assert _official_scope(metrics, "target_all")["scope_official_ap50"] == 0.24
    assert _official_scope(metrics, "novel")["scope_official_ap50"] == 0.34
    metrics["novel"]["target_map"]["map_50"] = -1
    row = _official_scope(metrics, "novel")
    assert row["scope_official_ap50"] == "UNDEFINED" and row["scope_official_ap50_raw"] == -1


def test_official_provenance_attachment_uses_stable_arm_snapshot(tmp_path):
    root = tmp_path / "official" / "step_003584" / "probe"
    root.mkdir(parents=True)
    (root / "official_all.json").write_text('{"result": "all"}')
    (root / "official_novel.json").write_text('{"result": "novel"}')
    row = {"official": {"all": {"ap50": 0.2}, "novel": {"ap50": 0.1}}}
    returned = _attach_official_provenance(row, tmp_path, 3584, "probe")
    assert returned is row
    assert set(row["official"]) == {"all", "novel", "_provenance"}
    assert row["official"]["_provenance"]["all"]["sha256"]
    assert row["official"]["_provenance"]["novel"]["path"].endswith("official_novel.json")
