import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.train_object_locus_v2_1_expanded import (
    EPOCHS, FINAL_GLOBAL_STEP, INITIAL_GLOBAL_STEP, TOTAL_NEW_UPDATES,
    _probe32, build_expanded_plan, expanded_epoch_order, expanded_lr_multiplier,
    expanded_lrs, window_identity, _restore_own_checkpoint,
)


def _window(scene, context, novel, index):
    return {"scene": scene, "context": context, "novel": novel, "index": index}


def test_lr_schedule_fixed_values():
    assert expanded_lrs(1) == (1e-4, 1e-6)
    assert expanded_lrs(TOTAL_NEW_UPDATES) == (1e-5, 1e-7)
    assert expanded_lr_multiplier(1) == 1.0
    assert math.isclose(expanded_lr_multiplier((TOTAL_NEW_UPDATES + 1) // 2), 0.55, abs_tol=1e-4)
    assert INITIAL_GLOBAL_STEP == 1792
    assert FINAL_GLOBAL_STEP == 17920


def test_plan_epochs_permute_each_window_once():
    windows = [_window(f"scene{i % 128:03d}", [i * 10, i * 10 + 1], [i * 10 + 2], i) for i in range(1008)]
    manifest = {"expanded_train_windows": windows}
    plan = build_expanded_plan(manifest)
    assert len(plan["entries"]) == 16128
    assert len(plan["epoch_permutations"]) == 16
    for epoch in range(16):
        order = expanded_epoch_order(epoch, 1008)
        assert sorted(order) == list(range(1008))
        rows = plan["entries"][epoch * 1008:(epoch + 1) * 1008]
        assert [r["pool_index"] for r in rows] == order


def test_probe32_selection_is_deterministic_by_scene_and_sort():
    windows = []
    for s in reversed(range(128)):
        scene = f"scene{s:03d}"
        windows.append(_window(scene, [2, 9], [5], 1))
        windows.append(_window(scene, [1, 8], [4], 0))
    first = _probe32(windows)
    second = _probe32(windows)
    assert first == second
    assert len(first) == 32
    assert [r["scene"] for r in first] == [f"scene{s:03d}" for s in range(0, 128, 4)]
    assert all(r["context"] == [1, 8] for r in first)


def test_optimizer_state_restore_comparison_allows_only_lr_change():
    model_a = torch.nn.Linear(3, 2)
    opt_a = torch.optim.AdamW([{"params": list(model_a.parameters()), "name": "object_locus_v2_1_decay", "lr": 1e-4}], betas=(0.9, 0.95), eps=1e-8)
    (model_a(torch.ones(1, 3)).sum()).backward(); opt_a.step()
    saved = copy.deepcopy(opt_a.state_dict())
    model_b = torch.nn.Linear(3, 2)
    model_b.load_state_dict(model_a.state_dict())
    opt_b = torch.optim.AdamW([{"params": list(model_b.parameters()), "name": "object_locus_v2_1_decay", "lr": 1e-4}], betas=(0.9, 0.95), eps=1e-8)
    opt_b.load_state_dict(saved)
    opt_b.param_groups[0]["lr"] = 1e-4
    from scripts.train_object_locus_v2_1_expanded import _assert_optimizer_state_exact
    assert _assert_optimizer_state_exact(saved, opt_b)["state_entries"] == 2


def test_window_identity_uses_scene_context_novel():
    assert window_identity(_window("scene000", [1, 2], [3], 0)) == ("scene000", (1, 2), (3,))


def test_locked_splits_keep_holdout_frames_out_and_dev_scenes_disjoint():
    path = Path("/space/mawb/ssst/group_plus/object_locus_v2_1/data_manifest.json")
    manifest = json.loads(path.read_text())
    expanded = manifest["expanded_train_windows"]
    assert len(expanded) == 1008 and len({w["scene"] for w in expanded}) == 128
    holds = {w["scene"]: set(w["context"]) | set(w["novel"]) for w in manifest["same_scene_holdout16"]}
    for row in expanded:
        held = holds.get(row["scene"])
        assert held is None or held.isdisjoint(set(row["context"]) | set(row["novel"]))
    assert not ({w["scene"] for w in manifest["dev8"]} & {w["scene"] for w in expanded})


def test_own_checkpoint_restores_next_plan_cursor(tmp_path):
    from scripts.object_locus_v2_1_runtime import capture_rng
    torch.manual_seed(7)
    source = torch.nn.Linear(3, 2)
    source_opt = torch.optim.AdamW(source.parameters(), lr=1e-4)
    (source(torch.ones(1, 3)).sum()).backward(); source_opt.step()
    rng = capture_rng()
    payload = {"model": source.state_dict(), "optimizer": source_opt.state_dict(), "rng": rng,
        "git_sha": "exec", "execution_git_sha": "exec", "architecture_name": "LOCUSGS_OBJECT_LOCUS_V2_1",
        "recipe": "OBJECT_LOCUS_V2_1_EXPANDED_RECON_LR_1E6", "training_plan_sha256": "plan", "data_manifest_sha256": "data",
        "source_checkpoint_sha256": "source", "source_git_sha": "d9a5cef3263dc7b16ad79784560b3c30d2045b09",
        "next_epoch": 3, "next_position": 0, "expanded_step": 3024}
    checkpoint = tmp_path / "resume.pt"; torch.save(payload, checkpoint)
    target = torch.nn.Linear(3, 2)
    target_opt = torch.optim.AdamW(target.parameters(), lr=1e-4)
    restored = _restore_own_checkpoint(checkpoint, target, target_opt, exec_sha="exec",
        plan_sha="plan", data_manifest_sha="data", source_sha="source")
    assert restored["next_epoch"] == 3 and restored["next_position"] == 0
    assert restored["expanded_step"] == 3024
    assert all(torch.equal(source.state_dict()[k], target.state_dict()[k]) for k in source.state_dict())
