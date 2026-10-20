# SSST Superpoint–GT matching audit

This is inference and statistics only. It never constructs an optimizer, calls
backward, updates parameters, edits training files, or selects latest Hungarian
weights. See `reports/source_audit.md` for exact indexing and denominator semantics.

Primary weight is pinned by absolute path and SHA256 in `snapshot/identity.json`:
completed sem256/A4 Epoch4, 4172 updates. The recorded base commit contains sem128;
`snapshot/semantic_memory.py` captures the actual sem256 execution class verbatim.
The audit selects this existing class in its own process and validates its output
shape. Tracked model files stay at the recorded base commit.

## Stages

Use `/space/mawb/anaconda3/envs/tokengs/bin/python` and set
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`.

```bash
python -m audit.ssst_matching.test_contracts
python -m audit.ssst_matching.run --stage cpu-preflight --cohort smoke8 --output /absolute/new/cpu_preflight
python -m audit.ssst_matching.run --stage smoke --cohort smoke8 --output /absolute/new/smoke8
```

The CPU preflight reads eight real windows but does **not** claim real prediction
parity. GPU smoke must independently pass exact native support score/allowed
parity and native Hungarian parity on all eight windows. Failure stops the run.
No tolerance is relaxed. Its partial progress and stack trace are retained.
Synthetic data exists only in CPU contracts and is never mixed into reports.

After real smoke passes, commit and push code; verify the remote branch SHA with
`git ls-remote`. Save a push receipt with `git_sha`, `remote`, `ref` (full
`refs/heads/...`), and `remote_verified: true`. Formal mode checks that receipt,
remote SHA, the smoke receipt, complete implementation hashes, and checkpoint SHA.

```bash
python -m audit.ssst_matching.run --stage formal --cohort heldout32 --smoke-receipt /absolute/smoke8/SMOKE_PASSED.json --push-receipt /absolute/push_verified.json --output /absolute/new/heldout32
python -m audit.ssst_matching.run --stage formal --cohort heldout128 --smoke-receipt /absolute/smoke8/SMOKE_PASSED.json --push-receipt /absolute/push_verified.json --output /absolute/new/heldout128
```

Heldout32 is the first 32 records of the existing, prediction-independent
representative128 cohort, not a newly selected dataset. Heldout128 expands to all
128 fixed records. One context window per scene. `train16` is available for the
existing training probe cohort but is not submitted by default. No new cohort or
training experiment is inferred. Repeated cohort windows across 32/128 are
reported as overlapping subsets, never added together as 160 independent scenes.

All thresholds share the same inputs. Size quantile levels are fixed in advance
at 1/3,2/3, computed from GT areas in each named cohort; boundaries are written to
audit_config. Size labels should only be compared across different cohorts after
checking their recorded boundaries. All method and threshold comparisons within
a cohort share boundaries. No predictions influence the cuts.

Outputs: requested seven report/config tables, `diagnostics/*.npz`, completion
receipt and progress. Graph archives include native GT IDs/classes, source hits,
valid denominators, purity, allowed edges, the actual native masked Cost,
Hungarian/Top-1 pairs, context IoU, and predicted class. Do not push checkpoint,
data, model predictions, or graph archives; push code and light CSV/Markdown/JSON
reports only. Every output directory is new; overwrite/resume is refused.

The task's per-GT argmin resolves cost ties by the first query index. The official
OneFormer SparseMatcher(topk=1) has a strict second-smallest cutoff, so an exact
minimum tie returns no pair; it is additionally recorded, not conflated with
argmin. No conflict is silently deduplicated. Its public source excerpt is saved.

Slurm jobs must not be polled after their one bounded submission check. Smoke is
a separate job stage: wait for user notification before examining completion,
then commit/push before formal submission. Formal32/128 and summarization do not
need any training job. GPU job script uses no background monitoring.

## Formal Slurm orchestration after reviewed Smoke

`submit_formal.py` creates a separate read-only copy of the pushed code. Publish a
lightweight immutable Git tag for that code commit and use that tag in the push
receipt; later report commits on the branch must not move this tag. This keeps
the runtime remote identity check stable while allowing final reports to be pushed.
Prepare with `--prepare-only --push-receipt /absolute/push_verified.json`, then
submit with the same push receipt. It submits Heldout32 on node14 / one4090,
and Heldout128 with afterok dependency on32, including each report writer. It
refuses duplicates and checks all implementation hashes against reviewed Smoke.
