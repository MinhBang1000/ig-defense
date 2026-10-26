# InteractionGuard

Offline evaluation of a defense for tool-using LLM agents:

- **Screen**: frozen Deepset injection classifier, applied channel-wise.
- **Gate**: adapted ModernBERT encoder and benign-only AutoEncoder.
- **Audit**: LLM semantic auditor of the visible interaction and user authorization.

Primary cascade: **Screen-Gate-Audit** (paper L1-L2-L3). Names identify mechanisms,
not positions. The legacy/paper mapping is explicit in `config/experiments.yaml`.

## Setup

Run from the repository root. Real experiments require an appropriate NVIDIA
GPU; functional tests need no models, credentials or GPU. Use separate environments:

```bash
conda create -n agent_defense python=3.10
conda activate agent_defense
pip install -r requirements/ml.txt

python3.11 -m venv .venv/audit_llama
.venv/audit_llama/bin/pip install -r requirements/audit_llama.txt

python3.11 -m venv .venv/audit_2026
.venv/audit_2026/bin/pip install -r requirements/audit_2026.txt
```

These files pin direct dependencies of the tested environments, not every
transitive dependency or CUDA driver. Match the PyTorch CUDA wheel/driver to
your host. Set `runtime.python` to your ML interpreter in the YAML before running
experiments on your machine. Configure Audit/candidate interpreters
separately. Do not upgrade frozen Llama to accommodate the modern candidates.

The code repository includes raw traces, fixed split membership and prompt
judge/review inputs. Prepared datasets and completed results are kept out of Git
for a separate Zenodo release; weights, environments and credentials are not included.
Obtain the required model access separately; revisions are pinned in YAML.

Required local inputs:
```text
data/raw/benign_traces.jsonl
data/raw/prompt_injection_traces.jsonl
data/raw/rag_poison_traces.jsonl
data/raw/tool_injection_traces.jsonl
data/raw/correlated_injection_traces.jsonl
data/splits/split_manifest.json
eval/prompt_success_judgements.jsonl
eval/prompt_success_reviews.jsonl
```

Keep original split membership and order. Item 0 refuses to invent new membership.
Judge/review caches are inputs; missing prompt outcomes require real judging
unless explicitly disabled. Raw traces remain unchanged.

Optional modern-backend preparation (explicit installation/download commands):
```bash
python -m research_runner.backend_setup --check
python -m research_runner.backend_setup --prepare
```
This helper prepares modern candidates, not the ML/frozen Llama environments.
`--smoke` explicitly runs a small GPU diagnostic.

## Run

Edit **only `config/experiments.yaml`** for scientific/runtime parameters.
No parameter prompts; overwrite confirmation defaults **No**. Each run snapshots
the effective configuration.

```bash
python main.py
python main.py --config config/experiments.yaml --run 2
```

| Item | Operation | Prerequisites |
| --- | --- | --- |
| 0 | Full dataset preparation | Raw traces, membership, outcome inputs |
| 1 | Train/calibrate encoder and Gate; calibrate Screen | 0 |
| 2 | Nine configurations, five seeds | 0, 1 |
| 3 | Channel-specific baselines and ASR | 0, 2 Stage 2 |
| 4 | Same-Audit channel/OR vs full-history ablation | 0 |
| 5 | Audit backends, Screen-Gate-Audit only | 2 Stage 1 and Stage 2 for Llama reuse |
| 6 | Paired Gate threshold sweep, two orders | 2 Stage 1; matching Stage 2 reused if present |
| 7 | Benign-ratio sweep, two orders | 2 Stage 1 |
| 8 | Manually selected auditor-targeted case | 0, 1, 2 |
| 9 | Verified ZIP snapshot | Configured local artifacts |

Missing/incompatible prerequisites fail with an explanation, never silently rerun.
`Q` exits. Progress reports identify seed, order, worker, requests and completion.

### Execution and Reuse

Item 2 `experiments.configuration.stage`: `stage1`, `stage2` or `full`.
Stage 1 stores shared Screen scores, Gate scores/checkpoints per seed, routing
and exact pending Audit requests per order. Its process exits before Stage 2
loads Audit. `full` executes sequentially. Standalone Audit receives all test
prefixes. Offline score collection order does not change cascade routing.

Screen positives stop immediately. A nonterminal Gate forwards normal histories
and escalates anomalies. Terminal Screen/Gate use strict `score > threshold`;
Audit uses the frozen schema/confidence rule. Tests cover all nine orders and
threshold boundaries.

Seeds 42-46 adapt encoder/Gate from prepared data. Correlated episodes are excluded
from adaptation/calibration. Encoder uses 0/1; AE only 0. Gate thresholds target
validation recall >=99.9% separately per seed; frozen Screen is calibrated once
with validation FPR <=1%. Test data never select thresholds.

Item 5 reuses exact Stage 1 pending sets across candidates within each seed, and
verified Llama decisions from Item 2 without rerunning Llama. Modern candidates
use separate runtimes. Nemotron has an explicit JSON whitespace safeguard:
this compares deployed backends, not identical decoding implementations.
Prompt, schema and decision policy remain fixed.

Item 6 sweeps validation max-F1 plus recall floors 90/95/98/99.9%, paired across
orders. The strict point reuses verified Item 2 results; missing/nonmatching
references run fresh with a recorded reason. Item 7 samples a workload per
seed/benign ratio (90/92/94/96/98%), reuses Screen/Gate scores and recomputes routing
on sampled IDs. Thresholds are not retuned for class imbalance.

## Data and Measurement

Item 0 stages annotation caches, annotates outcomes, preserves episode splits,
normalizes channel templates, expands chronological prefixes, and exports
encoder/AE data. Prefix labels follow observable payload onset, not scenario
intent or success. Metadata never enters model inputs. Identical prefixes can
occur across different episodes/splits: episode hold-out does not guarantee
novel text.

ASR is an **offline replay estimate** using stored outcomes and the safe-action
boundary in `research_runner/implementation/evaluation/asr.py`. Detection after
the harmful action does not establish prevention. Reports distinguish
category/macro/pooled ASR, attempts/successes, benign episode false blocking,
pre-injection blocks, first detection and late detection.

Sentinel inspects prompt payloads, adapted RAGuard retrieved documents, and
Prompt Guard 2 tool payloads. Combined OR aggregates their verdicts.
These are offline adaptations, not their original live-agent protocols.
Item 4 controls original-prompt anchoring and input budgets; its full-history
result is Audit alone, not a cascade.

Audit inputs have explicit character/block caps despite a longer model context.
Finite tests cannot guarantee defense against arbitrary/adaptive attacks.
Item 8 is one selected-case offline stress test, not population F1/ASR or
universal auditor robustness.

Timing distinguishes component-based pipeline estimates from measured end-to-end
latency. Model loading/stage changes are excluded. Cached verdicts are not fresh
latency measurements or independent trials. Deterministic reuse does not create
five independent inference repetitions.

## Artifacts and Backup

`results/research_runner/` contains `dataset`, `training`,
`configuration_stage1`, `configuration_stage2`, `baselines`, `channel_ablation`,
`backend`, `threshold`, `benign`, `auditor_targeted` and compatibility receipts.
Training checkpoints live here, not in a legacy `models/` directory.
Item 8 case copies live in `data/auditor_targeted/`, never training data.

Artifacts contain checksums, effective configs, model/data/prompt/code provenance,
IDs, scores/routes, requests/responses, errors and timing/cache provenance.
`resume: true` only reuses compatible completed results. Overwrite stages and
publishes atomically; failure preserves the previous completed result.
Reviewed source-only changes can have explicit hash-bound compatibility receipts;
these do not waive scientific input or indexed-file integrity checks.

Item 9 creates a verified ZIP in `backup/` with effective configs/checksums.
Checkpoints are optional. Secrets, environments and nested ZIPs are excluded.
Local docs, processed data, results and environments are Git-ignored. Raw traces,
split membership and outcome inputs are retained for dataset reconstruction.

To prepare a separate review copy for Zenodo (no ZIP or upload is created):

```bash
python publication/prepare_zenodo.py
cd zenodo
zip -r -X ig-defense-results.zip ig-defense
unzip -t ig-defense-results.zip
```

Only the first command prepares/redacts metadata; run the ZIP commands yourself.
The export has a short README and SHA-256 manifest. Source artifacts are unchanged.
The review copy omits weights/embeddings and is not a resumable checkpoint bundle.

## CPU Functional Tests

```bash
python -m unittest discover -s research_runner/tests -q
```

Tests use temporary synthetic data and fake models with real orchestration,
routing, serialization, calibration, ASR, atomic publication and backup.
No training, downloads, APIs or GPU experiments. Passing these tests is not
evidence of real-model accuracy or bit-identical results on another server.
