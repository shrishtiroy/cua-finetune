# cua-finetune

Fine-tune small open-source VLMs (Qwen3-VL-8B, Llama-3.2-11B-Vision, Kimi-VL-A3B, DeepSeek-VL2-Small) on Claude Opus CUA trajectories, then eval browser-in-the-loop. Phases 0 and 1 (data) run on a Mac; phases 2-4 (train + eval) run on Lambda.

## Setup (Mac, Phase 0–1)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # then fill SUPABASE_SERVICE_ROLE_KEY (mode 600)
chmod 600 .env
```

## Phase 0: pull trajectories

```bash
.venv/bin/python -m data.pull_trajectories          # all 3 sources, default limits
.venv/bin/python -m data.pull_trajectories --inventory   # print stats from existing manifest
```

Outputs:
- `data/manifests/pulled_trajectories.jsonl`
- `data/raw/supabase/<task_external_id>/<trial_name>/...`

## Phase 1: build dataset

```bash
.venv/bin/python -m data.categorize_tasks                   # → data/manifests/categories.yaml
.venv/bin/python -m data.atif_to_swift                      # rejection sampling, 80/20 split
.venv/bin/python -m data.validate_dataset                   # sanity check
```

Outputs:
- `data/cua_sft/{train,test}.jsonl`
- `data/manifests/{splits,held_out_tasks,dataset_stats}.yaml`

## Phase 2–4 (later, on Lambda)

`configs/` holds ms-swift YAMLs; `eval/` holds vLLM-backed Dillinger backends; `scripts/` is the launcher set. Those phases need `ms-swift`, `vllm`, `torch` — install on the GPU box, not here.

## Env vars

- `SUPABASE_URL` — `https://rwghorglubtbjfzjotzu.supabase.co`
- `SUPABASE_SERVICE_ROLE_KEY` — required for storage downloads. Never log or commit.

## Layout

```
data/
  pull_trajectories.py   # sources: supabase, local Dillinger, trajsh URLs
  categorize_tasks.py    # C1–C5 + C99_other
  atif_to_swift.py       # rejection sampling -> ms-swift sharegpt-multimodal JSONL
  validate_dataset.py    # PIL + JSON sanity, template smoke-test
  inventory.py           # (alias of pull_trajectories.py --inventory)
  manifests/             # JSONL/YAML stats committed; raw downloads gitignored
  raw/                   # mirrored Supabase storage; gitignored
  cua_sft/               # final train/test JSONL; gitignored
configs/                 # placeholder ms-swift YAMLs
eval/                    # placeholder vLLM backends
scripts/                 # placeholder launchers
results/                 # eval outputs; gitignored
```
