# Phase 4b: cherry-pick blog-post pairs (awaits GPU eval results)

This phase needs Phase 3d outputs (per-run `result.json`s under
`results/<backend>/<adapter>/<task>/<ts>/`). Once the eval sweep has
completed on Lambda, run:

```bash
# Find every (task, model) pair where baseline failed (score<80) and finetuned
# passed (score>=80). Order by largest score-delta first.
python -c '
import json, glob
from pathlib import Path
from collections import defaultdict

results = Path("results")
by_pair = defaultdict(dict)  # (model, task, attempt_idx) -> {adapter: {score, run_dir}}
for p in results.glob("*/*/*/*/result.json"):
    parts = p.parts[-5:]
    model, adapter, task, ts = parts[0], parts[1], parts[2], parts[3]
    s = (json.load(open(p)).get("grade") or {}).get("score") or 0.0
    if 0.0 <= s <= 1.0:
        s *= 100.0
    by_pair[(model, task, ts)][adapter] = {"score": s, "run_dir": str(p.parent)}

candidates = []
for (model, task, _), data in by_pair.items():
    base = data.get("baseline", {}).get("score", -1)
    cua = data.get("cua", {}).get("score", -1)
    if base < 80 and cua >= 80:
        candidates.append({
            "model": model, "task": task,
            "baseline": data["baseline"], "finetuned": data["cua"],
            "delta": cua - base,
        })
candidates.sort(key=lambda c: -c["delta"])
print(json.dumps(candidates[:20], indent=2))
'
```

Then for the top 3-5 per model:

1. Open each pair's `trace.jsonl` in the local Dillinger viewer (`conduit view`)
   to confirm the win is real, not a grading flake.
2. Push the trajectory pair to trajectories.sh:
   ```bash
   npx --yes trajectories-sh upload trajectory <run_dir>/trajectory.json \
     --label "${MODEL_KEY}_${ADAPTER}_${TASK}"
   ```
3. Drop the resulting viewer URLs into the blog draft alongside the headline
   table from `results/headline.md`.

Cherry-pick is intentionally not automated: choose pairs that are visually
illustrative (clear baseline failure mode, clear corrective action by the
finetune) — that's the human judgment the blog needs.
