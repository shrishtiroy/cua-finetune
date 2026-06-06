"""Phase 0b + 0c: pull trajectories from Supabase, local Dillinger, and trajectories.sh.

Builds ``data/manifests/pulled_trajectories.jsonl`` from three sources with dedup.

Sources, in priority order:
    1. supabase        — primary; queries ``public.trajectory_trials`` JOIN
                         ``public.trajectory_jobs`` via PostgREST and downloads
                         files from the private ``trajectory-jobs`` storage
                         bucket using the service-role key loaded from ``.env``.
    2. dillinger_local — walks ``../Dillinger/runs/**/result.json`` and records
                         absolute paths (no copy).
    3. trajsh          — best-effort GET on the 39 URLs in
                         ``../Dillinger/python_docs_trajectories.tsv``. Most
                         require auth and 404; that is expected.

Dedup key: ``(task_external_id, model_name, n_steps, reward_rounded_to_2)``.
Source preference on collision: supabase > dillinger_local > trajsh.

Run::

    python -m data.pull_trajectories [--sources supabase,local,trajsh]
                                     [--max-per-task 10]
                                     [--score-min 0.0]
                                     [--limit N]
                                     [--dry-run]
                                     [--inventory]
                                     [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

logger = logging.getLogger("pull_trajectories")

REPO_ROOT = Path(__file__).resolve().parents[1]
SIBLINGS = REPO_ROOT.parent
DILLINGER = SIBLINGS / "Dillinger"

DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "pulled_trajectories.jsonl"
DEFAULT_CHECKPOINT = REPO_ROOT / "data" / "manifests" / ".pull_checkpoint.json"
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
PYDOCS_TSV = DILLINGER / "python_docs_trajectories.tsv"
DILLINGER_RUNS = DILLINGER / "runs"

BUCKET = "trajectory-jobs"
SUPABASE_AGENT_FILTER = ("conduit", "dillinger", "taiga-nibbles", "refresh-editor", "computer-1")


@dataclass
class ManifestRow:
    source: str
    task_external_id: str
    agent_name: str
    model_name: str
    reward: float
    n_steps: int
    trial_id: str
    local_trajectory_path: str
    local_screenshots_dir: str
    storage_prefix: str | None
    dedup_key: str
    extra: dict[str, Any] = field(default_factory=dict)


def _dedup_key(task_external_id: str, model_name: str, n_steps: int, reward: float) -> str:
    return f"{task_external_id}|{model_name}|{n_steps}|{round(float(reward), 2)}"


def _source_rank(source: str) -> int:
    return {"supabase": 0, "dillinger_local": 1, "trajsh": 2}.get(source, 99)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)[:120] or "unknown"


class SupabaseStorage:
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"apikey": key, "Authorization": f"Bearer {key}"})

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _get(self, name: str) -> requests.Response:
        return self.session.get(
            f"{self.url}/storage/v1/object/{BUCKET}/{name}",
            timeout=60,
        )

    def download(self, name: str, target: Path) -> tuple[bool, int, str | None]:
        try:
            resp = self._get(name)
        except Exception as exc:  # noqa: BLE001
            return False, 0, f"req_error: {exc}"
        if resp.status_code == 404:
            return False, 0, "404"
        if not resp.ok:
            return False, 0, f"http_{resp.status_code}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.content)
        return True, len(resp.content), None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def _postgrest_get(url: str, key: str, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    resp = requests.get(f"{url.rstrip('/')}/rest/v1/{path}", headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_supabase_trials(url: str, key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    page_size = 1000
    offset = 0
    while True:
        rows = _postgrest_get(
            url,
            key,
            "trajectory_trials",
            params={
                "select": (
                    "id,job_id,task_external_id,agent_name,model_name,reward,n_steps,"
                    "trial_name,created_at,"
                    "trajectory_jobs(slug,storage_prefix,created_at)"
                ),
                "agent_name": f"in.({','.join(SUPABASE_AGENT_FILTER)})",
                "reward": "not.is.null",
                "trajectory_jobs.storage_prefix": "not.is.null",
                "order": "reward.desc.nullslast,n_steps.asc",
                "limit": str(page_size),
                "offset": str(offset),
            },
        )
        if not rows:
            break
        out.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    flat: list[dict[str, Any]] = []
    for r in out:
        job = r.get("trajectory_jobs") or {}
        if not job.get("storage_prefix"):
            continue
        flat.append({
            "trial_id": r["id"],
            "job_id": r["job_id"],
            "task_external_id": r["task_external_id"],
            "agent_name": r["agent_name"],
            "model_name": r["model_name"],
            "reward": r["reward"],
            "n_steps": r["n_steps"],
            "trial_name": r["trial_name"],
            "job_slug": job.get("slug"),
            "storage_prefix": job["storage_prefix"],
            "created_at": r.get("created_at"),
        })
    return flat


_SCREENSHOT_KEYS = ("image_path", "screenshot_path", "image_url", "path", "url")


def referenced_screenshots(traj: dict[str, Any]) -> list[str]:
    """Walk ``trajectory.json`` and collect anything that looks like a screenshot ref."""
    found: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _SCREENSHOT_KEYS and isinstance(v, str):
                    if v.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) or "screenshot" in v.lower():
                        found.add(v)
                visit(v)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(traj)
    return sorted(found)


def _download_one_trial(
    storage: SupabaseStorage,
    trial: dict[str, Any],
    raw_root: Path,
) -> tuple[ManifestRow | None, dict[str, int]]:
    counts: dict[str, int] = {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0}

    storage_prefix = trial["storage_prefix"].rstrip("/")
    trial_name = trial["trial_name"]
    if not trial_name:
        counts["failed"] += 1
        return None, counts

    task_dir = raw_root / "supabase" / _slug(trial["task_external_id"]) / trial_name
    agent_dir = task_dir / "agent"
    screenshots_dir = agent_dir / "screenshots"

    traj_local = agent_dir / "trajectory.json"
    if traj_local.exists() and traj_local.stat().st_size > 0:
        counts["skipped"] += 1
    else:
        ok, nbytes, err = storage.download(f"{storage_prefix}/{trial_name}/agent/trajectory.json", traj_local)
        if not ok:
            logger.debug("trajectory.json missing for %s (%s)", trial_name, err)
            counts["failed"] += 1
            return None, counts
        counts["downloaded"] += 1
        counts["bytes"] += nbytes

    try:
        traj = json.loads(traj_local.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("trajectory.json unreadable for %s: %s", trial_name, exc)
        counts["failed"] += 1
        return None, counts

    result_local = task_dir / "result.json"
    if not (result_local.exists() and result_local.stat().st_size > 0):
        ok, nbytes, err = storage.download(f"{storage_prefix}/{trial_name}/result.json", result_local)
        if ok:
            counts["downloaded"] += 1
            counts["bytes"] += nbytes
        else:
            counts["failed"] += 1

    refs = referenced_screenshots(traj)
    for ref in refs:
        if ref.startswith("http"):
            continue
        rel = ref.lstrip("/")
        # Two layouts observed:
        #   a) <prefix>/<trial>/agent/screenshots/step-NN.jpg
        #   b) <prefix>/<trial>/agent/screenshot_step_NN.webp (alt agents)
        # Mirror them with their original relative path under agent/.
        if rel.startswith("agent/"):
            rel = rel[len("agent/"):]
        local = agent_dir / rel
        if local.exists() and local.stat().st_size > 0:
            counts["skipped"] += 1
            continue
        ok, nbytes, err = storage.download(f"{storage_prefix}/{trial_name}/agent/{rel}", local)
        if ok:
            counts["downloaded"] += 1
            counts["bytes"] += nbytes
        else:
            counts["failed"] += 1

    screenshots_dir.mkdir(parents=True, exist_ok=True)

    reward = float(trial["reward"])
    n_steps = int(trial.get("n_steps") or len([s for s in traj.get("steps") or [] if s.get("source") == "agent"]))
    row = ManifestRow(
        source="supabase",
        task_external_id=trial["task_external_id"],
        agent_name=trial["agent_name"],
        model_name=trial["model_name"] or "",
        reward=reward,
        n_steps=n_steps,
        trial_id=str(trial["trial_id"]),
        local_trajectory_path=str(traj_local.resolve()),
        local_screenshots_dir=str(screenshots_dir.resolve()),
        storage_prefix=storage_prefix,
        dedup_key=_dedup_key(trial["task_external_id"], trial["model_name"] or "", n_steps, reward),
        extra={
            "trial_name": trial_name,
            "job_slug": trial.get("job_slug"),
            "result_path": str(result_local.resolve()) if result_local.exists() else None,
            "n_screenshot_refs": len(refs),
        },
    )
    return row, counts


def pull_supabase(
    url: str,
    key: str,
    raw_root: Path,
    max_per_task: int | None,
    score_min: float,
    limit: int | None,
    workers: int,
    checkpoint_path: Path,
    dry_run: bool,
) -> tuple[list[ManifestRow], dict[str, Any]]:
    stats: dict[str, Any] = {
        "queried": 0,
        "after_filter": 0,
        "downloaded_trials": 0,
        "failed_trials": 0,
        "files_downloaded": 0,
        "files_skipped": 0,
        "file_failures": 0,
        "bytes": 0,
    }
    logger.info("Querying Supabase for trials …")
    trials = fetch_supabase_trials(url, key)
    stats["queried"] = len(trials)
    logger.info("Got %d trials from Supabase metadata", len(trials))

    trials = [t for t in trials if t.get("reward") is not None and float(t["reward"]) >= score_min]
    if max_per_task is not None:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in trials:
            by_task[t["task_external_id"]].append(t)
        kept: list[dict[str, Any]] = []
        for _, rs in by_task.items():
            rs.sort(key=lambda r: (-(r.get("reward") or 0), r.get("n_steps") or 1_000_000))
            kept.extend(rs[:max_per_task])
        trials = kept
    if limit is not None:
        trials = trials[:limit]
    stats["after_filter"] = len(trials)

    done_trial_ids: set[str] = set()
    if checkpoint_path.exists():
        try:
            cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            done_trial_ids = set(cp.get("supabase_done", []))
        except (json.JSONDecodeError, OSError):
            done_trial_ids = set()

    if dry_run:
        logger.info("[dry-run] would download %d trials (already cached: %d)",
                    len(trials), sum(1 for t in trials if str(t["trial_id"]) in done_trial_ids))
        return [], stats

    storage = SupabaseStorage(url, key)
    rows: list[ManifestRow] = []
    new_done: list[str] = list(done_trial_ids)

    def _save_checkpoint() -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"supabase_done": sorted(set(new_done))}, indent=2))

    workers = max(1, min(workers, 16))
    pbar = tqdm(total=len(trials), desc="supabase", unit="trial")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one_trial, storage, t, raw_root): t for t in trials}
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                trial = futures[fut]
                try:
                    row, counts = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("trial %s failed: %s", trial["trial_name"], exc)
                    stats["failed_trials"] += 1
                    pbar.update(1)
                    continue
                stats["files_downloaded"] += counts["downloaded"]
                stats["files_skipped"] += counts["skipped"]
                stats["file_failures"] += counts["failed"]
                stats["bytes"] += counts["bytes"]
                if row is None:
                    stats["failed_trials"] += 1
                else:
                    rows.append(row)
                    stats["downloaded_trials"] += 1
                new_done.append(str(trial["trial_id"]))
                if i % 50 == 0:
                    _save_checkpoint()
                pbar.update(1)
        finally:
            pbar.close()
            _save_checkpoint()
    return rows, stats


def pull_dillinger_local() -> tuple[list[ManifestRow], dict[str, Any]]:
    rows: list[ManifestRow] = []
    stats: dict[str, Any] = {"scanned": 0, "kept": 0, "missing_trajectory": 0, "missing_screenshots": 0}
    if not DILLINGER_RUNS.exists():
        logger.warning("Dillinger runs dir not found: %s", DILLINGER_RUNS)
        return rows, stats

    for result_path in DILLINGER_RUNS.rglob("result.json"):
        stats["scanned"] += 1
        run_dir = result_path.parent
        task_dir = run_dir.parent
        task_name = task_dir.name
        traj_path = run_dir / "trajectory.json"
        screenshots_dir = run_dir / "screenshots"

        if not traj_path.exists():
            stats["missing_trajectory"] += 1
            continue
        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        try:
            traj = json.loads(traj_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        agent = traj.get("agent") or {}
        agent_name = agent.get("name") or "dillinger"
        model_name = (agent.get("model_name") or "") or ""
        n_steps = len(traj.get("steps") or [])

        grade = result_data.get("grade") or {}
        raw_score = grade.get("score")
        if not isinstance(raw_score, (int, float)):
            continue
        reward = float(raw_score)
        if not screenshots_dir.exists():
            stats["missing_screenshots"] += 1

        trial_id = run_dir.name
        rows.append(
            ManifestRow(
                source="dillinger_local",
                task_external_id=task_name,
                agent_name=agent_name,
                model_name=model_name,
                reward=reward,
                n_steps=n_steps,
                trial_id=trial_id,
                local_trajectory_path=str(traj_path.resolve()),
                local_screenshots_dir=str(screenshots_dir.resolve()),
                storage_prefix=None,
                dedup_key=_dedup_key(task_name, model_name, n_steps, reward),
                extra={
                    "result_path": str(result_path.resolve()),
                    "backend": result_data.get("backend"),
                    "elapsed_seconds": result_data.get("elapsed_seconds"),
                    "source_file": (result_data.get("metadata") or {}).get("source_file"),
                },
            )
        )
        stats["kept"] += 1
    return rows, stats


def pull_trajsh(raw_root: Path) -> tuple[list[ManifestRow], dict[str, Any]]:
    rows: list[ManifestRow] = []
    stats: dict[str, Any] = {"attempted": 0, "succeeded": 0, "failed": 0}
    if not PYDOCS_TSV.exists():
        return rows, stats

    cache_root = raw_root / "trajsh"
    cache_root.mkdir(parents=True, exist_ok=True)

    for raw_line in PYDOCS_TSV.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split("\t")
        if not parts or not parts[0].strip():
            continue
        url = parts[0].strip()
        task_name = parts[1].strip() if len(parts) > 1 else ""
        if not task_name:
            continue
        stats["attempted"] += 1
        target_dir = cache_root / _slug(task_name)
        traj_target = target_dir / "trajectory.json"
        try:
            resp = requests.get(url, timeout=20)
        except (requests.ConnectionError, requests.Timeout):
            stats["failed"] += 1
            continue
        ctype = resp.headers.get("content-type", "")
        if not resp.ok or not ctype.startswith(("application/json", "text/")):
            stats["failed"] += 1
            continue
        try:
            traj = resp.json()
        except ValueError:
            stats["failed"] += 1
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        traj_target.write_text(json.dumps(traj), encoding="utf-8")
        n_steps = len(traj.get("steps") or [])
        rows.append(
            ManifestRow(
                source="trajsh",
                task_external_id=task_name,
                agent_name="conduit",
                model_name="",
                reward=0.0,
                n_steps=n_steps,
                trial_id=url.rsplit("/", 2)[-2] if "/t/" in url else _slug(task_name),
                local_trajectory_path=str(traj_target.resolve()),
                local_screenshots_dir=str((target_dir / "screenshots").resolve()),
                storage_prefix=None,
                dedup_key=_dedup_key(task_name, "", n_steps, 0.0),
                extra={"url": url},
            )
        )
        stats["succeeded"] += 1
    return rows, stats


def dedup_rows(rows: list[ManifestRow]) -> tuple[list[ManifestRow], dict[str, int]]:
    by_key: dict[str, ManifestRow] = {}
    dropped: int = 0
    for row in rows:
        existing = by_key.get(row.dedup_key)
        if existing is None:
            by_key[row.dedup_key] = row
            continue
        if _source_rank(row.source) < _source_rank(existing.source):
            by_key[row.dedup_key] = row
        dropped += 1
    return list(by_key.values()), {"deduped_rows": dropped, "unique_keys": len(by_key)}


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for row in rows:
            h.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def inventory(manifest_path: Path, sample_size: int = 50, seed: int = 42) -> None:
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    rows = load_manifest(manifest_path)
    n = len(rows)
    by_source: Counter[str] = Counter(r["source"] for r in rows)
    by_agent: Counter[str] = Counter(r["agent_name"] for r in rows)

    perfect = [r for r in rows if r["reward"] == 1.0]
    high = [r for r in rows if 0.8 <= r["reward"] < 1.0]
    mid = [r for r in rows if 0.5 <= r["reward"] < 0.8]
    low = [r for r in rows if 0.0 < r["reward"] < 0.5]
    zero = [r for r in rows if r["reward"] == 0.0]

    def utasks(subset: list[dict[str, Any]]) -> int:
        return len({r["task_external_id"] for r in subset})

    top_perfect = Counter(r["task_external_id"] for r in perfect).most_common(20)

    print("=" * 72)
    print(f"INVENTORY — {manifest_path}")
    print("=" * 72)
    print(f"Total trajectories:                {n}")
    print()
    print("Score distribution (reward):")
    print(f"  == 1.0                            {len(perfect):>5d}   unique tasks: {utasks(perfect)}")
    print(f"  >= 0.8 (and < 1.0)                {len(high):>5d}   unique tasks: {utasks(high)}")
    print(f"  >= 0.5 (and < 0.8)                {len(mid):>5d}   unique tasks: {utasks(mid)}")
    print(f"  >  0.0 (and < 0.5)                {len(low):>5d}   unique tasks: {utasks(low)}")
    print(f"  == 0.0                            {len(zero):>5d}   unique tasks: {utasks(zero)}")
    print()
    print(f"Unique tasks (any reward):         {utasks(rows)}")
    print(f"Unique tasks with >=1 perfect run: {utasks(perfect)}")
    print(f"Unique tasks with reward >= 0.8:   {utasks(perfect + high)}")
    print(f"Unique tasks with reward >= 0.5:   {utasks(perfect + high + mid)}")
    print()
    print("Top 20 tasks by number of perfect runs:")
    for task, count in top_perfect:
        print(f"  {count:>3d}  {task}")
    print()
    print("Per-model breakdown — perfect runs (top 15):")
    perfect_by_model: Counter[str] = Counter(r.get("model_name") or "(unknown)" for r in perfect)
    for model, count in perfect_by_model.most_common(15):
        print(f"  {count:>4d}  {model}")
    print()
    print("Per-agent breakdown (all rows):")
    for agent, count in by_agent.most_common():
        print(f"  {count:>4d}  {agent}")
    print()
    print("Sources breakdown (after dedup; manifest has the surviving row only):")
    for src, count in by_source.most_common():
        print(f"  {count:>4d}  {src}  ({100.0 * count / n:.1f}%)")
    print()

    rng = random.Random(seed)
    sample = rng.sample(rows, min(sample_size, n))
    ok = bad_missing = bad_decode = 0
    bad_examples: list[str] = []
    for r in sample:
        try:
            traj = json.loads(Path(r["local_trajectory_path"]).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            bad_decode += 1
            continue
        refs = referenced_screenshots(traj)
        rel_refs = [ref for ref in refs if not ref.startswith("http")]
        if not rel_refs:
            ok += 1
            continue
        screenshot_root = Path(r["local_screenshots_dir"])
        traj_dir = Path(r["local_trajectory_path"]).parent
        missing_in_this = False
        for ref in rel_refs[:6]:
            name = Path(ref).name
            rel = ref.lstrip("/")
            if rel.startswith("agent/"):
                rel = rel[len("agent/"):]
            candidates = [
                traj_dir / rel,
                screenshot_root / name,
                traj_dir / ref,
                traj_dir.parent / ref,
            ]
            if not any(p.exists() and p.stat().st_size > 0 for p in candidates):
                missing_in_this = True
                if len(bad_examples) < 3:
                    bad_examples.append(f"{r['source']}/{r['trial_id']}: {ref}")
                break
        if missing_in_this:
            bad_missing += 1
        else:
            ok += 1
    print(f"Screenshot integrity sample (n={len(sample)}): ok={ok} missing={bad_missing} decode_err={bad_decode}")
    for ex in bad_examples:
        print(f"  example missing: {ex}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull CUA trajectories into a manifest")
    parser.add_argument(
        "--sources",
        default="supabase,local,trajsh",
        help="Comma-separated subset of {supabase,local,trajsh}",
    )
    parser.add_argument("--max-per-task", type=int, default=10,
                        help="Cap Supabase trials per task_external_id before downloading")
    parser.add_argument("--score-min", type=float, default=0.0,
                        help="Skip Supabase trials with reward below this")
    parser.add_argument("--limit", type=int, default=None,
                        help="Hard cap on total Supabase trials (debug)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="Query Supabase but don't download")
    parser.add_argument("--inventory", action="store_true",
                        help="Skip pulling — just summarize an existing manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if args.inventory:
        inventory(args.manifest)
        return 0

    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    all_rows: list[ManifestRow] = []
    source_stats: dict[str, Any] = {}

    if "supabase" in sources:
        load_dotenv(REPO_ROOT / ".env")
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not (url and key):
            print("ERROR: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env", file=sys.stderr)
            return 2
        t0 = time.time()
        rows, stats = pull_supabase(
            url=url,
            key=key,
            raw_root=args.raw_root,
            max_per_task=args.max_per_task,
            score_min=args.score_min,
            limit=args.limit,
            workers=args.workers,
            checkpoint_path=args.checkpoint,
            dry_run=args.dry_run,
        )
        stats["elapsed_sec"] = round(time.time() - t0, 1)
        all_rows.extend(rows)
        source_stats["supabase"] = stats

    if "local" in sources:
        rows, stats = pull_dillinger_local()
        all_rows.extend(rows)
        source_stats["dillinger_local"] = stats

    if "trajsh" in sources:
        rows, stats = pull_trajsh(args.raw_root)
        all_rows.extend(rows)
        source_stats["trajsh"] = stats

    deduped, dedup_stats = dedup_rows(all_rows)
    deduped.sort(key=lambda r: (r.task_external_id, _source_rank(r.source), r.n_steps, -r.reward))

    if args.dry_run and "supabase" in sources and not source_stats.get("supabase", {}).get("downloaded_trials"):
        logger.info("[dry-run] skipping manifest write")
    else:
        write_manifest(deduped, args.manifest)

    summary = {
        "total_rows": len(deduped),
        "by_source": dict(Counter(r.source for r in deduped)),
        "dedup": dedup_stats,
        "source_stats": source_stats,
        "manifest_path": str(args.manifest),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
