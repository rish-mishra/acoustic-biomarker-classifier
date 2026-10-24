"""Unattended, sequential runner for the remaining seed-study runs (validation only).

Reads reports/runs/seed_study/plan.json and trains every planned new run that is
not already completed, one at a time, with exactly the plan's CLI
(``--model <m> --lr 1e-4``) plus ``--seed`` and the seed-study output
directories. Nothing else is passed, so every other setting is train.py's
frozen default. Completed runs are reused; existing incomplete artifacts are
never overwritten (that run is recorded as blocked and skipped). A failed run
is recorded with its log and the runner continues with the next run; there
are no automatic retries and no setting is ever changed.

After all planned runs are completed, the post-training phase runs:
  1. analyze_run.py on all new seed-study runs (validation only)
  2. a reproduction check of each checkpoint against its run.json
  3. the unit test suite (python -m unittest discover -s tests)
  4. src/evaluation/seed_study.py (writes summary.json / summary.md)

Progress (all under reports/runs/seed_study/_runner/):
  status.json     current state of every run and step (rewritten atomically)
  runner.log      timestamped event log (start / complete / fail)
  post_*.log      output of each post-training step
  verification.json  reproduction check results
Each training run's stdout+stderr goes to reports/runs/seed_study/<run>/train_stdout.log.

The test split is never loaded: train.py and analyze_run.py refuse it.

Usage::

    python src/models/run_seed_study.py [--dry-run]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import msvcrt
import os
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_REPORTS = "reports/runs/seed_study"
STUDY_MODELS = "models/seed_study"
RUNNER_DIR = REPO_ROOT / STUDY_REPORTS / "_runner"
EXPECTED_TO_TRAIN = ["crnn_v1_s1337", "mobilenetv3_pretrained_v1_s1337", "baseline_cnn_v3_s2026",
                     "crnn_v1_s2026", "mobilenetv3_pretrained_v1_s2026"]
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: no console that can be closed (cf. attempt 2's window-CLOSE abort)
PYTHON = Path(sys.executable).with_name("python.exe")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Runner:
    def __init__(self):
        RUNNER_DIR.mkdir(parents=True, exist_ok=True)
        self.status = {"state": "starting", "pid": os.getpid(), "started": now(), "updated": now(),
                       "runs": {}, "post": {}}

    def log(self, msg: str):
        with open(RUNNER_DIR / "runner.log", "a", encoding="utf-8") as f:
            f.write(f"[{now()}] {msg}\n")

    def save(self):
        self.status["updated"] = now()
        tmp = RUNNER_DIR / "status.json.tmp"
        tmp.write_text(json.dumps(self.status, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, RUNNER_DIR / "status.json")


def capture(cmd: list[str]) -> str:
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, creationflags=NO_WINDOW)
    return (r.stdout + r.stderr).strip()


def planned_runs(plan: dict) -> list[dict]:
    """Every planned new run, in plan order, with its exact command."""
    by_base = {m["frozen_from"]: (name, m["cli"]) for name, m in plan["models"].items()}
    out = []
    for seed, names in plan["new_runs"].items():
        for run in names:
            arch, cli = by_base[run.rsplit("_s", 1)[0]]
            cmd = [str(PYTHON), "-u", "src/models/train.py", "--run-name", run, *shlex.split(cli),
                   "--seed", str(int(seed)), "--checkpoint-dir", STUDY_MODELS, "--reports-dir", STUDY_REPORTS]
            out.append({"run": run, "architecture": arch, "seed": int(seed), "cmd": cmd})
    return out


def run_state(run: str, seed: int) -> str:
    """'completed' (valid, reuse), 'absent' (train it) or 'incomplete' (do not touch)."""
    rj = REPO_ROOT / STUDY_REPORTS / run / "run.json"
    ck = REPO_ROOT / STUDY_MODELS / run / "best.pt"
    if rj.exists():
        j = json.loads(rj.read_text(encoding="utf-8"))
        ok = (j.get("status") == "completed" and j.get("seed") == seed and j.get("test_split_evaluated") is False
              and ck.exists())
        return "completed" if ok else "incomplete"
    return "incomplete" if ck.exists() else "absent"


def other_training_pids() -> list[str]:
    out = capture(["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                   "Where-Object { $_.CommandLine -like '*train.py*' } | ForEach-Object { $_.ProcessId }"])
    return [p for p in out.split() if p.isdigit()]


def last_line(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        return lines[-1][:200] if lines else ""
    except OSError:
        return ""


def train(r: Runner, item: dict) -> bool:
    run, st = item["run"], r.status["runs"][item["run"]]
    while pids := other_training_pids():
        st.update(state="waiting_for_other_training", other_pids=pids)
        r.save()
        time.sleep(60)
    report_dir = REPO_ROOT / STUDY_REPORTS / run
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "train_stdout.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(item["cmd"], cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                creationflags=NO_WINDOW)
        st.update(state="running", started=now(), child_pid=proc.pid, log=str(log_path.relative_to(REPO_ROOT)))
        r.save()
        r.log(f"START {run} (pid {proc.pid}): {subprocess.list2cmdline(item['cmd'])}")
        t0 = time.time()
        while proc.poll() is None:
            st.update(elapsed_min=round((time.time() - t0) / 60, 1), last_log_line=last_line(log_path))
            r.save()
            time.sleep(30)
    st.update(returncode=proc.returncode, finished=now(), elapsed_min=round((time.time() - t0) / 60, 1),
              last_log_line=last_line(log_path))
    rj = report_dir / "run.json"
    j = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    ok = proc.returncode == 0 and run_state(run, item["seed"]) == "completed"
    if ok:
        st.update(state="completed", epochs_completed=j["epochs_completed"], best_epoch=j["best_epoch"],
                  stop_reason=j["stop_reason"])
        r.log(f"COMPLETE {run}: {j['epochs_completed']} epochs, best epoch {j['best_epoch']}, "
              f"{st['elapsed_min']} min; {j['stop_reason']}")
    else:
        st.update(state="failed", run_json_status=j.get("status"), error=j.get("error"))
        r.log(f"FAIL {run}: returncode {proc.returncode}, run.json status {j.get('status')!r}, "
              f"error {j.get('error')!r}; log kept at {log_path.relative_to(REPO_ROOT)}")
    r.save()
    return ok


def post_step(r: Runner, name: str, cmd: list[str]) -> bool:
    r.status["post"][name] = {"state": "running", "started": now(), "cmd": subprocess.list2cmdline(cmd)}
    r.save()
    r.log(f"START post step {name}")
    with open(RUNNER_DIR / f"post_{name}.log", "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, creationflags=NO_WINDOW,
                            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}).returncode
    r.status["post"][name].update(state="completed" if rc == 0 else "failed", returncode=rc, finished=now())
    r.save()
    r.log(f"{'COMPLETE' if rc == 0 else 'FAIL'} post step {name} (returncode {rc})")
    return rc == 0


def verify_reproduction(r: Runner, runs: list[str]) -> bool:
    """Checkpoint re-prediction (val_analysis.json) must match run.json's best-epoch validation results."""
    results = {}
    for run in runs:
        d = REPO_ROOT / STUDY_REPORTS / run
        rj = json.loads((d / "run.json").read_text(encoding="utf-8"))
        va = json.loads((d / "val_analysis.json").read_text(encoding="utf-8"))
        best, ov = rj["best_validation"], va["overall"]
        checks = {
            "split_is_val": va["split"] == "val",
            "checkpoint_epoch_is_best_epoch": va["checkpoint_epoch"] == rj["best_epoch"],
            "confusion_matrix_identical": ov["confusion_matrix"] == best["confusion_matrix"],
            "macro_f1_matches": abs(ov["macro_f1"] - best["macro_f1"]) < 1e-9,
            "balanced_accuracy_matches": abs(ov["balanced_accuracy"] - best["balanced_accuracy"]) < 1e-9,
            "accuracy_matches": abs(ov["accuracy"] - best["accuracy"]) < 1e-9,
            "reproduces_logged_val_confusion_matrix": va["reproduces_logged_val_confusion_matrix"] is True,
            "data_sha256_matches_checkpoint": va["data_sha256_matches_checkpoint"] is True,
            "test_split_not_evaluated": rj["test_split_evaluated"] is False,
        }
        results[run] = {"all_ok": all(checks.values()), **checks}
    ok = all(v["all_ok"] for v in results.values())
    (RUNNER_DIR / "verification.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    r.status["post"]["verify_reproduction"] = {"state": "completed" if ok else "failed", "finished": now()}
    r.save()
    r.log(f"{'COMPLETE' if ok else 'FAIL'} post step verify_reproduction: "
          + ", ".join(f"{k}={'ok' if v['all_ok'] else 'MISMATCH'}" for k, v in results.items()))
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the remaining seed-study runs sequentially, unattended.")
    p.add_argument("--dry-run", action="store_true", help="print the plan and commands; train nothing")
    a = p.parse_args(argv)
    plan = json.loads((REPO_ROOT / STUDY_REPORTS / "plan.json").read_text(encoding="utf-8"))
    items = planned_runs(plan)
    states = {i["run"]: run_state(i["run"], i["seed"]) for i in items}
    to_train = [i for i in items if states[i["run"]] == "absent"]
    if a.dry_run:
        for i in items:
            print(f"{i['run']:34s} {states[i['run']]:10s} {subprocess.list2cmdline(i['cmd'])}")
        print("to train:", [i["run"] for i in to_train], "matches expected:",
              [i["run"] for i in to_train] == EXPECTED_TO_TRAIN)
        return 0

    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(RUNNER_DIR / "runner.lock", "a+")
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("another seed-study runner holds the lock; exiting", file=sys.stderr)
        return 2
    r = Runner()
    try:
        # keep the system awake (not the display) while this thread runs; cleared automatically on exit
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        r.status.update(git_head=capture(["git", "rev-parse", "HEAD"]),
                        git_status=capture(["git", "status", "--porcelain"]).splitlines(),
                        python=str(PYTHON))
        r.log(f"RUNNER START pid {os.getpid()}; git HEAD {r.status['git_head']}")
        r.log("nvidia-smi: " + capture(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                                        "--format=csv,noheader"]))
        for i in items:
            r.status["runs"][i["run"]] = {"architecture": i["architecture"], "seed": i["seed"],
                                          "state": {"completed": "reused_completed", "absent": "pending",
                                                    "incomplete": "blocked_existing_incomplete_artifacts"}[states[i["run"]]],
                                          "cmd": subprocess.list2cmdline(i["cmd"])}
            if states[i["run"]] != "absent":
                r.log(f"{'REUSE' if states[i['run']] == 'completed' else 'BLOCKED (not overwritten)'} {i['run']}")
        if [i["run"] for i in to_train] != EXPECTED_TO_TRAIN:
            r.status["state"] = "aborted_unexpected_run_set"
            r.log(f"ABORT: runs to train {[i['run'] for i in to_train]} != expected {EXPECTED_TO_TRAIN}")
            r.save()
            return 1
        r.status["state"] = "training"
        r.save()
        for n, i in enumerate(to_train, 1):
            r.status["current"] = f"{n}/{len(to_train)} {i['run']}"
            train(r, i)
        r.status.pop("current", None)
        failed = [k for k, v in r.status["runs"].items() if v["state"] not in ("completed", "reused_completed")]
        if failed:
            r.status["state"] = "finished_with_failures"
            r.log(f"TRAINING FINISHED WITH FAILURES: {failed}; post-training steps skipped")
            r.save()
            return 1

        r.status["state"] = "post_training"
        r.save()
        new_runs = [i["run"] for i in items]
        steps_ok = [post_step(r, "analyze_run", [str(PYTHON), "src/evaluation/analyze_run.py", *new_runs,
                                                 "--checkpoint-dir", STUDY_MODELS, "--reports-dir", STUDY_REPORTS])]
        steps_ok.append(steps_ok[0] and verify_reproduction(r, new_runs))
        steps_ok.append(post_step(r, "unittest", [str(PYTHON), "-m", "unittest", "discover", "-s", "tests", "-v"]))
        steps_ok.append(post_step(r, "seed_study", [str(PYTHON), "src/evaluation/seed_study.py"]))
        r.status["state"] = "all_done" if all(steps_ok) else "done_with_post_step_failures"
        r.log(f"RUNNER END: {r.status['state']}")
        r.save()
        return 0 if all(steps_ok) else 1
    except BaseException:
        r.status["state"] = "runner_crashed"
        r.log("RUNNER CRASHED:\n" + traceback.format_exc())
        r.save()
        raise


if __name__ == "__main__":
    sys.exit(main())
