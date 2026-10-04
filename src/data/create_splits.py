"""Create a patient-level train/val/test split from data/metadata.csv.

Every patient is assigned to exactly one split, so no patient's recordings or
respiratory cycles can appear in more than one split. Uses only the Python
standard library.

Methods (``--method``):

patient-balanced (default)
  Primary use: evaluation on unseen patients. The ratios set exact *patient*
  quotas (e.g. 0.70/0.15/0.15 of 126 patients -> 88/19/19). The script draws
  ``--candidates`` random patient permutations from a seeded RNG, cuts each one
  into those quotas, and keeps the candidate with the lowest score:

      score = 1.00 * worst-split TVD of cycle-label distribution
            + 1.00 * worst-split TVD of device distribution (by cycles)
            + 0.25 * worst-split TVD of sample-rate distribution (by cycles)
            + 0.25 * worst-split |cycle share - patient share|

  TVD is total variation distance between one split's distribution and the
  whole dataset's (0 = identical, 1 = disjoint). "Worst split" is the maximum
  over train/val/test. Sample rate and cycle balance get lower weights because
  sample rate largely follows device and cycle balance is secondary. Ties go
  to the earliest candidate, so the result is fully determined by the seed and
  the candidate count.

cycle-greedy (original method, kept for comparison)
  The ratios are targets for *cycle* counts. Patients are stratified by label
  profile and assigned largest-first to the split furthest below its target.
  This balances cycle counts but concentrates large patients (mostly COPD /
  AKGC417L) in train, which shifts device mix between splits.

Diagnosis is NOT used by either method and is not a prediction target. It is
only reported. In ICBHI 2017, recording device and diagnosis are confounded
(e.g., every Healthy recording is from the Meditron stethoscope, and AKGC417L /
Litt3200 were used only on COPD patients). Balancing device mix reduces, but
cannot remove, this confound. Device and diagnosis distributions are reported
for every split so it can be checked.

Outputs (in --output-dir, default data/splits/):
  patient_split.csv   one row per patient: patient_id, split, label_profile,
                      n_recordings, n_cycles
  split_summary.json  configuration, score components, and per-split statistics

Usage::

    python src/data/create_splits.py [--metadata FILE] [--output-dir DIR]
                                     [--method {patient-balanced,cycle-greedy}]
                                     [--seed N] [--candidates N]
                                     [--ratios TRAIN VAL TEST]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "data" / "metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_CANDIDATES = 5000

METHODS = ("patient-balanced", "cycle-greedy")
DEFAULT_METHOD = "patient-balanced"
SCORE_WEIGHTS = {"label_tvd": 1.0, "device_tvd": 1.0, "sample_rate_tvd": 0.25, "cycle_share_gap": 0.25}

SPLITS = ("train", "val", "test")
LABELS = ("normal", "crackle", "wheeze", "both")
SPLIT_FILE = "patient_split.csv"
SUMMARY_FILE = "split_summary.json"
REQUIRED_COLUMNS = ("recording_filename", "patient_id", "diagnosis", "device", "sample_rate", "cycle_label")


class SplitIntegrityError(Exception):
    """Raised when a split leaks patients across splits or is incomplete."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"{len(errors)} split integrity error(s): " + "; ".join(errors[:5]))


@dataclass(frozen=True)
class PatientSummary:
    patient_id: str
    n_cycles: int
    n_recordings: int
    label_counts: tuple[int, int, int, int]  # in LABELS order
    device_cycles: dict[str, int]
    sample_rate_cycles: dict[str, int]

    @property
    def label_profile(self) -> str:
        return label_profile(dict(zip(LABELS, self.label_counts)))


def label_profile(label_counts: dict[str, int]) -> str:
    """Return a patient's stratum based on which adventitious sounds they have."""
    has_crackle = label_counts.get("crackle", 0) + label_counts.get("both", 0) > 0
    has_wheeze = label_counts.get("wheeze", 0) + label_counts.get("both", 0) > 0
    if has_crackle and has_wheeze:
        return "crackle+wheeze"
    if has_crackle:
        return "crackle"
    if has_wheeze:
        return "wheeze"
    return "none"


def _sort_key(value: str):
    return (0, int(value), "") if value.isdigit() else (1, 0, value)


def parse_ratios(values: Iterable[float]) -> dict[str, float]:
    """Validate three positive ratios that sum to 1 and key them by split name."""
    values = [float(v) for v in values]
    if len(values) != len(SPLITS):
        raise ValueError(f"expected {len(SPLITS)} ratios (train val test), got {len(values)}")
    if any(v <= 0 for v in values):
        raise ValueError(f"ratios must all be positive, got {values}")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {values} (sum={sum(values)})")
    return dict(zip(SPLITS, values))


def patient_quotas(n_patients: int, ratios: dict[str, float]) -> dict[str, int]:
    """Exact patient counts per split; val/test are rounded, train takes the rest."""
    quotas = {s: round(ratios[s] * n_patients) for s in SPLITS[1:]}
    quotas = {"train": n_patients - sum(quotas.values()), **quotas}
    if any(q < 1 for q in quotas.values()):
        raise ValueError(f"{n_patients} patients cannot fill every split with ratios {ratios}: {quotas}")
    return quotas


def load_metadata(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    unknown = sorted({r["cycle_label"] for r in rows} - set(LABELS))
    if unknown:
        raise ValueError(f"{path} contains unknown cycle labels: {unknown}")
    return rows


def summarize_patients(rows: list[dict[str, str]]) -> dict[str, PatientSummary]:
    labels: dict[str, Counter] = defaultdict(Counter)
    devices: dict[str, Counter] = defaultdict(Counter)
    rates: dict[str, Counter] = defaultdict(Counter)
    recordings: dict[str, set] = defaultdict(set)
    for r in rows:
        pid = r["patient_id"]
        labels[pid][r["cycle_label"]] += 1
        devices[pid][r["device"]] += 1
        rates[pid][r["sample_rate"]] += 1
        recordings[pid].add(r["recording_filename"])
    return {
        pid: PatientSummary(
            patient_id=pid,
            n_cycles=sum(labels[pid].values()),
            n_recordings=len(recordings[pid]),
            label_counts=tuple(labels[pid][label] for label in LABELS),
            device_cycles=dict(sorted(devices[pid].items())),
            sample_rate_cycles=dict(sorted(rates[pid].items(), key=lambda kv: _sort_key(kv[0]))),
        )
        for pid in sorted(labels)
    }


def _tvd(counts: Counter, reference: Counter) -> float:
    """Total variation distance between two count distributions."""
    n, m = sum(counts.values()), sum(reference.values())
    if n == 0 or m == 0:
        return 1.0
    keys = set(counts) | set(reference)
    return 0.5 * sum(abs(counts.get(k, 0) / n - reference.get(k, 0) / m) for k in keys)


def score_split(patients: dict[str, PatientSummary], assignment: dict[str, str],
                ratios: dict[str, float]) -> dict[str, float]:
    """Score how closely each split matches the whole dataset (lower is better).

    ``ratios`` are the patient-share targets used for the cycle-share gap.
    Returns the worst-split value of each component plus the weighted total.
    """
    per_split = {s: {"label": Counter(), "device": Counter(), "rate": Counter()} for s in SPLITS}
    overall = {"label": Counter(), "device": Counter(), "rate": Counter()}
    for pid, p in patients.items():
        for target in (per_split[assignment[pid]], overall):
            target["label"].update(dict(zip(LABELS, p.label_counts)))
            target["device"].update(p.device_cycles)
            target["rate"].update(p.sample_rate_cycles)

    total_cycles = sum(overall["label"].values())
    components = {
        "label_tvd": max(_tvd(per_split[s]["label"], overall["label"]) for s in SPLITS),
        "device_tvd": max(_tvd(per_split[s]["device"], overall["device"]) for s in SPLITS),
        "sample_rate_tvd": max(_tvd(per_split[s]["rate"], overall["rate"]) for s in SPLITS),
        "cycle_share_gap": max(
            abs(sum(per_split[s]["label"].values()) / total_cycles - ratios[s]) for s in SPLITS
        ),
    }
    components["total"] = sum(SCORE_WEIGHTS[k] * v for k, v in components.items())
    return components


def assign_patient_balanced(patients: dict[str, PatientSummary], ratios: dict[str, float],
                            seed: int, n_candidates: int = DEFAULT_CANDIDATES) -> dict[str, str]:
    """Best of ``n_candidates`` seeded random splits with exact patient quotas."""
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    quotas = patient_quotas(len(patients), ratios)
    labels = [s for s in SPLITS for _ in range(quotas[s])]
    pids = sorted(patients)
    rng = random.Random(seed)

    best, best_score = None, None
    for _ in range(n_candidates):
        order = pids[:]
        rng.shuffle(order)
        candidate = dict(zip(order, labels))
        score = score_split(patients, candidate, ratios)["total"]
        if best_score is None or score < best_score:
            best, best_score = candidate, score
    return {pid: best[pid] for pid in pids}


def assign_cycle_greedy(patients: dict[str, PatientSummary], ratios: dict[str, float],
                        seed: int) -> dict[str, str]:
    """Original method: label-profile strata, largest-first fill toward cycle targets."""
    rng = random.Random(seed)
    tiebreak = {pid: rng.random() for pid in sorted(patients)}

    strata: dict[str, list[str]] = defaultdict(list)
    for pid in sorted(patients):
        strata[patients[pid].label_profile].append(pid)

    assignment: dict[str, str] = {}
    for stratum in sorted(strata):
        pids = sorted(strata[stratum], key=lambda p: (-patients[p].n_cycles, tiebreak[p]))
        total = sum(patients[p].n_cycles for p in pids)
        filled = dict.fromkeys(SPLITS, 0)
        for pid in pids:
            # Split furthest below its target cycle count; ties go to the earlier split.
            split = max(SPLITS, key=lambda s: ratios[s] * total - filled[s])
            assignment[pid] = split
            filled[split] += patients[pid].n_cycles
    return assignment


def assign_splits(patients: dict[str, PatientSummary], ratios: dict[str, float], seed: int,
                  method: str = DEFAULT_METHOD, n_candidates: int = DEFAULT_CANDIDATES) -> dict[str, str]:
    """Assign each patient to one split. Deterministic for given arguments."""
    if method == "patient-balanced":
        assignment = assign_patient_balanced(patients, ratios, seed, n_candidates)
    elif method == "cycle-greedy":
        assignment = assign_cycle_greedy(patients, ratios, seed)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    verify_split(assignment.items(), set(patients))
    return assignment


def verify_split(pairs: Iterable[tuple[str, str]], expected_patients: set[str]) -> None:
    """Fail loudly unless every expected patient is in exactly one valid split.

    ``pairs`` is (patient_id, split) entries, e.g. rows read back from disk.
    Every split must also contain at least one patient.
    """
    errors: list[str] = []
    seen: dict[str, set[str]] = defaultdict(set)
    counts: Counter = Counter()
    for pid, split in pairs:
        if split not in SPLITS:
            errors.append(f"patient {pid} has unknown split {split!r}")
        seen[pid].add(split)
        counts[pid] += 1

    for pid in sorted(seen):
        if len(seen[pid]) > 1:
            errors.append(f"LEAKAGE: patient {pid} appears in multiple splits {sorted(seen[pid])}")
        elif counts[pid] > 1:
            errors.append(f"patient {pid} is listed {counts[pid]} times")

    missing = sorted(expected_patients - set(seen))
    extra = sorted(set(seen) - expected_patients)
    if missing:
        errors.append(f"{len(missing)} patient(s) not assigned to any split: {missing[:10]}")
    if extra:
        errors.append(f"{len(extra)} assigned patient(s) not in metadata: {extra[:10]}")

    populated = {s for splits in seen.values() for s in splits}
    for split in SPLITS:
        if split not in populated:
            errors.append(f"split {split!r} has no patients")

    if errors:
        raise SplitIntegrityError(errors)


def _fractions(counter: Counter, keys: list[str]) -> dict[str, float]:
    n = sum(counter.values())
    return {k: round(counter.get(k, 0) / n, 4) if n else 0.0 for k in keys}


def split_statistics(rows: list[dict[str, str]], assignment: dict[str, str]) -> dict[str, dict]:
    """Per-split counts and distributions, plus an 'all' entry for comparison."""
    groups: dict[str, list[dict[str, str]]] = {s: [] for s in (*SPLITS, "all")}
    for r in rows:
        groups[assignment[r["patient_id"]]].append(r)
        groups["all"].append(r)

    devices = sorted({r["device"] for r in rows})
    rates = sorted({r["sample_rate"] for r in rows}, key=_sort_key)
    n_patients_total = len({r["patient_id"] for r in rows})
    stats = {}
    for name, group in groups.items():
        recordings = {r["recording_filename"]: r for r in group}
        patients = {r["patient_id"]: r for r in group}
        labels = Counter(r["cycle_label"] for r in group)
        device_cycles = Counter(r["device"] for r in group)
        rate_cycles = Counter(r["sample_rate"] for r in group)
        stats[name] = {
            "patients": len(patients),
            "patient_fraction": round(len(patients) / n_patients_total, 4),
            "recordings": len(recordings),
            "cycles": len(group),
            "cycle_fraction": round(len(group) / len(rows), 4),
            "cycle_labels": {label: labels.get(label, 0) for label in LABELS},
            "cycle_label_fractions": _fractions(labels, list(LABELS)),
            "cycles_by_device": {d: device_cycles.get(d, 0) for d in devices},
            "cycle_fractions_by_device": _fractions(device_cycles, devices),
            "recordings_by_device": dict(sorted(Counter(r["device"] for r in recordings.values()).items())),
            "cycles_by_sample_rate": {r: rate_cycles.get(r, 0) for r in rates},
            "cycle_fractions_by_sample_rate": _fractions(rate_cycles, rates),
            "patients_by_diagnosis": dict(sorted(Counter(r["diagnosis"] for r in patients.values()).items())),
            "recordings_by_diagnosis": dict(sorted(Counter(r["diagnosis"] for r in recordings.values()).items())),
        }
    return stats


def write_outputs(output_dir: Path, patients: dict[str, PatientSummary], assignment: dict[str, str],
                  stats: dict, config: dict, score: dict[str, float]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / SPLIT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["patient_id", "split", "label_profile", "n_recordings", "n_cycles"])
        for pid in sorted(patients):
            p = patients[pid]
            writer.writerow([pid, assignment[pid], p.label_profile, p.n_recordings, p.n_cycles])

    summary = {
        **config,
        "diagnosis_used_for_split": False,
        "score_weights": SCORE_WEIGHTS,
        "score": {k: round(v, 4) for k, v in score.items()},
        "splits": stats,
    }
    with open(output_dir / SUMMARY_FILE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def read_split_file(path: Path) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["patient_id"], r["split"]) for r in csv.DictReader(f)]


def format_summary(stats: dict, score: dict[str, float]) -> str:
    names = (*SPLITS, "all")
    lines = ["split   patients (share)  recordings  cycles (share)"]
    for name in names:
        s = stats[name]
        lines.append(f"{name:<7} {s['patients']:>8} ({s['patient_fraction']:>6.1%})  {s['recordings']:>10}"
                     f"  {s['cycles']:>6} ({s['cycle_fraction']:>6.1%})")

    def table(title: str, key: str) -> list[str]:
        cats = list(stats["all"][key])
        width = max(len(c) for c in cats)
        out = [title, "  " + "".ljust(width) + "".join(f"{n:>8}" for n in names)]
        for c in cats:
            cells = "".join(
                f"{stats[n][key].get(c, 0):>8.1%}" if isinstance(stats[n][key].get(c, 0), float)
                else f"{stats[n][key].get(c, 0):>8}" for n in names
            )
            out.append("  " + c.ljust(width) + cells)
        return out

    lines += table("Cycle labels (% of split cycles):", "cycle_label_fractions")
    lines += table("Device (% of split cycles):", "cycle_fractions_by_device")
    lines += table("Sample rate Hz (% of split cycles):", "cycle_fractions_by_sample_rate")
    lines += table("Patients by diagnosis (reported only, not used for splitting):", "patients_by_diagnosis")
    lines.append("Worst-split distances vs. whole dataset: " + ", ".join(
        f"{k}={v:.3f}" for k, v in score.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                        help="per-cycle metadata CSV (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="directory for split files (default: %(default)s)")
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD,
                        help="split method (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default: %(default)s)")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES,
                        help="candidates evaluated by patient-balanced (default: %(default)s)")
    parser.add_argument("--ratios", type=float, nargs=3, default=DEFAULT_RATIOS,
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="patient ratios (patient-balanced) or cycle ratios (cycle-greedy); "
                             "must sum to 1 (default: 0.70 0.15 0.15)")
    args = parser.parse_args(argv)

    try:
        ratios = parse_ratios(args.ratios)
        rows = load_metadata(args.metadata)
        patients = summarize_patients(rows)
        assignment = assign_splits(patients, ratios, args.seed, args.method, args.candidates)
        stats = split_statistics(rows, assignment)
        score = score_split(patients, assignment, ratios)
        config = {
            "method": args.method,
            "seed": args.seed,
            "candidates": args.candidates if args.method == "patient-balanced" else None,
            "target_ratios": ratios,
            "ratio_basis": "patients" if args.method == "patient-balanced" else "cycles",
            "metadata": Path(args.metadata).name,
        }
        write_outputs(args.output_dir, patients, assignment, stats, config, score)
        # Re-check what was actually written to disk.
        verify_split(read_split_file(Path(args.output_dir) / SPLIT_FILE), set(patients))
    except (ValueError, OSError, SplitIntegrityError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Patient-level split (method={args.method}, seed={args.seed})")
    print(format_summary(stats, score))
    print(f"Wrote {Path(args.output_dir) / SPLIT_FILE} and {Path(args.output_dir) / SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
