"""Build a per-respiratory-cycle metadata table for the ICBHI 2017 dataset.

Reads the raw dataset in place (nothing is moved, renamed, or modified) and
writes one CSV row per annotated respiratory cycle. Uses only the Python
standard library.

Expected dataset layout (``--dataset-dir``)::

    <dataset-dir>/
    ├── audio_and_txt_files/   # <stem>.wav + <stem>.txt pairs
    └── patient_diagnosis.csv  # "<patient_id>,<diagnosis>" rows, no header

Integrity checks (any failure aborts without writing output):
  * every .wav has a matching .txt annotation and vice versa
  * every filename follows the ICBHI naming convention
  * every patient ID appears exactly once in patient_diagnosis.csv
  * every .wav header is readable
  * every annotation line has 4 tab-separated fields, numeric times,
    start >= 0, end > start, end within the recording, cycles in
    chronological non-overlapping order, and crackle/wheeze labels in {0, 1}

Known dataset caveat (documented, NOT corrected here):
  In ICBHI 2017, diagnosis is strongly confounded with recording device and
  acquisition mode. All Healthy, URTI, Bronchiectasis, Bronchiolitis, and LRTI
  recordings come from the Meditron stethoscope in single-channel mode, while
  the AKGC417L and Litt3200 devices were used only on COPD patients. A model
  may therefore learn device/channel artifacts instead of respiratory sounds.
  The summary printed by this script includes a device x diagnosis table so
  the confound stays visible; handling it is deferred to the splitting and
  evaluation stages.

Usage::

    python src/data/build_metadata.py [--dataset-dir DIR] [--output FILE]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = (
    REPO_ROOT / "data" / "raw" / "Respiratory_Sound_Database" / "Respiratory_Sound_Database"
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "metadata.csv"
AUDIO_SUBDIR = "audio_and_txt_files"
DIAGNOSIS_FILE = "patient_diagnosis.csv"

# Tolerance (seconds) for annotations ending slightly past the audio end.
DURATION_TOLERANCE_S = 0.01

CHEST_LOCATIONS = ("Tc", "Al", "Ar", "Pl", "Pr", "Ll", "Lr")
ACQUISITION_MODES = ("sc", "mc")
DEVICES = ("AKGC417L", "LittC2SE", "Litt3200", "Meditron")

FILENAME_RE = re.compile(
    r"^(?P<patient_id>\d+)"
    r"_(?P<recording_index>\d+[a-z]\d+)"
    rf"_(?P<chest_location>{'|'.join(CHEST_LOCATIONS)})"
    rf"_(?P<acquisition_mode>{'|'.join(ACQUISITION_MODES)})"
    rf"_(?P<device>{'|'.join(DEVICES)})$"
)

COLUMNS = [
    "recording_filename",
    "cycle_index",
    "patient_id",
    "diagnosis",
    "recording_index",
    "chest_location",
    "acquisition_mode",
    "device",
    "sample_rate",
    "channels",
    "bit_depth",
    "recording_duration_s",
    "cycle_start_s",
    "cycle_end_s",
    "cycle_duration_s",
    "crackle",
    "wheeze",
    "cycle_label",
]


class MetadataValidationError(Exception):
    """Raised when the raw dataset fails one or more integrity checks."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"{len(errors)} integrity error(s) found")


@dataclass(frozen=True)
class Cycle:
    start_s: float
    end_s: float
    crackle: int
    wheeze: int


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    bit_depth: int
    duration_s: float


def parse_filename(stem: str) -> dict[str, str]:
    """Split an ICBHI recording stem into its named fields.

    Raises ValueError if the stem does not follow the naming convention.
    """
    match = FILENAME_RE.match(stem)
    if match is None:
        raise ValueError(f"filename does not match ICBHI convention: {stem!r}")
    return match.groupdict()


def cycle_label(crackle: int, wheeze: int) -> str:
    """Map crackle/wheeze flags to a single categorical label."""
    if crackle not in (0, 1) or wheeze not in (0, 1):
        raise ValueError(f"labels must be 0 or 1, got crackle={crackle}, wheeze={wheeze}")
    return {(0, 0): "normal", (1, 0): "crackle", (0, 1): "wheeze", (1, 1): "both"}[(crackle, wheeze)]


def parse_annotation(text: str, source: str = "<annotation>") -> tuple[list[Cycle], list[str]]:
    """Parse annotation text into cycles, returning (cycles, errors).

    Blank lines are ignored. Each remaining line must be
    ``start<TAB>end<TAB>crackle<TAB>wheeze``.
    """
    cycles: list[Cycle] = []
    errors: list[str] = []
    prev_end: float | None = None
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{source}:{line_no}"
        fields = line.strip().split("\t")
        if len(fields) != 4:
            errors.append(f"{where}: expected 4 tab-separated fields, got {len(fields)}: {line!r}")
            continue
        try:
            start, end = float(fields[0]), float(fields[1])
        except ValueError:
            errors.append(f"{where}: non-numeric cycle time: {line!r}")
            continue
        try:
            crackle, wheeze = int(fields[2]), int(fields[3])
        except ValueError:
            errors.append(f"{where}: non-integer label: {line!r}")
            continue
        if crackle not in (0, 1) or wheeze not in (0, 1):
            errors.append(f"{where}: labels must be 0 or 1, got crackle={crackle}, wheeze={wheeze}")
            continue
        if start < 0:
            errors.append(f"{where}: negative start time {start}")
            continue
        if end <= start:
            errors.append(f"{where}: end time {end} is not after start time {start}")
            continue
        if prev_end is not None and start < prev_end:
            errors.append(f"{where}: cycle starts at {start}, before previous cycle ends at {prev_end}")
        prev_end = end
        cycles.append(Cycle(start, end, crackle, wheeze))
    if not cycles and not errors:
        errors.append(f"{source}: annotation file contains no cycles")
    return cycles, errors


def load_diagnoses(path: Path) -> tuple[dict[str, str], list[str]]:
    """Read patient_diagnosis.csv (no header) into {patient_id: diagnosis}."""
    diagnoses: dict[str, str] = {}
    errors: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row_no, row in enumerate(csv.reader(f), start=1):
            if not row or not "".join(row).strip():
                continue
            if len(row) != 2:
                errors.append(f"{path.name}:{row_no}: expected 2 columns, got {len(row)}: {row!r}")
                continue
            patient_id, diagnosis = row[0].strip(), row[1].strip()
            if not patient_id.isdigit() or not diagnosis:
                errors.append(f"{path.name}:{row_no}: malformed row {row!r}")
                continue
            if patient_id in diagnoses:
                errors.append(f"{path.name}:{row_no}: duplicate patient ID {patient_id}")
                continue
            diagnoses[patient_id] = diagnosis
    return diagnoses, errors


def read_wav_info(path: Path) -> WavInfo:
    """Read sample rate, channels, bit depth, and duration from a WAV header."""
    with wave.open(str(path), "rb") as w:
        sample_rate = w.getframerate()
        return WavInfo(
            sample_rate=sample_rate,
            channels=w.getnchannels(),
            bit_depth=w.getsampwidth() * 8,
            duration_s=w.getnframes() / sample_rate,
        )


def check_pairs(wav_stems: set[str], txt_stems: set[str]) -> list[str]:
    """Report recordings missing an annotation and annotations missing audio."""
    errors = [f"{s}.wav has no matching annotation {s}.txt" for s in sorted(wav_stems - txt_stems)]
    errors += [f"{s}.txt has no matching audio {s}.wav" for s in sorted(txt_stems - wav_stems)]
    return errors


def build_metadata(dataset_dir: Path) -> list[dict[str, object]]:
    """Validate the dataset and return one metadata row per respiratory cycle.

    Rows are ordered by recording filename, then by cycle order within the
    annotation file. Raises MetadataValidationError listing every problem found.
    """
    dataset_dir = Path(dataset_dir)
    audio_dir = dataset_dir / AUDIO_SUBDIR
    diagnosis_path = dataset_dir / DIAGNOSIS_FILE
    missing = [str(p) for p in (audio_dir, diagnosis_path) if not p.exists()]
    if missing:
        raise MetadataValidationError([f"required path not found: {p}" for p in missing])

    errors: list[str] = []
    diagnoses, diag_errors = load_diagnoses(diagnosis_path)
    errors += diag_errors

    wav_stems = {p.stem for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"}
    txt_stems = {p.stem for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
    errors += check_pairs(wav_stems, txt_stems)

    rows: list[dict[str, object]] = []
    for stem in sorted(wav_stems & txt_stems):
        try:
            fields = parse_filename(stem)
        except ValueError as e:
            errors.append(str(e))
            continue

        diagnosis = diagnoses.get(fields["patient_id"])
        if diagnosis is None:
            errors.append(f"{stem}: patient ID {fields['patient_id']} not found in {DIAGNOSIS_FILE}")
            continue

        try:
            info = read_wav_info(audio_dir / f"{stem}.wav")
        except (wave.Error, EOFError, OSError, ZeroDivisionError) as e:
            errors.append(f"{stem}.wav: unreadable WAV header ({e})")
            continue

        text = (audio_dir / f"{stem}.txt").read_text(encoding="utf-8")
        cycles, cycle_errors = parse_annotation(text, source=f"{stem}.txt")
        errors += cycle_errors

        for i, cycle in enumerate(cycles):
            if cycle.end_s > info.duration_s + DURATION_TOLERANCE_S:
                errors.append(
                    f"{stem}.txt: cycle {i} ends at {cycle.end_s}s, "
                    f"past recording end {info.duration_s:.3f}s"
                )
                continue
            rows.append({
                "recording_filename": f"{stem}.wav",
                "cycle_index": i,
                "patient_id": fields["patient_id"],
                "diagnosis": diagnosis,
                "recording_index": fields["recording_index"],
                "chest_location": fields["chest_location"],
                "acquisition_mode": fields["acquisition_mode"],
                "device": fields["device"],
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "bit_depth": info.bit_depth,
                "recording_duration_s": round(info.duration_s, 6),
                "cycle_start_s": cycle.start_s,
                "cycle_end_s": cycle.end_s,
                "cycle_duration_s": round(cycle.end_s - cycle.start_s, 6),
                "crackle": cycle.crackle,
                "wheeze": cycle.wheeze,
                "cycle_label": cycle_label(cycle.crackle, cycle.wheeze),
            })

    if errors:
        raise MetadataValidationError(errors)
    return rows


def write_metadata(rows: list[dict[str, object]], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> str:
    """Return a short human-readable summary of the metadata table."""
    recordings = {r["recording_filename"]: r for r in rows}
    patient_diag = {r["patient_id"]: r["diagnosis"] for r in rows}
    labels = Counter(r["cycle_label"] for r in rows)
    rates = Counter(r["sample_rate"] for r in recordings.values())
    total_hours = sum(r["recording_duration_s"] for r in recordings.values()) / 3600

    lines = [
        f"Recordings: {len(recordings)} | Patients: {len(patient_diag)} | "
        f"Cycles: {len(rows)} | Audio: {total_hours:.2f} h",
        "Cycle labels: " + ", ".join(
            f"{k}={labels.get(k, 0)}" for k in ("normal", "crackle", "wheeze", "both")
        ),
        "Patients per diagnosis: " + ", ".join(
            f"{d}={n}" for d, n in sorted(Counter(patient_diag.values()).items(), key=lambda x: (-x[1], x[0]))
        ),
        "Sample rates (recordings): " + ", ".join(f"{sr} Hz={n}" for sr, n in sorted(rates.items())),
    ]

    # Device x diagnosis (recording counts) to keep the known confound visible.
    table: dict[str, Counter] = defaultdict(Counter)
    for r in recordings.values():
        table[r["diagnosis"]][r["device"]] += 1
    devices = sorted({r["device"] for r in recordings.values()})
    width = max(len(d) for d in table) if table else 9
    lines.append("Recordings by diagnosis x device (note: device is confounded with diagnosis):")
    lines.append("  " + "".ljust(width) + "".join(d.rjust(10) for d in devices))
    for diagnosis in sorted(table):
        lines.append(
            "  " + diagnosis.ljust(width) + "".join(str(table[diagnosis][d]).rjust(10) for d in devices)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
        help=f"directory containing {AUDIO_SUBDIR}/ and {DIAGNOSIS_FILE} (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="output CSV path (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        rows = build_metadata(args.dataset_dir)
    except MetadataValidationError as e:
        print(f"ERROR: {e}. No output written.", file=sys.stderr)
        for msg in e.errors[:50]:
            print(f"  - {msg}", file=sys.stderr)
        if len(e.errors) > 50:
            print(f"  ... and {len(e.errors) - 50} more", file=sys.stderr)
        return 1

    write_metadata(rows, args.output)
    print(summarize(rows))
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
