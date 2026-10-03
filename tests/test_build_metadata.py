"""Tests for src/data/build_metadata.py (standard-library unittest).

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import csv
import io
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))

import build_metadata as bm  # noqa: E402


def write_wav(path: Path, seconds: float, sample_rate: int = 4000, sampwidth: int = 2) -> None:
    n_frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00" * sampwidth * n_frames)


class SyntheticDataset:
    """Builds a tiny ICBHI-shaped dataset in a temporary directory."""

    def __init__(self, root: Path):
        self.root = root
        self.audio = root / bm.AUDIO_SUBDIR
        self.audio.mkdir(parents=True)
        self.set_diagnoses("101,URTI\r\n102,COPD\r\n")

    def set_diagnoses(self, text: str) -> None:
        (self.root / bm.DIAGNOSIS_FILE).write_text(text, encoding="utf-8")

    def add_recording(self, stem: str, annotation: str | None, seconds: float = 5.0,
                      sample_rate: int = 4000, sampwidth: int = 2, wav: bool = True) -> None:
        if wav:
            write_wav(self.audio / f"{stem}.wav", seconds, sample_rate, sampwidth)
        if annotation is not None:
            (self.audio / f"{stem}.txt").write_text(annotation, encoding="utf-8")


VALID_A = "0.0\t1.5\t0\t0\n1.5\t3.0\t1\t0\n3.0\t4.5\t0\t1\n"
VALID_B = "0.5\t2.0\t1\t1\n"


class TestParseFilename(unittest.TestCase):
    def test_valid_filename(self):
        self.assertEqual(
            bm.parse_filename("101_1b1_Al_sc_Meditron"),
            {"patient_id": "101", "recording_index": "1b1", "chest_location": "Al",
             "acquisition_mode": "sc", "device": "Meditron"},
        )

    def test_all_known_devices_and_locations(self):
        for device in bm.DEVICES:
            for loc in bm.CHEST_LOCATIONS:
                fields = bm.parse_filename(f"226_2p4_{loc}_mc_{device}")
                self.assertEqual((fields["device"], fields["chest_location"]), (device, loc))

    def test_invalid_filenames(self):
        for stem in ["101_1b1_Al_sc", "101_1b1_Xx_sc_Meditron", "101_1b1_Al_xx_Meditron",
                     "101_1b1_Al_sc_Unknown", "abc_1b1_Al_sc_Meditron", "101_1b1_Al_sc_Meditron_extra"]:
            with self.subTest(stem=stem), self.assertRaises(ValueError):
                bm.parse_filename(stem)


class TestCycleLabel(unittest.TestCase):
    def test_label_mapping(self):
        self.assertEqual(bm.cycle_label(0, 0), "normal")
        self.assertEqual(bm.cycle_label(1, 0), "crackle")
        self.assertEqual(bm.cycle_label(0, 1), "wheeze")
        self.assertEqual(bm.cycle_label(1, 1), "both")

    def test_invalid_flags(self):
        for c, w in [(2, 0), (0, -1), (1, 2)]:
            with self.subTest(c=c, w=w), self.assertRaises(ValueError):
                bm.cycle_label(c, w)


class TestParseAnnotation(unittest.TestCase):
    def test_valid_annotation(self):
        cycles, errors = bm.parse_annotation("0.036\t0.579\t0\t0\r\n0.579\t2.45\t1\t1\r\n\n")
        self.assertEqual(errors, [])
        self.assertEqual(cycles, [bm.Cycle(0.036, 0.579, 0, 0), bm.Cycle(0.579, 2.45, 1, 1)])

    def assert_single_error(self, text, fragment):
        _, errors = bm.parse_annotation(text, source="x.txt")
        self.assertEqual(len(errors), 1, errors)
        self.assertIn(fragment, errors[0])

    def test_wrong_field_count(self):
        self.assert_single_error("0.0\t1.0\t0\n", "expected 4")

    def test_non_numeric_time(self):
        self.assert_single_error("a\t1.0\t0\t0\n", "non-numeric")

    def test_non_integer_label(self):
        self.assert_single_error("0.0\t1.0\t0.5\t0\n", "non-integer")

    def test_label_out_of_range(self):
        self.assert_single_error("0.0\t1.0\t2\t0\n", "labels must be 0 or 1")

    def test_negative_start(self):
        self.assert_single_error("-0.1\t1.0\t0\t0\n", "negative start")

    def test_end_not_after_start(self):
        self.assert_single_error("1.0\t1.0\t0\t0\n", "not after start")
        self.assert_single_error("2.0\t1.0\t0\t0\n", "not after start")

    def test_overlapping_cycles(self):
        self.assert_single_error("0.0\t2.0\t0\t0\n1.5\t3.0\t0\t0\n", "before previous cycle ends")

    def test_empty_annotation(self):
        self.assert_single_error("\n\n", "no cycles")


class TestLoadDiagnoses(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "patient_diagnosis.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, text):
        self.path.write_text(text, encoding="utf-8")
        return bm.load_diagnoses(self.path)

    def test_valid_without_trailing_newline(self):
        diagnoses, errors = self.load("101,URTI\r\n102,Healthy")
        self.assertEqual(errors, [])
        self.assertEqual(diagnoses, {"101": "URTI", "102": "Healthy"})

    def test_duplicate_patient(self):
        _, errors = self.load("101,URTI\n101,COPD\n")
        self.assertTrue(any("duplicate patient ID 101" in e for e in errors))

    def test_wrong_column_count(self):
        _, errors = self.load("101,URTI,extra\n")
        self.assertTrue(any("expected 2 columns" in e for e in errors))


class TestBuildMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ds = SyntheticDataset(Path(self.tmp.name) / "dataset")
        self.ds.add_recording("101_1b1_Al_sc_Meditron", VALID_A, seconds=5.0, sample_rate=4000)
        self.ds.add_recording("102_2b3_Tc_mc_AKGC417L", VALID_B, seconds=3.0, sample_rate=8000, sampwidth=3)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_validation_error(self, fragment):
        with self.assertRaises(bm.MetadataValidationError) as ctx:
            bm.build_metadata(self.ds.root)
        self.assertTrue(any(fragment in e for e in ctx.exception.errors), ctx.exception.errors)

    def test_rows_and_fields(self):
        rows = bm.build_metadata(self.ds.root)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(list(r) == bm.COLUMNS for r in rows))

        first, last = rows[0], rows[-1]
        self.assertEqual(first["recording_filename"], "101_1b1_Al_sc_Meditron.wav")
        self.assertEqual((first["patient_id"], first["diagnosis"]), ("101", "URTI"))
        self.assertEqual((first["sample_rate"], first["channels"], first["bit_depth"]), (4000, 1, 16))
        self.assertEqual(first["recording_duration_s"], 5.0)
        self.assertEqual([r["cycle_label"] for r in rows[:3]], ["normal", "crackle", "wheeze"])
        self.assertEqual([r["cycle_index"] for r in rows[:3]], [0, 1, 2])

        self.assertEqual(last["recording_filename"], "102_2b3_Tc_mc_AKGC417L.wav")
        self.assertEqual((last["recording_index"], last["chest_location"]), ("2b3", "Tc"))
        self.assertEqual((last["acquisition_mode"], last["device"]), ("mc", "AKGC417L"))
        self.assertEqual((last["sample_rate"], last["bit_depth"]), (8000, 24))
        self.assertEqual((last["cycle_start_s"], last["cycle_end_s"]), (0.5, 2.0))
        self.assertEqual(last["cycle_duration_s"], 1.5)
        self.assertEqual((last["crackle"], last["wheeze"], last["cycle_label"]), (1, 1, "both"))

    def test_missing_annotation(self):
        self.ds.add_recording("101_1b2_Pr_sc_Meditron", annotation=None)
        self.assert_validation_error("has no matching annotation")

    def test_missing_audio(self):
        self.ds.add_recording("101_1b2_Pr_sc_Meditron", VALID_A, wav=False)
        self.assert_validation_error("has no matching audio")

    def test_unknown_patient(self):
        self.ds.add_recording("999_1b1_Al_sc_Meditron", VALID_A)
        self.assert_validation_error("patient ID 999 not found")

    def test_bad_filename(self):
        self.ds.add_recording("101_1b1_Al_sc_NewDevice", VALID_A)
        self.assert_validation_error("does not match ICBHI convention")

    def test_cycle_past_recording_end(self):
        self.ds.add_recording("101_1b2_Pr_sc_Meditron", "0.0\t6.0\t0\t0\n", seconds=5.0)
        self.assert_validation_error("past recording end")

    def test_invalid_cycle_in_file(self):
        self.ds.add_recording("101_1b2_Pr_sc_Meditron", "0.0\t1.0\t0\t3\n")
        self.assert_validation_error("labels must be 0 or 1")

    def test_unreadable_wav(self):
        (self.ds.audio / "101_1b2_Pr_sc_Meditron.wav").write_bytes(b"not a wav file")
        (self.ds.audio / "101_1b2_Pr_sc_Meditron.txt").write_text(VALID_A, encoding="utf-8")
        self.assert_validation_error("unreadable WAV header")

    def test_reports_all_errors(self):
        self.ds.add_recording("999_1b1_Al_sc_Meditron", VALID_A)
        self.ds.add_recording("101_1b2_Pr_sc_Meditron", annotation=None)
        with self.assertRaises(bm.MetadataValidationError) as ctx:
            bm.build_metadata(self.ds.root)
        self.assertEqual(len(ctx.exception.errors), 2)

    def test_missing_dataset_dir(self):
        with self.assertRaises(bm.MetadataValidationError):
            bm.build_metadata(Path(self.tmp.name) / "does_not_exist")


class TestMainAndDeterminism(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = SyntheticDataset(self.root / "dataset")
        # Created out of sorted order to check output ordering.
        self.ds.add_recording("102_2b3_Tc_mc_AKGC417L", VALID_B, seconds=3.0)
        self.ds.add_recording("101_1b1_Al_sc_Meditron", VALID_A, seconds=5.0)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, output):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = bm.main(["--dataset-dir", str(self.ds.root), "--output", str(output)])
        return code, out.getvalue(), err.getvalue()

    def test_writes_csv_deterministically(self):
        out1, out2 = self.root / "a" / "metadata.csv", self.root / "b" / "metadata.csv"
        code1, stdout, _ = self.run_main(out1)
        code2, _, _ = self.run_main(out2)
        self.assertEqual((code1, code2), (0, 0))
        self.assertEqual(out1.read_bytes(), out2.read_bytes())
        self.assertIn("Cycles: 4", stdout)

        with open(out1, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), bm.COLUMNS)
        self.assertEqual([r["recording_filename"] for r in rows],
                         ["101_1b1_Al_sc_Meditron.wav"] * 3 + ["102_2b3_Tc_mc_AKGC417L.wav"])

    def test_failure_writes_nothing(self):
        self.ds.add_recording("101_1b2_Pr_sc_Meditron", annotation=None)
        output = self.root / "out" / "metadata.csv"
        code, _, stderr = self.run_main(output)
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())
        self.assertIn("No output written", stderr)


@unittest.skipUnless(bm.DEFAULT_DATASET_DIR.exists(), "ICBHI dataset not present")
class TestRealDataset(unittest.TestCase):
    """Smoke test against the real raw dataset, if available locally."""

    def test_real_dataset_validates(self):
        rows = bm.build_metadata(bm.DEFAULT_DATASET_DIR)
        audio_dir = bm.DEFAULT_DATASET_DIR / bm.AUDIO_SUBDIR
        wavs = {p.name for p in audio_dir.glob("*.wav")}
        self.assertEqual({r["recording_filename"] for r in rows}, wavs)
        self.assertTrue(all(r["cycle_duration_s"] > 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
