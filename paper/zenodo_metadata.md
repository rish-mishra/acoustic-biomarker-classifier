# Zenodo metadata (draft, not yet uploaded)

## Title

**Beyond Validation Scores: Patient-Independent Generalization, Robustness and Uncertainty in Respiratory-Sound Classification**

Title options considered:
1. *Beyond Validation Scores: Patient-Independent Generalization, Robustness and Uncertainty in Respiratory-Sound Classification*. This is the chosen title.
2. *Evaluating Generalization, Robustness and Uncertainty in Respiratory-Sound Classification Under a Frozen Patient-Independent Protocol*.
3. *Unseen Patients, Degraded Audio and the Limits of Confidence: A Strict Evaluation of a Respiratory-Sound Classifier*.

## Author

- **Name:** Mishra, Rishabh
- **Affiliation:** Independent Researcher, Vaughan, Ontario, Canada
- **ORCID:** [optional, recommended]

## Upload details

| Field | Value |
|---|---|
| Upload type | Publication |
| Publication type | Preprint |
| Version | 1.0 |
| Language | English |
| Publication date | [date of upload] |
| DOI | [Zenodo DOI to be added after publication] (Zenodo reserves one on upload) |
| Peer review | Not peer reviewed (independent research manuscript) |

## Description

This preprint asks how well a respiratory-sound classifier generalizes to completely unseen patients and to degraded audio, and whether its own uncertainty can flag predictions that should not be trusted.

Using the ICBHI 2017 Respiratory Sound Database (126 patients, 6,898 annotated respiratory cycles), respiratory cycles were classified as normal, crackle, wheeze or both under a fixed patient-level split (88/19/19 patients). A small CNN, a convolutional-recurrent network and an ImageNet-pretrained MobileNetV3 were compared on validation data only. A predeclared three-seed study found no architecture convincingly better than the others, so the simplest and most stable model was kept. All model, calibration, abstention and degradation settings were then frozen and hash-verified before the test patients were evaluated once.

Macro F1 fell from 0.4271 on validation to 0.3175 on the untouched test patients (patient-level bootstrap 95% CI 0.218–0.413), and the mixed "both" class essentially failed. Several degradation failure patterns reproduced from validation to test, including the collapse of predictions onto crackle under heavy noise. Temperature scaling reduced the average overconfidence gap on test by about 40% (from +0.208 to +0.127), but uncertainty remained a weak detector of individual errors (AUROC 0.542–0.554), and abstention helped only modestly.

The results suggest that clean validation performance alone substantially overstates how such classifiers behave on new patients and recording conditions.

This is an independent research preprint and has not been peer reviewed. The model is not a medical device or diagnostic system and has not been clinically validated.

## Notes / disclosure

Generative AI tools assisted with code, documentation, language editing, structuring and literature searches. The research design, evaluation protocol, interpretation of the locked results and final responsibility remain with the author. All numerical results come from the project's stored experimental artifacts, and all references were checked against their primary sources.

## Keywords

respiratory sounds; lung auscultation; ICBHI 2017; crackles; wheezes; patient-independent evaluation; generalization; domain shift; robustness; calibration; temperature scaling; uncertainty estimation; selective prediction; machine learning; audio classification

## Related identifiers

| Relation | Identifier | Type |
|---|---|---|
| Is supplemented by | https://github.com/rish-mishra/acoustic-biomarker-classifier | Software (code, frozen protocol, locked results) |
| Is supplemented by | https://github.com/rish-mishra/acoustic-biomarker-classifier/releases/tag/v1.0.1 | Final software release described by the manuscript (documentation and test maintenance; no result changed) |
| Is supplemented by | https://github.com/rish-mishra/acoustic-biomarker-classifier/releases/tag/v1.0.0 | Release carrying the reference checkpoint |

The trained checkpoint is attached **only** to release v1.0.0, as `baseline_cnn_v3_seed42_best.pt` (1,188,469 bytes, SHA-256 `2c2c7c03ed00b6562c63ff88be12d43bd8f9ff890074f9dac4d4ce5874e3b567`). It was not re-uploaded with v1.0.1 and is unchanged.

**Note:** make sure the GitHub repository is public before publishing. If it isn't, readers of the Zenodo record won't be able to reach these links.

## Licences

- **Manuscript:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Repository code:** MIT License, in the repository's `LICENSE` file, copyright (c) 2026 Rishabh Mishra. It covers only the project's own source code. It does **not** apply to:
  - the ICBHI 2017 raw dataset, which is not redistributed and remains under its own terms of use;
  - any third-party material, such as the pretrained ImageNet weights, which keep their own licences;
  - the research manuscript, which is CC BY 4.0.

## Files to upload

- The manuscript, exported to PDF from `paper/acoustic_biomarker_classifier_manuscript.md` with figures embedded. The Markdown references SVG figures by relative path:
  - `reports/final_report/figures/seed_comparison.svg`
  - `reports/final_report/figures/test_confusion_matrix.svg`
  - `reports/final_report/figures/robustness_validation_vs_test.svg`
  - `reports/final_evaluation/clean_test/reliability_diagram.svg`
  - `reports/final_report/figures/risk_coverage_validation_vs_test.svg`
- Optionally, the Markdown source.

## Before uploading

- Make the repository public, or remove the links above.
- Export to PDF and check that the SVG figures render on a light background. They contain dark-mode styling for screens.
