# Reference checkpoint release record

The frozen reference checkpoint is git-ignored and not stored in the repository. It is to be attached to a GitHub Release under the filename below. Any copy can be checked against the SHA-256, which is also recorded in protocol v3 (`1_frozen_model.checkpoint_sha256`).

| Field | Value |
|---|---|
| Release filename | `baseline_cnn_v3_seed42_best.pt` |
| Local path | `models/baseline_cnn_v3/best.pt` |
| SHA-256 | `2c2c7c03ed00b6562c63ff88be12d43bd8f9ff890074f9dac4d4ce5874e3b567` |
| Size | 1188469 bytes |
| Architecture | `baseline_cnn`: BaselineCNN, channels 16-32-64-128, dropout 0.3, 294,740 parameters |
| Run | `baseline_cnn_v3` (learning rate 1e-4, AdamW weight decay 0.01) |
| Seed | 42 |
| Best epoch | 13 (validation macro F1 0.4271) |
| Frozen protocol | version 3 (`reports/final_evaluation/protocol.json`, SHA-256 `05f7fd0048f49d9a0d018319142fc6b07397b46dff97995e1bcda07eeb524f65`) |

To check a downloaded copy:

```bash
sha256sum baseline_cnn_v3_seed42_best.pt   # must print 2c2c7c03ed00b6562c63ff88be12d43bd8f9ff890074f9dac4d4ce5874e3b567
mkdir -p models/baseline_cnn_v3 && cp baseline_cnn_v3_seed42_best.pt models/baseline_cnn_v3/best.pt
python src/evaluation/final_protocol.py --verify
```
