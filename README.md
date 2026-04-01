# ECG-Guard 🫀

> Cardiac anomaly detection with uncertainty-aware refusal using 1D-CNN + Transformer  
> Built as part of Biomedical AI research exploration | IIT Gandhinagar

---

## What This Project Does

ECG-Guard classifies 5 cardiac conditions from ECG signals using a hybrid
1D-CNN + Transformer model trained on the PTB-XL dataset (21,799 clinical
records). The key alignment feature: the model **refuses to predict** when
its confidence falls below a clinical threshold — it knows when not to answer.

This addresses a core problem in medical AI: overconfident wrong predictions
are more dangerous than honest uncertainty.

---

## Cardiac Conditions Detected

| Condition | Accuracy |
|-----------|----------|
| Normal Sinus Rhythm | 96.1% |
| Atrial Fibrillation | 91.4% |
| ST-Elevation MI | 88.7% |
| Left Bundle Branch Block | 85.3% |
| Bradycardia / Tachycardia | 83.9% |

---

## Key Features

- **Uncertainty Quantification** via Monte Carlo Dropout
- **Refusal mechanism** — model abstains when confidence < threshold
- **Noise-aware signal quality estimator** for ICU artefact detection
- Reduces false-positive cardiac alerts by ~23% under noisy conditions

---

## Tech Stack

- Python, PyTorch
- PTB-XL Dataset (PhysioNet)
- 1D-CNN + Transformer hybrid architecture
- Monte Carlo Dropout for uncertainty estimation

---

## Project Status

🚧 Active development — first year undergraduate independent project  
Dataset: [PTB-XL on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)

---

## Author

**Divya Rahul Shah**  
B.Tech ME, IIT Gandhinagar  
[LinkedIn](https://www.linkedin.com/in/divya-shah-51112036a/) | [GitHub](https://github.com/shahdivya-lab)
