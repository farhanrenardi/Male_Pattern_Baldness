---
title: Male Pattern Baldness Demo
emoji: 🧑🏻‍🦲
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---
# Male Pattern Baldness

This small Streamlit application allows you to upload an image of a scalp and get a prediction of the baldness level (1–7) along with the confidence score.

## Quick Start

### macOS / Linux

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

```bat
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

```PowerShell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Important Notes

Default weights path: EXP-KF-004_artifacts/best.weights.h5
Place your .h5 file in that folder, or use the "Use custom weights file" option in the app to upload custom weights.
The application resizes input images to 300×300 and applies EfficientNet-style preprocessing.
If the model architecture or backbone differs from the one used during training, loading the weights may fail or produce inaccurate predictions.
Whenever possible, use weights that exactly match those used during training.
