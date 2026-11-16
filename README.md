# Pneumonia Detection from Chest X-Rays

Binary classification of chest X-rays into **NORMAL** vs **PNEUMONIA**, comparing:

1. A custom CNN trained from scratch
2. Transfer learning with DenseNet121 (two-phase: feature extraction, then fine-tuning)

Dataset: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Run it

Open `pneumonia_detection.ipynb` in Google Colab, or run it locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install tensorflow kagglehub pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook pneumonia_detection.ipynb
```

The notebook's `DOWNLOAD_METHOD` cell controls how the dataset is fetched:

- `"api"` (default): needs a `kaggle.json` (Kaggle → Settings → API → Create New Token)
- `"kagglehub"`: no credentials needed
- `"manual"`: place the downloaded zip in the data folder yourself

Outputs: plots and metrics go to `results/`, trained models to `models/`.

## Note

Educational project. Not a medical device. Predictions must not be used for diagnosis.
