# Promotion Predictor — Publish bundle

HR promotion prediction app (from hrdata.ipynb). Push this folder to Git to deploy.

## Run locally

```bash
cd HR/publish
pip install -r requirements.txt
flask run
```

App runs on http://127.0.0.1:5001 (or 5000 if no other app). Dashboard shows dataset stats if `WA_Fn-UseC_-HR-Employee-Attrition.csv` is in this folder. **Predict promotion** works with `pipeline.pkl` and `feature_columns.pkl` only.

## Retrain

Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` here, then:

```bash
python train_model.py
```

## Deploy

Push this folder. On server: `pip install -r requirements.txt`, then run with Gunicorn or your host (e.g. `gunicorn -w 4 -b 0.0.0.0:5001 app:app`).
