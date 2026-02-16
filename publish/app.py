"""
Flask web app for HR promotion prediction (from hrdata.ipynb).
Run train_model.py first (with WA_Fn-UseC_-HR-Employee-Attrition.csv), then: flask run
"""
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "pipeline.pkl")
COLS_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

pipeline = None
feature_columns = None


def load_model():
    global pipeline, feature_columns
    if os.path.isfile(PIPELINE_PATH) and os.path.isfile(COLS_PATH):
        pipeline = joblib.load(PIPELINE_PATH)
        feature_columns = joblib.load(COLS_PATH)
        return True
    return False


def form_to_row():
    """Build one row DataFrame from request form (same columns as training)."""
    return {
        "Age": int(request.form.get("Age", 35)),
        "Attrition": request.form.get("Attrition", "No"),
        "BusinessTravel": request.form.get("BusinessTravel", "Travel_Rarely"),
        "DailyRate": float(request.form.get("DailyRate", 800)),
        "Department": request.form.get("Department", "Research & Development"),
        "DistanceFromHome": int(request.form.get("DistanceFromHome", 9)),
        "Education": int(request.form.get("Education", 3)),
        "EducationField": request.form.get("EducationField", "Life Sciences"),
        "EmployeeCount": 1,
        "EnvironmentSatisfaction": int(request.form.get("EnvironmentSatisfaction", 3)),
        "Gender": request.form.get("Gender", "Male"),
        "HourlyRate": float(request.form.get("HourlyRate", 65)),
        "JobInvolvement": int(request.form.get("JobInvolvement", 3)),
        "JobLevel": int(request.form.get("JobLevel", 2)),
        "JobRole": request.form.get("JobRole", "Research Scientist"),
        "JobSatisfaction": int(request.form.get("JobSatisfaction", 3)),
        "MaritalStatus": request.form.get("MaritalStatus", "Married"),
        "MonthlyIncome": float(request.form.get("MonthlyIncome", 5000)),
        "MonthlyRate": float(request.form.get("MonthlyRate", 20000)),
        "NumCompaniesWorked": int(request.form.get("NumCompaniesWorked", 2)),
        "Over18": "Y",
        "OverTime": request.form.get("OverTime", "No"),
        "PercentSalaryHike": int(request.form.get("PercentSalaryHike", 14)),
        "PerformanceRating": int(request.form.get("PerformanceRating", 3)),
        "RelationshipSatisfaction": int(request.form.get("RelationshipSatisfaction", 3)),
        "StandardHours": 80,
        "StockOptionLevel": int(request.form.get("StockOptionLevel", 1)),
        "TotalWorkingYears": int(request.form.get("TotalWorkingYears", 11)),
        "TrainingTimesLastYear": int(request.form.get("TrainingTimesLastYear", 3)),
        "WorkLifeBalance": int(request.form.get("WorkLifeBalance", 3)),
        "YearsAtCompany": int(request.form.get("YearsAtCompany", 7)),
        "YearsInCurrentRole": int(request.form.get("YearsInCurrentRole", 4)),
        "YearsWithCurrManager": int(request.form.get("YearsWithCurrManager", 4)),
    }


@app.route("/")
def index():
    if not load_model():
        return render_template("index.html", model_loaded=False, stats=None)
    stats = None
    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["Promoted"] = (df["YearsSinceLastPromotion"] == 0).astype(int)
        total = len(df)
        promoted = int(df["Promoted"].sum())
        stats = {
            "total_employees": int(total),
            "promoted_count": int(promoted),
            "promotion_rate_pct": round(100 * promoted / total, 2),
            "by_department": df.groupby("Department")["Promoted"].agg(["count", "sum"]).to_dict("index"),
        }
        stats["dept_labels"] = list(stats["by_department"].keys())
        stats["dept_counts"] = [int(stats["by_department"][k]["count"]) for k in stats["dept_labels"]]
        stats["dept_promoted"] = [int(stats["by_department"][k]["sum"]) for k in stats["dept_labels"]]
    return render_template("index.html", model_loaded=True, stats=stats)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not load_model():
        return render_template("predict.html", model_loaded=False, result=None)
    if request.method == "GET":
        return render_template("predict.html", model_loaded=True, result=None)
    try:
        row = form_to_row()
        row_df = pd.DataFrame([row])[feature_columns]
        pred = pipeline.predict(row_df)[0]
        proba = pipeline.predict_proba(row_df)[0]
        promo_prob = float(proba[1]) if len(proba) > 1 else 0.0
        label = "Likely promoted" if pred == 1 else "Not promoted"
        return render_template(
            "predict.html",
            model_loaded=True,
            result={
                "label": label,
                "promotion_probability": round(promo_prob * 100, 2),
                "raw": int(pred),
            },
        )
    except Exception as e:
        return render_template(
            "predict.html", model_loaded=True, result=None, error=str(e)
        )


def _default_row():
    return {
        "Age": 35, "Attrition": "No", "BusinessTravel": "Travel_Rarely", "DailyRate": 800.0,
        "Department": "Research & Development", "DistanceFromHome": 9, "Education": 3,
        "EducationField": "Life Sciences", "EmployeeCount": 1, "EnvironmentSatisfaction": 3,
        "Gender": "Male", "HourlyRate": 65.0, "JobInvolvement": 3, "JobLevel": 2,
        "JobRole": "Research Scientist", "JobSatisfaction": 3, "MaritalStatus": "Married",
        "MonthlyIncome": 5000.0, "MonthlyRate": 20000.0, "NumCompaniesWorked": 2, "Over18": "Y",
        "OverTime": "No", "PercentSalaryHike": 14, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StandardHours": 80, "StockOptionLevel": 1,
        "TotalWorkingYears": 11, "TrainingTimesLastYear": 3, "WorkLifeBalance": 3,
        "YearsAtCompany": 7, "YearsInCurrentRole": 4, "YearsWithCurrManager": 4,
    }


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not load_model():
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
    try:
        data = request.get_json() or request.form
        defaults = _default_row()
        row = {k: data.get(k, defaults[k]) for k in feature_columns}
        for k in ["Age", "DistanceFromHome", "Education", "EmployeeCount", "EnvironmentSatisfaction",
                  "JobInvolvement", "JobLevel", "JobSatisfaction", "NumCompaniesWorked", "PercentSalaryHike",
                  "PerformanceRating", "RelationshipSatisfaction", "StandardHours", "StockOptionLevel",
                  "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
                  "YearsInCurrentRole", "YearsWithCurrManager"]:
            if k in row:
                row[k] = int(float(row[k]))
        for k in ["DailyRate", "HourlyRate", "MonthlyIncome", "MonthlyRate"]:
            if k in row:
                row[k] = float(row[k])
        row["Over18"] = "Y"
        row_df = pd.DataFrame([row])[feature_columns]
        pred = pipeline.predict(row_df)[0]
        proba = pipeline.predict_proba(row_df)[0]
        return jsonify({
            "prediction": "promoted" if pred == 1 else "not_promoted",
            "promotion_probability": round(float(proba[1]) * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5001)
