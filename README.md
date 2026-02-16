# 🚀 HR Promotion Prediction System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Vercel Deployment](https://img.shields.io/badge/deploy-vercel-black.svg)](https://vercel.com)

A professional Machine Learning web application that predicts the likelihood of employee promotions. Built with **Flask** and **Scikit-Learn**, this tool provides both an interactive dashboard for HR analytics and a predictive engine.

## 🌟 Key Features

- **📊 HR Analytics Dashboard**: Real-time statistics on employee distribution and promotion rates across departments.
- **🔮 Predictive Engine**: Interactive form to calculate individual promotion probability based on 33 unique employee attributes.
- **🔌 RESTful API**: Endpoint available for integrating predictions into other HR management systems.
- **🎨 Modern UI**: Clean, responsive interface with intuitive form interactions and real-time loading states.

## 📁 Project Structure

```text
├── app.py                  # Flask Application & API
├── train_model.py          # ML Training Pipeline
├── requirements.txt        # Project Dependencies
├── vercel.json             # Vercel Deployment Config
├── LICENSE                 # MIT License
├── WA_Fn-UseC...csv        # Dataset (IBM HR Attrition)
├── pipeline.pkl            # Trained Model (Serialized)
├── feature_columns.pkl     # Feature Metadata
└── templates/              # HTML Templates (Jinja2)
    ├── base.html
    ├── index.html          # Dashboard
    └── predict.html        # Prediction Form
```

## 🛠️ Local Installation

### 1. Clone & Setup Environment
```bash
# Recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare the Model
Before running the app, you must train the model once to generate the serialization files:
```bash
python train_model.py
```

### 3. Launch the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5001`.

## 🌐 Deployment (Vercel)

This project is optimized for **Vercel**.

1.  **Install Vercel CLI**: `npm i -g vercel`
2.  **Deploy**: Run `vercel` in the root directory.
3.  **Note**: Ensure `pipeline.pkl` and `feature_columns.pkl` are committed/uploaded for the prediction engine to function immediately.

## 📡 API Usage

**Endpoint:** `POST /api/predict`  
**Content-Type:** `application/json`

Example Request Body:
```json
{
  "Age": 30,
  "Department": "Sales",
  "JobLevel": 3,
  "MonthlyIncome": 6000
}
```

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---
*Built with ❤️ for HR Professionals*
