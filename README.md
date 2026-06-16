<<<<<<< HEAD
# 🏠 House Price Prediction — End-to-End ML Project

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-2ecc71?style=flat"/>
</p>

> An intermediate-level Machine Learning project predicting California house prices using EDA, feature engineering, multi-model comparison, and XGBoost hyperparameter tuning.

---

## 📌 Problem Statement

Predict the **median house value** for California districts using socioeconomic and geographic features from the California Housing Dataset (20,640 samples, 8 features).

---

## 🗂️ Project Structure

```
house-price-prediction/
│
├── notebooks/
│   └── house_price_prediction.ipynb   ← Full walkthrough (EDA → Model → Results)
│
├── src/
│   ├── preprocess.py                  ← Data loading, cleaning & feature engineering
│   ├── train.py                       ← Model training, tuning & saving
│   └── predict.py                     ← Inference on new data
│
├── models/
│   ├── xgboost_best_model.pkl         ← Saved best model
│   └── scaler.pkl                     ← Fitted StandardScaler
│
├── reports/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_vs_target.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔍 What's Inside

### 1. Exploratory Data Analysis (EDA)
- Target variable distribution (raw + log-transformed)
- Correlation heatmap across all features
- Scatter plots: each feature vs house price

### 2. Feature Engineering
| New Feature | Formula |
|---|---|
| `RoomsPerHousehold` | AveRooms / AveOccup |
| `BedroomsPerRoom` | AveBedrms / AveRooms |
| `PopulationPerHouse` | Population / HouseAge |
| `IncomePerRoom` | MedInc / AveRooms |

### 3. Models Trained & Compared
| Model | R² Score |
|---|---|
| Linear Regression | ~0.60 |
| Ridge Regression | ~0.61 |
| Lasso Regression | ~0.60 |
| Random Forest | ~0.81 |
| Gradient Boosting | ~0.83 |
| **XGBoost (Tuned)** | **~0.85+** |

### 4. Hyperparameter Tuning
GridSearchCV on XGBoost across:
- `n_estimators`: [100, 200]
- `max_depth`: [4, 6]
- `learning_rate`: [0.05, 0.1]
- `subsample`: [0.8, 1.0]

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Best Model | XGBoost (Tuned) |
| R² Score | ~0.85 |
| RMSE | ~0.44 ($44k) |
| Top Feature | `MedInc` (Median Income) |

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/Akash-Agale/house-price-prediction.git
cd house-price-prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook notebooks/house_price_prediction.ipynb

# 5. OR train from terminal
cd src
python train.py

# 6. Run inference
python predict.py
```

---

## 🧪 Sample Prediction

```python
from src.predict import predict

sample = {
    "MedInc"     : 5.0,
    "HouseAge"   : 20.0,
    "AveRooms"   : 6.0,
    "AveBedrms"  : 1.2,
    "Population" : 1200.0,
    "AveOccup"   : 3.0,
    "Latitude"   : 34.05,
    "Longitude"  : -118.25,
}

price = predict(sample)
print(f"Predicted Price: ${price * 100_000:,.0f}")
# Output → Predicted Price: $248,000
```

---

## 📈 Visualizations

| Plot | Description |
|---|---|
| `target_distribution.png` | Raw vs log-transformed house price distribution |
| `correlation_heatmap.png` | Feature correlation matrix |
| `model_comparison.png` | R² and RMSE across all models |
| `actual_vs_predicted.png` | Predicted vs actual scatter + residuals |
| `feature_importance.png` | XGBoost feature importance ranking |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 🔮 Future Improvements

- [ ] Deploy with **FastAPI** + Docker container
- [ ] Add **LightGBM** and **CatBoost** to comparison
- [ ] Try **Stacking Ensemble** (XGBoost + RF + GBM)
- [ ] Build a **Streamlit** web app for live predictions
- [ ] Add **SHAP** explainability plots

---

## 👨‍💻 Author

**Akash Agale**
- 🔗 [LinkedIn](https://linkedin.com/in/akash-agale-b61202371)
- 🐙 [GitHub](https://github.com/Akash-Agale)
- 📍 Pune, India

---

## 📄 License

This project is licensed under the MIT License — feel free to use and modify it.

---

<p align="center">⭐ If you found this helpful, please star the repo!</p>
=======
# house-price-prediction
Machine Learning project for predicting house prices using regression models.
>>>>>>> 7c2895a0b8d16352ce37f1d4ef13cdbb950f0750
