# 🚀 Startup Success Prediction

A machine learning web app that predicts whether a startup is likely to be **Acquired (Success)** or **Closed (Fail)** based on startup funding, investor backing, milestone activity, and company-level traits.

This project is built with **Python, Streamlit, XGBoost, Scikit-learn, SHAP, Pandas, NumPy, and Matplotlib**.

## ✨ Features

- Single-startup prediction
- Batch prediction via CSV upload
- What-if scenario simulator
- Startup comparison dashboard
- SHAP-based local explanations
- Recommendation engine for startup improvements
- Investor-style report export
- Portfolio dashboard with filters and summary charts
- Automatic handling of missing values using medians
- Model training and evaluation metrics
- Persistence of trained model and stats as `.pkl` files

## 🧠 Model and Approach

The app trains an **XGBoost Classifier** with regularization settings designed to reduce overfitting, including:

- `max_depth = 3`
- `learning_rate = 0.03`
- `n_estimators = 200`
- `subsample = 0.6`
- `colsample_bytree = 0.6`
- `reg_alpha = 2.0`
- `reg_lambda = 5.0`
- `min_child_weight = 8`
- `scale_pos_weight` based on class imbalance

The model is evaluated using:

- Accuracy
- Confusion matrix
- ROC-AUC
- 5-fold cross-validation
- Learning curve
- Train vs. test accuracy comparison

## 📊 Important Features

The model uses the following startup features when present in the dataset:

- `has_VC`
- `has_angel`
- `has_roundA`
- `has_roundB`
- `has_roundC`
- `has_roundD`
- `funding_rounds`
- `funding_total_usd`
- `avg_participants`
- `is_top500`
- `milestones`

Target:

- `acquired` → Success (`1`)
- `closed` → Failure (`0`)

## 🚀 Run the App

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
Mini_Project/
├── app.py
├── startup data.csv
├── requirements.txt
├── README.md
├── startup_model.pkl
├── startup_stats.pkl
├── last_input_example.csv
├── encoders.pkl
└── .gitignore
```

## 🔍 How to Use

1. Place your dataset as `startup data.csv` in the project folder.
2. Run the Streamlit app.
3. Use the app to:
   - predict a single startup
   - upload a batch of startups
   - test a what-if scenario
   - compare startups
   - read SHAP explanations
   - review recommendations
   - export an investor report
   - review portfolio-level trends

## 💡 Extra Features Added

This version includes:

- CSV upload for batch prediction
- Scenario simulation for changing startup inputs
- Startup comparison dashboard
- Recommendation engine
- Investor report in plain text format
- Portfolio dashboard with filters and charts
- Batch recommendation summaries

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Pickle

## ⚠️ Notes

- The dataset should contain a `status` column with values like `acquired` and `closed`.
- Only matching feature columns are used automatically.
- Missing numeric values are filled using medians from the dataset.
- Generated model files can be ignored in Git if desired.

## 📄 License

This project is intended for educational and portfolio use.

## 👨‍💻 Author

Add your name and profile links here if you want to personalize the project
