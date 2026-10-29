# Startup Success Prediction

[![Python](https://img.shields.io/badge/Python-ML-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-189FDD)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-7A3E9D)](https://shap.readthedocs.io/)

A Streamlit application that estimates whether a startup is likely to be acquired or closed using its funding history, investor backing, milestones, and other company-level indicators. The project combines a regularized XGBoost classifier with interactive predictions, model diagnostics, and explanations.

> This project is intended for educational and portfolio use. Predictions are estimates based on historical data, not investment advice or a guarantee of future outcomes.

---

## 1. Overview

Startup Success Prediction provides an interactive workflow for exploring startup outcomes. The app trains a classification model from the included dataset and lets users evaluate individual startup profiles, compare scenarios, and analyze batches of startups.

The current target labels are:

- `acquired` — success (`1`)
- `closed` — failure (`0`)

---

## 2. Features

### Individual startup analysis

- Predict an acquisition probability and outcome from startup attributes
- Generate rule-based recommendations using dataset medians
- Download a plain-text investor report
- View global model feature importance
- Explore a local SHAP explanation and a plain-English summary

### Batch and portfolio analysis

- Upload a CSV of startup features and download predictions
- Review the highest-probability predictions and probability charts
- Download a batch summary report with recommendations
- Filter results by success probability and funding amount
- Review predicted outcome counts and the average probability for the filtered portfolio

### Scenario and model diagnostics

- Change startup attributes in a what-if simulator and compare the resulting prediction
- Compare two startup profiles side by side
- Review test-set accuracy diagnostics, a confusion matrix, ROC-AUC, and a learning curve
- Inspect five-fold cross-validation results during model preparation

---

## 3. Screenshots

Screenshots are not included in the repository yet. Add genuine captures of the running app under `docs/screenshots/`, then embed them here. Useful views to capture:

- **Prediction dashboard** — startup inputs and prediction result
- **Batch analysis** — uploaded results, charts, and portfolio filters
- **Explainability** — feature importance and SHAP explanation

For example, after adding `docs/screenshots/prediction-dashboard.png`, embed it with:

```markdown
![Startup prediction dashboard](docs/screenshots/prediction-dashboard.png)
```

---

## 4. Model and Evaluation

The app trains an `XGBClassifier` with regularization settings intended to limit model complexity:

| Parameter | Value |
|---|---:|
| Estimators | 200 |
| Learning rate | 0.03 |
| Maximum tree depth | 3 |
| Subsample | 0.6 |
| Column subsample | 0.6 |
| L1 regularization (`reg_alpha`) | 2.0 |
| L2 regularization (`reg_lambda`) | 5.0 |
| Minimum child weight | 8 |
| Positive-class weight | Calculated from the training split |

The dataset is split into training and test sets using an 80/20 stratified split. The app reports or visualizes:

- Test accuracy and train-versus-test accuracy
- Five-fold stratified cross-validation accuracy
- Confusion matrix
- ROC-AUC and ROC curve
- Learning curve
- Feature importance
- SHAP-based local feature contributions

Metrics are specific to the included data and evaluation setup; they should not be interpreted as guarantees of performance on new or future startups.

---

## 5. Input Data

The app reads `startup data.csv` from the project directory. The dataset should include a `status` target column with `acquired` and `closed` values. Rows with other or missing target values are excluded from training.

The model uses these feature columns when present:

| Feature | Description |
|---|---|
| `has_VC` | Venture-capital backing indicator |
| `has_angel` | Angel-investor backing indicator |
| `has_roundA` | Series A funding indicator |
| `has_roundB` | Series B funding indicator |
| `has_roundC` | Series C funding indicator |
| `has_roundD` | Series D funding indicator |
| `funding_rounds` | Number of funding rounds |
| `funding_total_usd` | Total funding in USD |
| `avg_participants` | Average number of participants |
| `is_top500` | Top-500 indicator |
| `milestones` | Number of milestones |

Numeric values that are missing in the training data are filled with column medians. Binary indicators are normalized to numeric values. For batch uploads, include the feature columns above; missing feature columns are filled with `0`.

---

## 6. Project Structure

```text
Startup-Success-Prediction/
├── test_2.py                # Streamlit application entry point
├── startup data.csv         # Training dataset
├── last_input_example.csv   # Example/saved individual input
├── requirements.txt         # Python dependencies
├── startup_model.pkl        # Saved trained model
├── startup_stats.pkl        # Saved model statistics
├── encoders.pkl             # Serialized project artifact
├── .gitignore
└── README.md
```

The application currently lives in `test_2.py`, so use that filename when launching Streamlit.

---

## 7. Technology Stack

- **Language:** Python
- **App framework:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine learning:** XGBoost, Scikit-learn
- **Model explainability:** SHAP
- **Visualization:** Matplotlib
- **Model persistence:** Pickle

---

## 8. Getting Started

### 8.1 Clone the repository

```bash
git clone https://github.com/Varadraj2992/Startup-Success-Prediction.git
cd Startup-Success-Prediction
```

### 8.2 Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 8.3 Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Run the Application

From the repository root, start the Streamlit app:

```bash
streamlit run test_2.py
```

Open the local URL printed by Streamlit, typically:

```text
http://localhost:8501
```

The included `startup data.csv` must remain in the project directory unless the dataset path in the application is changed.

---

## 10. Using the App

1. Start the app using the command above.
2. Enter startup attributes for an individual prediction, or use the what-if simulator to explore a scenario.
3. Review the predicted outcome, probability, recommendations, and available explanations.
4. Upload a CSV containing startup feature columns for batch predictions.
5. Use the portfolio filters to explore the batch results and download prediction or report files.
6. Compare two startup profiles using the comparison dashboard.

---

## 11. Generated Files

During use, the application writes the trained model and model statistics to `startup_model.pkl` and `startup_stats.pkl`. Individual predictions can also update `last_input_example.csv`. These are local serialized or data artifacts; only load pickle files from sources you trust.

---

## 12. Limitations

- The model is trained on the supplied historical dataset and may not generalize to other populations or time periods.
- Predictions reflect associations in the data; they do not establish that a funding or company attribute causes an acquisition.
- The application expects the `status` target and the feature names documented above.
- The recommendation engine uses simple median-based rules and should be treated as guidance for exploration, not professional business advice.
- This project is a research/demo application, not a production decision system.

---

## 13. Author

**Varadraj Bhandare**

M.Sc. Data Science & Big Data Analytics

Areas of interest:

- Data Analytics
- Data Science
- Machine Learning
- Quantitative Finance
- Business Intelligence
- Big Data Analytics

Visit [Varadraj2992 on GitHub](https://github.com/Varadraj2992) for project updates and contact options.

---

## 14. License

This project is intended for educational, research, and portfolio purposes. No `LICENSE` file is currently included, so these intended uses do not grant permission to reuse, modify, or redistribute the code. Add a license file to define those terms.

---

## 15. Project Goals

Startup Success Prediction aims to provide an accessible, interactive way to explore how startup funding, investor backing, and milestone activity relate to acquisition or closure outcomes in the supplied dataset.

The project brings together a trained classification model, prediction and comparison tools, batch analysis, model diagnostics, and SHAP explanations. It is designed as a learning and portfolio project, not as a production investment decision system.

Future development may include a more modular application structure, expanded validation on external data, additional model comparisons, and improved documentation and automated tests.

---

## 16. Development Practices

The project applies and demonstrates the following practices:

- Keeps the Streamlit interface and model workflow runnable from a documented entry point.
- Cleans feature data and handles missing numeric values using training-data medians.
- Uses a stratified train/test split and cross-validation to evaluate model performance.
- Provides evaluation outputs such as accuracy, confusion matrix, ROC-AUC, and a learning curve.
- Uses SHAP to help inspect individual prediction explanations.
- Tracks project files and changes in Git.

Automated CI, a dedicated test suite, and formal linting are not currently documented as part of this repository.

---

## 17. Repository

- **GitHub repository:** [Varadraj2992/Startup-Success-Prediction](https://github.com/Varadraj2992/Startup-Success-Prediction)
- **Application entry point:** `test_2.py`
- **Training dataset:** `startup data.csv`
- **Dependency list:** `requirements.txt`

Clone the repository:

```bash
git clone https://github.com/Varadraj2992/Startup-Success-Prediction.git
cd Startup-Success-Prediction
```
