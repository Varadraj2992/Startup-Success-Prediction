# 🚀 Startup Success Prediction

A machine learning web application that predicts whether a startup is likely to be **Acquired (Success)** or **Closed (Fail)** using startup funding, investment, milestone, and company-related features.

The application is built with **Python, XGBoost, Scikit-learn, SHAP, Pandas, Matplotlib, and Streamlit**.

## ✨ Features

- Startup success/failure prediction
- XGBoost classification model
- Strong L1/L2 regularization to reduce overfitting
- Class imbalance handling with `scale_pos_weight`
- Train/test accuracy comparison
- 5-fold cross-validation
- Confusion matrix
- ROC curve and ROC-AUC score
- Learning curve
- Global feature importance
- SHAP-based local explanations for individual predictions
- Automatic handling of missing numeric values using medians
- Saves the trained model and statistics as `.pkl` files
- Saves the latest prediction input as `last_input_example.csv`

## 🧠 Machine Learning Model

The project uses an **XGBoost Classifier** with relatively shallow trees and stronger regularization.

Key configuration includes:

- `n_estimators = 200`
- `learning_rate = 0.03`
- `max_depth = 3`
- `subsample = 0.6`
- `colsample_bytree = 0.6`
- `reg_alpha = 2.0`
- `reg_lambda = 5.0`
- `min_child_weight = 8`
- `scale_pos_weight` calculated from the training data

These settings are intended to control model complexity and reduce overfitting.

## 📊 Input Features

The model uses the following features when they are available in the dataset:

| Feature | Description |
|---|---|
| `has_VC` | Whether the startup has VC funding |
| `has_angel` | Whether the startup has angel funding |
| `has_roundA` | Whether the startup received Series A funding |
| `has_roundB` | Whether the startup received Series B funding |
| `has_roundC` | Whether the startup received Series C funding |
| `has_roundD` | Whether the startup received Series D funding |
| `funding_rounds` | Number of funding rounds |
| `funding_total_usd` | Total funding raised in USD |
| `avg_participants` | Average number of investors per funding round |
| `is_top500` | Whether the company is a Top 500 company |
| `milestones` | Number of major milestones |

The target column is:

- `acquired` → Success (`1`)
- `closed` → Failure (`0`)

## 📁 Project Structure

```text
startup-success-predictor/
│
├── test_2.py
├── startup data.csv
├── requirements.txt
├── README.md
│
├── startup_model.pkl          # Generated after running the app
├── startup_stats.pkl          # Generated after running the app
└── last_input_example.csv     # Generated after making a prediction
```

> Keep `startup data.csv` in the same directory as `test_2.py`, or update the `DATA_FILE` variable in the Python file.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application with:

```bash
streamlit run test_2.py
```

The application will open in your browser.

## 🔮 How to Use

1. Place `startup data.csv` in the project directory.
2. Start the Streamlit application.
3. Enter the startup's funding and business information.
4. Click **Predict**.
5. The application displays:
   - Predicted outcome
   - Probability of success
   - Global feature importance
   - Confusion matrix
   - ROC-AUC
   - Learning curve
   - SHAP explanation showing which features influenced the prediction

## 📈 Model Evaluation

The application evaluates the model using:

- Accuracy
- Classification report
- Confusion matrix
- ROC-AUC
- 5-fold stratified cross-validation
- Learning curve
- Train vs. test accuracy gap

This makes the project useful not only as a prediction application but also as a demonstration of model evaluation and explainable AI.

## 🔍 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) is used to explain individual predictions.

The application shows:

- Features pushing the prediction toward **success**
- Features pushing the prediction toward **failure**
- A SHAP waterfall visualization

This helps make the machine learning prediction more interpretable.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Streamlit** — Interactive web application
- **Pandas** — Data processing
- **NumPy** — Numerical operations
- **Scikit-learn** — Model evaluation and cross-validation
- **XGBoost** — Machine learning model
- **SHAP** — Explainable AI
- **Matplotlib** — Data visualization
- **Pickle** — Model/statistics persistence

## ⚠️ Important Notes

- The dataset must contain a `status` column with values such as `acquired` and `closed`.
- Feature names in the CSV should match the feature names expected by the application.
- The application automatically uses only the expected features that are present in the CSV.
- Missing numeric values are filled using the median calculated from the dataset.
- The generated `.pkl` files do not need to be committed to GitHub unless you specifically want to distribute the trained model.
- For a public GitHub repository, consider adding large datasets and generated model files to `.gitignore`.

## 📄 License

This project is intended for educational, portfolio, and demonstration purposes. Add a license such as MIT if you want others to reuse the code under explicit terms.

## 👨‍💻 Author

**Your Name**

GitHub: `https://github.com/YOUR_USERNAME`

LinkedIn: `https://www.linkedin.com/in/YOUR_PROFILE/`
