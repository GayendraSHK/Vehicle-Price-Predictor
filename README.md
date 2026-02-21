# 🚗 Sri Lankan Vehicle Price Predictor

An end-to-end machine learning pipeline that **scrapes** used vehicle listings from [Riyasewana.com](https://riyasewana.com), **preprocesses** the data, trains an **XGBoost** regression model, and serves predictions through an interactive **Streamlit** web app with SHAP explainability.

---

## 📁 Project Structure

```
├── riyasewana_detailed_scraper.py   # Web scraper for Riyasewana.com listings
├── preprocess_data.py               # Data cleaning & preprocessing
├── prepare_data.py                  # Feature encoding & train/test split
├── xgboost_vehicle_model.py         # XGBoost model training & evaluation
├── streamlit_app.py                 # Streamlit web app for predictions
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## ⚙️ Prerequisites

- **Python 3.9+** — [Download here](https://www.python.org/downloads/)
- **pip** — comes bundled with Python
- **Git** — [Download here](https://git-scm.com/downloads)

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | Label encoding, train/test split, metrics |
| `xgboost` | Gradient boosting model |
| `shap` | Model explainability |
| `matplotlib` | Plotting & visualisation |
| `streamlit` | Interactive web app |
| `requests` | HTTP requests (scraper) |
| `curl-cffi` | HTTP requests with browser impersonation (scraper) |
| `beautifulsoup4` | HTML parsing (scraper) |
| `lxml` | Fast XML/HTML parser |

---

## 🚀 How to Run

The pipeline runs in **4 sequential steps**. Each step produces output files that the next step consumes.

### Step 1 — Scrape Vehicle Data

```bash
python riyasewana_detailed_scraper.py
```

- Scrapes vehicle listings (make, model, year, price, mileage, etc.) from Riyasewana.com
- Saves raw data to a timestamped CSV file (e.g. `riyasewana_search_YYYYMMDD_HHMMSS.csv`)

### Step 2 — Preprocess the Data

```bash
python preprocess_data.py
```

- Cleans the raw CSV (removes duplicates, handles missing values, fixes outliers)
- Standardises categorical fields
- Outputs a `*_preprocessed.csv` file

### Step 3 — Encode Features & Split Data

```bash
python prepare_data.py
```

- Label-encodes categorical features (make, model, gear, fuel type, options, location)
- Splits data into 80/20 train/test sets
- Saves `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, and `label_encoders.pkl`

### Step 4 — Train the XGBoost Model

```bash
python xgboost_vehicle_model.py
```

- Trains a baseline XGBoost model
- Performs hyperparameter tuning with GridSearchCV
- Runs cross-validation
- Generates SHAP explainability plots and feature importance charts
- Saves the trained model (`xgboost_vehicle_model.pkl`) and evaluation metrics

### Step 5 — Launch the Web App

```bash
python -m streamlit run streamlit_app.py
```

- Opens an interactive web app at `http://localhost:8501`
- Select vehicle details (make, model, year, mileage, etc.)
- Get an instant price prediction with SHAP-based explanation

---

## 📊 Model Details

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Regressor |
| Tuning | GridSearchCV (3-fold CV) |
| Explainability | SHAP (waterfall, summary, bar, dependence plots) |

---

## 📝 Notes

- The scraper uses `curl_cffi` for browser impersonation to avoid bot detection. Make sure your network allows outbound HTTP requests.
- The Streamlit app requires the trained model file (`xgboost_vehicle_model.pkl`), label encoders (`label_encoders.pkl`), and a preprocessed CSV to be present in the project directory.
- The CSV data file is **not** included in the repository — you must run the scraper first (Step 1).

---

## 📄 License

This project is for educational purposes.
