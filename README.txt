# ML Preprocessing Tool — Flask + scikit-learn

WEKA-style ML tool built with Flask, pandas, NumPy, and scikit-learn.
Supports large datasets (chunked CSV reading for files > 50 MB).

---

## HOW TO RUN (step-by-step)

### Step 1 — Install Python
Download Python 3.9 or newer from https://python.org
Make sure "Add Python to PATH" is ticked during installation (Windows).

### Step 2 — Open a terminal / command prompt
- **Windows**: press Win+R, type `cmd`, press Enter
- **Mac/Linux**: open Terminal

### Step 3 — Navigate to this folder
```
cd path/to/ml_flask
```
Example (Windows):  `cd C:\Users\You\Downloads\ml_flask`
Example (Mac):      `cd ~/Downloads/ml_flask`

### Step 4 — Create a virtual environment
```
python -m venv venv
```

### Step 5 — Activate the virtual environment
Windows:
```
venv\Scripts\activate
```
Mac / Linux:
```
source venv/bin/activate
```
You should see `(venv)` appear at the start of the prompt.

### Step 6 — Install dependencies
```
pip install -r requirements.txt
```
This installs Flask, pandas, NumPy, and scikit-learn.
Internet connection required (one-time only).

### Step 7 — Start the app
```
python app.py
```
You will see:
```
 * Running on http://127.0.0.1:5000
```

### Step 8 — Open in browser
Go to:  http://127.0.0.1:5000

---

## USAGE

1. **Upload** — drag & drop any CSV (any size), or click "Upload CSV"
2. **View Data** — inspect your data, see missing/duplicate counts
3. **Preprocessing** — pick a step from the dropdown, click Apply
   - Remove missing rows
   - Fill missing with mean / median / mode / 0
   - Min-max normalize or z-score standardize
   - Remove duplicates
   - Drop a column
4. **Switch mode** — Supervised or Unsupervised tab
5. **Algorithm** — pick from the dropdown:
   - Supervised classification: KNN, Decision Tree, Naive Bayes,
     Logistic Regression, Random Forest
   - Supervised regression: Linear, Ridge, Random Forest Regressor
   - Unsupervised: K-Means, DBSCAN, PCA
6. **Set parameters** — k value, number of CV folds
7. **Run** — click the Run button, switch to Results tab

---

## PROJECT FILES

```
ml_flask/
├── app.py               ← Flask backend (all API routes + ML logic)
├── requirements.txt     ← Python dependencies
├── templates/
│   └── index.html       ← Single-page HTML interface
├── static/
│   ├── css/style.css    ← Stylesheet
│   └── js/app.js        ← Frontend JavaScript
└── uploads/             ← Uploaded files stored here (auto-created)
```

---

## STOPPING THE APP

Press Ctrl+C in the terminal window.

## RESTARTING

Just run `python app.py` again (no need to reinstall).
