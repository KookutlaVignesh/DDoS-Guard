import os, uuid, shutil, traceback
from pathlib import Path

from flask import Flask, request, jsonify, render_template, session

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)

# Supervised
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Unsupervised
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

MAX_PREVIEW_ROWS = 500
CHUNK_SIZE       = 50_000   # rows per chunk for files > 50 MB

# In-process session store  {sid: {"path": str, "orig_path": str, "df": DataFrame}}
STORE = {}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_sid() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def load_df(path: str) -> pd.DataFrame:
    """Read CSV with chunked I/O for files > 50 MB."""
    na_vals = ["?", "NA", "N/A", "null", "NULL", "NaN", "nan", "n/a", ""]
    if os.path.getsize(path) > 50 * 1024 * 1024:
        chunks = pd.read_csv(path, chunksize=CHUNK_SIZE,
                             low_memory=False, na_values=na_vals)
        return pd.concat(chunks, ignore_index=True)
    return pd.read_csv(path, low_memory=False, na_values=na_vals)


def safe_val(v):
    """Convert a cell value to a JSON-serialisable Python scalar."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return str(v)


def df_stats(df: pd.DataFrame) -> dict:
    return {
        "rows":    int(len(df)),
        "cols":    int(len(df.columns)),
        "missing": int(df.isnull().sum().sum()),
        "dups":    int(df.duplicated().sum()),
    }


def df_preview(df: pd.DataFrame, target: str = "") -> dict:
    """Return a JSON-safe preview dict."""
    cols_to_show = list(df.columns[:50])
    preview      = df[cols_to_show].head(MAX_PREVIEW_ROWS)
    rows = [
        {c: safe_val(row[c]) for c in cols_to_show}
        for _, row in preview.iterrows()
    ]
    col_info = [
        {
            "name":    c,
            "dtype":   str(df[c].dtype),
            "missing": int(df[c].isnull().sum()),
            "target":  c == target,
        }
        for c in cols_to_show
    ]
    return {
        "columns":      col_info,
        "rows":         rows,
        "total_rows":   int(len(df)),
        "total_cols":   int(len(df.columns)),
        "truncated_cols": len(df.columns) > 50,
    }


def save_df(sid: str, df: pd.DataFrame):
    df.to_csv(STORE[sid]["path"], index=False)
    STORE[sid]["df"] = df


def prepare_X(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Convert a mixed dataframe slice → clean float64 numpy array.
    • Label-encodes any string/categorical column (including pandas StringDtype).
    • Median-imputes any remaining NaNs.
    """
    X = df[feature_cols].copy()

    for c in X.columns:
        col = X[c]
        # Detect string-like dtype regardless of pandas version
        is_str = (col.dtype == object or
                  str(col.dtype).startswith("string") or
                  str(col.dtype).startswith("String"))
        if is_str:
            le     = LabelEncoder()
            filled = col.fillna("__missing__").astype(str)
            X[c]   = le.fit_transform(filled)

    # Force every column to numeric float64 (catches any remaining issues)
    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float64)

    if X.isnull().any().any():
        imp     = SimpleImputer(strategy="median")
        X_arr   = imp.fit_transform(X)
        # imp may drop columns if all-NaN; rebuild with original column names if sizes match
        if X_arr.shape[1] == len(X.columns):
            X = pd.DataFrame(X_arr, columns=X.columns)
        else:
            X = pd.DataFrame(X_arr)

    return X.values.astype(np.float64)


def encode_y(y_raw) -> tuple:
    """
    Encode a target array to integers; returns (y_encoded, label_encoder_or_None).
    Handles object, StringDtype, numeric arrays correctly.
    """
    # Coerce to plain Python object array first
    y = np.array(y_raw, dtype=object)

    # Check if already numeric
    try:
        y_num = y.astype(np.float64)
        return y_num, None          # regression / already encoded
    except (ValueError, TypeError):
        pass

    le    = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc.astype(np.int64), le


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        sid  = get_sid()
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400
        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported"}), 400

        path = str(UPLOAD_FOLDER / f"{sid}_data.csv")
        orig = str(UPLOAD_FOLDER / f"{sid}_orig.csv")
        file.save(path)
        shutil.copy(path, orig)

        df = load_df(path)
        STORE[sid] = {"path": path, "orig_path": orig, "df": df}

        return jsonify({
            "message": f"Loaded {len(df):,} rows × {len(df.columns)} cols",
            "stats":   df_stats(df),
            "columns": list(df.columns),
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    try:
        sid = get_sid()
        if sid not in STORE:
            return jsonify({"error": "No dataset loaded"}), 400
        df = load_df(STORE[sid]["orig_path"])
        save_df(sid, df)
        return jsonify({"message": "Dataset reset to original",
                        "stats": df_stats(df),
                        "columns": list(df.columns)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview")
def preview():
    try:
        sid    = get_sid()
        if sid not in STORE:
            return jsonify({"error": "No dataset loaded"}), 400
        target = request.args.get("target", "")
        df     = STORE[sid]["df"]
        return jsonify({**df_preview(df, target), "stats": df_stats(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preprocess", methods=["POST"])
def preprocess():
    try:
        sid  = get_sid()
        if sid not in STORE:
            return jsonify({"error": "No dataset loaded"}), 400

        body = request.json or {}
        step = body.get("step", "")
        col  = body.get("col", "")
        df   = STORE[sid]["df"].copy()

        # Which numeric columns to target
        num_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols  = [col] if (col and col in df.columns) else num_cols

        if step == "remove_missing":
            before = len(df)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            msg = (f"Removed {before - len(df):,} rows with missing values "
                   f"→ {len(df):,} rows remain")

        elif step == "remove_duplicates":
            before = len(df)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            msg = f"Removed {before - len(df):,} duplicate rows"

        elif step == "drop_col":
            if not col or col not in df.columns:
                return jsonify({"error": "Select a valid column to drop"}), 400
            df.drop(columns=[col], inplace=True)
            msg = f"Dropped column: {col}"

        elif step in ("fill_mean", "fill_median", "fill_mode", "fill_zero"):
            lbl_map = {"fill_mean": "mean", "fill_median": "median",
                       "fill_mode": "mode", "fill_zero": "0"}
            strat_map = {"fill_mean": "mean", "fill_median": "median",
                         "fill_mode": "most_frequent", "fill_zero": "constant"}
            count = 0
            for c in target_cols:
                if c not in df.columns:
                    continue
                nulls = int(df[c].isnull().sum())
                if nulls == 0:
                    continue
                s = strat_map[step]
                # mean/median only work for numeric
                if s in ("mean", "median") and (
                        df[c].dtype == object or
                        str(df[c].dtype).startswith("string")):
                    s = "most_frequent"
                kw = {"strategy": s}
                if s == "constant":
                    kw["fill_value"] = 0
                imp = SimpleImputer(**kw)
                df[[c]] = imp.fit_transform(df[[c]])
                count  += nulls
            msg = (f"Filled {count:,} missing values with "
                   f"{lbl_map[step]}{' in ' + col if col else ''}")

        elif step == "normalize_minmax":
            cols = [c for c in target_cols
                    if c in df.columns and df[c].dtype != object
                    and not str(df[c].dtype).startswith("string")]
            if not cols:
                return jsonify({"error": "No numeric columns to normalize"}), 400
            df[cols] = MinMaxScaler().fit_transform(df[cols])
            msg = f"Min-max normalized {len(cols)} column(s)"

        elif step == "normalize_zscore":
            cols = [c for c in target_cols
                    if c in df.columns and df[c].dtype != object
                    and not str(df[c].dtype).startswith("string")]
            if not cols:
                return jsonify({"error": "No numeric columns to standardize"}), 400
            df[cols] = StandardScaler().fit_transform(df[cols])
            msg = f"Z-score standardized {len(cols)} column(s)"

        else:
            return jsonify({"error": f"Unknown preprocessing step: {step}"}), 400

        save_df(sid, df)
        return jsonify({"message": msg,
                        "stats":   df_stats(df),
                        "columns": list(df.columns)})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/run", methods=["POST"])
def run_algo():
    try:
        sid  = get_sid()
        if sid not in STORE:
            return jsonify({"error": "No dataset loaded"}), 400

        body   = request.json or {}
        algo   = body.get("algo", "")
        target = body.get("target", "")
        mode   = body.get("mode", "supervised")
        k      = max(1, int(body.get("k", 3)))
        folds  = max(2, int(body.get("folds", 5)))
        df     = STORE[sid]["df"]

        feature_cols = [c for c in df.columns if c != target]
        if not feature_cols:
            return jsonify({"error": "No feature columns available"}), 400

        # Drop rows only where the TARGET is null (for supervised).
        # Features with nulls are handled inside prepare_X (imputation).
        if mode == "supervised":
            if not target or target not in df.columns:
                return jsonify({"error": "Select a valid target column"}), 400
            clean = df.dropna(subset=[target]).reset_index(drop=True)
        else:
            clean = df.copy()

        min_rows = max(folds * 2, 10)
        if len(clean) < min_rows:
            return jsonify({
                "error": (f"Only {len(clean)} usable rows — need at least {min_rows}. "
                          "Apply preprocessing (fill/remove missing) first.")
            }), 400

        X = prepare_X(clean, feature_cols)

        if mode == "unsupervised":
            result = run_unsupervised(algo, X, feature_cols, k)
        else:
            y_raw  = clean[target].values
            result = run_supervised(algo, X, y_raw, k, folds)

        return jsonify({
            "mode":       mode,
            "algo":       algo,
            "n_samples":  int(len(clean)),
            "n_features": int(len(feature_cols)),
            **result,
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── Supervised learning ──────────────────────────────────────────────────────
def run_supervised(algo: str, X: np.ndarray, y_raw, k: int, folds: int) -> dict:
    is_reg = algo in ("linreg", "ridge", "rfr")

    if is_reg:
        # Regression — encode y as float
        try:
            y = np.array(y_raw, dtype=np.float64)
        except (ValueError, TypeError):
            return {"error": "Target column must be numeric for regression"}
    else:
        # Classification — label-encode y
        y, le = encode_y(y_raw)
        # If encode_y returned floats (was numeric), treat as classification labels
        if le is None:
            y = y.astype(np.int64)
            le_classes = [str(int(c)) for c in np.unique(y)]
        else:
            le_classes = list(le.classes_)

    n_classes  = len(np.unique(y)) if not is_reg else 0
    # safe_folds: cap at half the dataset size, but keep user's requested folds
    # StratifiedKFold requires each fold to have at least 1 sample per class
    safe_folds = min(folds, len(y) // 2)
    safe_folds = max(2, safe_folds)

    # ── Build estimator (scale-first pipeline for distance/linear models) ──────
    needs_scaling = algo in ("knn", "lr", "linreg", "ridge")

    def make_model():
        if algo == "knn":
            base = (KNeighborsClassifier(n_neighbors=min(k, len(y)-1),
                                         metric="euclidean", n_jobs=-1)
                    if not is_reg else
                    KNeighborsRegressor(n_neighbors=min(k, len(y)-1),
                                        metric="euclidean", n_jobs=-1))
        elif algo == "dt":
            base = (DecisionTreeClassifier(min_samples_split=5,
                                           min_samples_leaf=2, random_state=42)
                    if not is_reg else
                    DecisionTreeRegressor(min_samples_split=5,
                                          min_samples_leaf=2, random_state=42))
        elif algo == "nb":
            base = GaussianNB(var_smoothing=1e-9)
        elif algo == "lr":
            base = LogisticRegression(max_iter=2000, solver="lbfgs",
                                      C=1.0, random_state=42)
        elif algo == "rfc":
            base = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                          n_jobs=-1, random_state=42)
        elif algo == "linreg":
            base = LinearRegression(n_jobs=-1)
        elif algo == "ridge":
            base = Ridge(alpha=1.0)
        elif algo == "rfr":
            base = RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                          n_jobs=-1, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        if needs_scaling:
            return Pipeline([("scaler", StandardScaler()), ("model", base)])
        return base

    estimator = make_model()

    cv_obj = (KFold(n_splits=safe_folds, shuffle=True, random_state=42)
              if is_reg else
              StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=42))

    # ── Cross-validation ────────────────────────────────────────────────────
    fold_results = []

    if is_reg:
        cv_res = cross_validate(
            estimator, X, y, cv=cv_obj, n_jobs=-1,
            scoring={"rmse": "neg_root_mean_squared_error",
                     "r2":   "r2",
                     "mae":  "neg_mean_absolute_error"},
        )
        rmses = -cv_res["test_rmse"]
        r2s   =  cv_res["test_r2"]
        maes  = -cv_res["test_mae"]
        for i in range(safe_folds):
            fold_results.append({
                "fold": i + 1,
                "RMSE": round(float(rmses[i]), 4),
                "R²":   round(float(r2s[i]),   4),
                "MAE":  round(float(maes[i]),   4),
            })
        summary = {
            "Mean RMSE": round(float(rmses.mean()), 4),
            "± Std":     round(float(rmses.std()),  4),
            "Mean R²":   round(float(r2s.mean()),   4),
            "Mean MAE":  round(float(maes.mean()),  4),
        }
    else:
        # Use macro for multiclass, macro also works for binary in sklearn 1.8
        cv_res = cross_validate(
            estimator, X, y, cv=cv_obj, n_jobs=-1,
            scoring={
                "accuracy":  "accuracy",
                "f1":        "f1_macro",
                "precision": "precision_macro",
                "recall":    "recall_macro",
            },
        )
        accs  = cv_res["test_accuracy"]
        f1s   = cv_res["test_f1"]
        precs = cv_res["test_precision"]
        recs  = cv_res["test_recall"]
        for i in range(safe_folds):
            fold_results.append({
                "fold":      i + 1,
                "Accuracy":  f"{round(float(accs[i])*100, 2)}%",
                "F1":        round(float(f1s[i]),   4),
                "Precision": round(float(precs[i]), 4),
                "Recall":    round(float(recs[i]),  4),
            })
        summary = {
            "Mean Accuracy":  f"{round(float(accs.mean())*100, 2)}%",
            "± Std":          f"{round(float(accs.std())*100,  2)}%",
            "Mean F1":        round(float(f1s.mean()),   4),
            "Mean Precision": round(float(precs.mean()), 4),
            "Mean Recall":    round(float(recs.mean()),  4),
        }

    # ── Full-data fit for detailed metrics ────────────────────────────────────
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    result = {"fold_results": fold_results, "summary": summary}

    if is_reg:
        result["full_metrics"] = {
            "RMSE": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
            "R²":   round(float(r2_score(y, y_pred)),                   4),
            "MAE":  round(float(mean_absolute_error(y, y_pred)),         4),
        }
    else:
        classes = le_classes if "le_classes" in dir() else [str(c) for c in np.unique(y)]
        prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
            y, y_pred, zero_division=0)
        result["class_report"] = [
            {
                "class":     classes[i] if i < len(classes) else str(i),
                "precision": round(float(prec_c[i]), 4),
                "recall":    round(float(rec_c[i]),  4),
                "f1":        round(float(f1_c[i]),   4),
                "support":   int(sup_c[i]),
            }
            for i in range(len(prec_c))
        ]
        cm = confusion_matrix(y, y_pred).tolist()
        result["confusion_matrix"] = {
            "labels": classes[:len(cm)],
            "matrix": cm,
        }

    return result


# ─── Unsupervised learning ────────────────────────────────────────────────────
def run_unsupervised(algo: str, X: np.ndarray,
                     feature_cols: list, k: int) -> dict:
    # Always scale before clustering/PCA
    Xs = StandardScaler().fit_transform(X)

    if algo == "kmeans":
        k = max(2, min(k, len(X) - 1))
        model  = KMeans(n_clusters=k, init="k-means++",
                        n_init=20, max_iter=500, random_state=42)
        labels = model.fit_predict(Xs)
        inertia = float(model.inertia_)
        counts  = {str(i): int(np.sum(labels == i)) for i in range(k)}

        # Silhouette — sample for speed on large datasets
        sil = 0.0
        if k > 1 and len(Xs) > k:
            n_sil = min(5000, len(Xs))
            idx   = np.random.choice(len(Xs), n_sil, replace=False)
            try:
                sil = float(silhouette_score(Xs[idx], labels[idx]))
            except Exception:
                pass

        centroids = []
        for ci in range(k):
            row = {"cluster": ci + 1, "size": int(counts[str(ci)])}
            for j, fname in enumerate(feature_cols[:6]):
                row[fname] = round(float(model.cluster_centers_[ci][j]), 4)
            centroids.append(row)

        return {
            "inertia":        round(inertia, 2),
            "silhouette":     round(sil, 4),
            "cluster_counts": counts,
            "centroids":      centroids,
        }

    elif algo == "dbscan":
        # Auto-tune eps from k-NN distance distribution (sample for speed)
        n_sample = min(3000, len(Xs))
        nn       = NearestNeighbors(
                       n_neighbors=min(5, n_sample - 1)).fit(Xs[:n_sample])
        dists, _ = nn.kneighbors(Xs[:n_sample])
        eps      = float(np.percentile(dists[:, -1], 90))
        min_samp = max(3, int(np.log(len(Xs))))

        model  = DBSCAN(eps=eps, min_samples=min_samp, n_jobs=-1)
        labels = model.fit_predict(Xs)
        n_clust = int(len(set(labels)) - (1 if -1 in labels else 0))
        noise   = int(np.sum(labels == -1))
        counts  = {str(i): int(np.sum(labels == i)) for i in range(n_clust)}

        return {
            "eps":            round(eps, 4),
            "min_samples":    min_samp,
            "n_clusters":     n_clust,
            "noise_points":   noise,
            "noise_pct":      round(noise / len(Xs) * 100, 2),
            "cluster_counts": counts,
        }

    elif algo == "pca":
        n_comp = min(len(feature_cols), len(Xs), 10)
        pca    = PCA(n_components=n_comp, random_state=42)
        pca.fit(Xs)
        ev  = pca.explained_variance_ratio_
        cum = np.cumsum(ev)

        components = [
            {
                "pc":             i + 1,
                "variance_pct":   round(float(ev[i]) * 100, 2),
                "cumulative_pct": round(float(cum[i]) * 100, 2),
            }
            for i in range(n_comp)
        ]
        loadings = []
        for i in range(min(3, n_comp)):
            top_idx = np.argsort(np.abs(pca.components_[i]))[::-1][:5]
            loadings.append({
                "pc": i + 1,
                "top_features": [
                    {
                        "feature": feature_cols[j],
                        "loading": round(float(pca.components_[i][j]), 4),
                    }
                    for j in top_idx if j < len(feature_cols)
                ],
            })

        n90 = int(np.searchsorted(cum, 0.90) + 1)
        return {
            "components":          components,
            "loadings":            loadings,
            "n_components_90pct":  n90,
        }

    raise ValueError(f"Unknown unsupervised algorithm: {algo}")


# ─── CLAPP Algorithm ──────────────────────────────────────────────────────────
class CLAPP:
    """
    CLAPP: A self-constructing feature clustering approach for anomaly detection.
    Based on: Gunupudi Rajesh Kumar et al., Future Generation Computer Systems 74 (2017) 417-429.

    Pipeline:
      1. Compute posterior probability of each feature w.r.t. each class label
         → System call pattern vectors C(j)
      2. Self-constructing fuzzy clustering of the pattern vectors
         → Clusters G1, G2, ... (threshold-driven, no fixed k)
      3. Build Transformation Matrix [features × clusters] using fuzzy similarity
      4. Reduced Process Matrix = Original[P×S] × Transformation[S×C]
      5. Classify new instances via kNN on the reduced matrix
    """

    def __init__(self, threshold: float = 0.64, sigma: float = 0.5, k: int = 1):
        self.threshold = threshold   # similarity threshold for cluster assignment
        self.sigma     = sigma       # initial deviation for membership function
        self.k         = k          # k for kNN classification
        self.fitted_   = False

    # ── Step 1: Posterior probability ─────────────────────────────────────────
    def _compute_posterior(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        For each feature j and each class d:
          Pr(Dd | Sj) = Σ[ X(i,j) * Md ] / Σ X(i,j)
        Returns pattern_vectors shape (n_features, n_classes).
        """
        classes = np.unique(y)
        n_feat  = X.shape[1]
        C = np.zeros((n_feat, len(classes)))
        for fi in range(n_feat):
            col_sum = X[:, fi].sum()
            if col_sum == 0:
                # Feature never appears → uniform distribution
                C[fi] = 1.0 / len(classes)
                continue
            for di, cls in enumerate(classes):
                mask  = (y == cls).astype(float)
                C[fi, di] = (X[:, fi] * mask).sum() / col_sum
        return C  # (n_features, n_classes)

    # ── Step 2: Fuzzy membership between two pattern vectors ──────────────────
    def _membership(self, ci: np.ndarray, cj: np.ndarray, sigma: float) -> float:
        """
        μ(C(i), C(j)) = Π_d [ ½ × (1 + exp( -( (ci_d - cj_d) / sigma )² )) ]
        Result always in (0, 1].
        """
        mu = 1.0
        for d in range(len(ci)):
            nr   = (ci[d] - cj[d]) / sigma
            mu  *= 0.5 * (1.0 + np.exp(-(nr ** 2)))
        return float(mu)

    # ── Step 3: Self-constructing clustering ──────────────────────────────────
    def _cluster_patterns(self, C: np.ndarray):
        """
        Incrementally cluster feature pattern vectors.
        Returns:
          cluster_means   list of np.ndarray  (one per cluster)
          cluster_members list of list[int]   (feature indices in each cluster)
          cluster_devs    list of float       (updated deviation per cluster)
        """
        n_feat    = C.shape[0]
        means     = [C[0].copy()]          # cluster means (initially = first pattern)
        members   = [[0]]                  # cluster member lists
        devs      = [self.sigma]           # one deviation per cluster

        for fi in range(1, n_feat):
            best_mu  = -1.0
            best_cid = -1

            for cid, (mean, dev) in enumerate(zip(means, devs)):
                mu = self._membership(C[fi], mean, dev)
                if mu > best_mu:
                    best_mu  = mu
                    best_cid = cid

            if best_mu >= self.threshold:
                # Add to best matching cluster and update mean
                members[best_cid].append(fi)
                means[best_cid] = np.mean(C[members[best_cid]], axis=0)
            else:
                # Create new cluster
                means.append(C[fi].copy())
                members.append([fi])
                devs.append(self.sigma)

        # Update deviations: sigma_final = initial_sigma + std of members
        for cid in range(len(members)):
            if len(members[cid]) > 1:
                member_vecs = C[members[cid]]
                devs[cid]   = self.sigma + float(np.std(member_vecs))
            # single-member cluster keeps initial sigma

        return means, members, devs

    # ── Step 4: Build transformation matrix ──────────────────────────────────
    def _build_transform(self, C: np.ndarray,
                         means: list, devs: list) -> np.ndarray:
        """
        T[fi, cid] = fuzzy similarity of pattern vector C[fi] to cluster mean cid
        Shape: (n_features, n_clusters)
        """
        n_feat  = C.shape[0]
        n_clust = len(means)
        T = np.zeros((n_feat, n_clust))
        for fi in range(n_feat):
            for cid in range(n_clust):
                if np.array_equal(C[fi], means[cid]):
                    T[fi, cid] = 1.0
                else:
                    T[fi, cid] = self._membership(C[fi], means[cid], devs[cid])
        return T

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X : (n_samples, n_features)  — already numeric, no NaNs
        y : (n_samples,)             — class labels
        """
        self.classes_       = np.unique(y)
        self.n_classes_     = len(self.classes_)

        # 1. Posterior probabilities → pattern vectors
        self.C_             = self._compute_posterior(X, y)   # (n_feat, n_cls)

        # 2. Self-constructing clustering
        self.cluster_means_, self.cluster_members_, self.cluster_devs_ = \
            self._cluster_patterns(self.C_)

        # 3. Transformation matrix
        self.T_             = self._build_transform(
                                  self.C_, self.cluster_means_, self.cluster_devs_)

        # 4. Reduced training matrix
        self.X_reduced_     = X @ self.T_          # (n_samples, n_clusters)
        self.y_train_       = y.copy()
        self.fitted_        = True
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def _reduce(self, X: np.ndarray) -> np.ndarray:
        return X @ self.T_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("CLAPP must be fit before predict")
        X_r    = self._reduce(X)
        k_safe = min(self.k, len(self.y_train_))
        
        # Use sklearn's optimized kNN instead of pure Python loops
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=k_safe, metric='euclidean', n_jobs=-1)
        knn.fit(self.X_reduced_, self.y_train_)
        return knn.predict(X_r)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def get_diagnostics(self, X_train: np.ndarray,
                        feature_names: list) -> dict:
        """Return all intermediate tables for UI visualisation."""
        n_feat    = X_train.shape[1]
        n_clust   = len(self.cluster_means_)

        # Posterior prob table
        posterior_table = []
        for fi in range(n_feat):
            row = {"feature": feature_names[fi] if fi < len(feature_names) else f"F{fi+1}"}
            for di, cls in enumerate(self.classes_):
                row[f"Pr({cls})"] = round(float(self.C_[fi, di]), 4)
            posterior_table.append(row)

        # Cluster summary
        cluster_summary = []
        for cid in range(n_clust):
            entry = {
                "cluster": f"C{cid+1}",
                "size":    len(self.cluster_members_[cid]),
                "members": [feature_names[fi] if fi < len(feature_names)
                            else f"F{fi+1}"
                            for fi in self.cluster_members_[cid]],
                "deviation": round(float(self.cluster_devs_[cid]), 4),
            }
            for di, cls in enumerate(self.classes_):
                entry[f"mean_D{di+1}({cls})"] = round(float(self.cluster_means_[cid][di]), 4)
            cluster_summary.append(entry)

        # Transformation matrix
        transform_rows = []
        for fi in range(n_feat):
            row = {"feature": feature_names[fi] if fi < len(feature_names) else f"F{fi+1}"}
            for cid in range(n_clust):
                row[f"C{cid+1}"] = round(float(self.T_[fi, cid]), 4)
            transform_rows.append(row)

        # Reduced matrix
        reduced_rows = []
        max_preview = min(self.X_reduced_.shape[0], 100) # Cap at 100 rows to prevent browser freezes
        for si in range(max_preview):
            row = {"process": f"P{si+1}", "label": str(self.y_train_[si])}
            for cid in range(n_clust):
                row[f"C{cid+1}"] = round(float(self.X_reduced_[si, cid]), 4)
            reduced_rows.append(row)

        return {
            "n_features":     n_feat,
            "n_clusters":     n_clust,
            "dim_reduction":  round((1 - n_clust / n_feat) * 100, 1) if n_feat > 0 else 0,
            "classes":        [str(c) for c in self.classes_],
            "threshold":      self.threshold,
            "sigma":          self.sigma,
            "posterior_table":  posterior_table,
            "cluster_summary":  cluster_summary,
            "transform_matrix": transform_rows,
            "reduced_matrix":   reduced_rows,
        }


# ─── CLAPP Route ──────────────────────────────────────────────────────────────
@app.route("/api/clapp", methods=["POST"])
def run_clapp():
    try:
        sid  = get_sid()
        if sid not in STORE:
            return jsonify({"error": "No dataset loaded"}), 400

        body      = request.json or {}
        target    = body.get("target", "")
        k         = max(1, int(body.get("k", 1)))
        threshold = float(body.get("threshold", 0.64))
        sigma     = float(body.get("sigma", 0.5))
        folds     = max(2, int(body.get("folds", 5)))

        df = STORE[sid]["df"]
        if not target or target not in df.columns:
            return jsonify({"error": "Select a valid target column"}), 400

        feature_cols = [c for c in df.columns if c != target]
        if not feature_cols:
            return jsonify({"error": "No feature columns available"}), 400

        clean = df.dropna(subset=[target]).reset_index(drop=True)
        if len(clean) < max(folds * 2, 10):
            return jsonify({"error": f"Only {len(clean)} rows — need at least {max(folds*2,10)}. Preprocess first."}), 400

        X      = prepare_X(clean, feature_cols)
        y_raw  = clean[target].values
        y, le  = encode_y(y_raw)
        y      = y.astype(int)
        if le is None:
            le_classes = [str(int(c)) for c in np.unique(y)]
        else:
            le_classes = list(le.classes_)

        # ── Cross-validation ──────────────────────────────────────────────────
        safe_folds   = max(2, min(folds, len(y) // max(len(np.unique(y)), 1)))
        cv_obj       = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=42)
        fold_results = []
        all_y_true, all_y_pred = [], []

        for fold_i, (tr_idx, te_idx) in enumerate(cv_obj.split(X, y)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            clf = CLAPP(threshold=threshold, sigma=sigma, k=k)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            acc = float(np.mean(y_pred == y_te))
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(y_pred.tolist())
            fold_results.append({"fold": fold_i + 1,
                                 "Accuracy": f"{round(acc*100, 2)}%",
                                 "Samples":  int(len(y_te))})

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        overall_acc = float(np.mean(all_y_true == all_y_pred))

        prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
            all_y_true, all_y_pred, zero_division=0)
        class_report = [
            {
                "class":     le_classes[i] if i < len(le_classes) else str(i),
                "precision": round(float(prec_c[i]), 4),
                "recall":    round(float(rec_c[i]),  4),
                "f1":        round(float(f1_c[i]),   4),
                "support":   int(sup_c[i]),
            }
            for i in range(len(prec_c))
        ]
        cm = confusion_matrix(all_y_true, all_y_pred).tolist()

        # ── Full-data fit for diagnostics ─────────────────────────────────────
        final_model = CLAPP(threshold=threshold, sigma=sigma, k=k)
        final_model.fit(X, y)
        diagnostics = final_model.get_diagnostics(X, feature_cols)

        summary = {
            "Overall Accuracy": f"{round(overall_acc*100, 2)}%",
            "Mean F1 (macro)":  round(float(np.mean(f1_c)), 4),
            "Mean Precision":   round(float(np.mean(prec_c)), 4),
            "Mean Recall":      round(float(np.mean(rec_c)), 4),
        }

        return jsonify({
            "mode":             "supervised",
            "algo":             "clapp",
            "n_samples":        int(len(clean)),
            "n_features":       int(len(feature_cols)),
            "fold_results":     fold_results,
            "summary":          summary,
            "class_report":     class_report,
            "confusion_matrix": {"labels": le_classes[:len(cm)], "matrix": cm},
            "diagnostics":      diagnostics,
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
