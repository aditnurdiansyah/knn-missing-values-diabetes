import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Konfigurasi
MISSING_COLS = ["Glucose", "BloodPressure", "Insulin", "BMI"]
TARGET = "Outcome"
SEEDS = list(range(42, 52))  # 10 split acak; seed 42 = "split tunggal"
TEST_SIZE = 0.20

PARAM_GRID = {
    "knn__n_neighbors": list(range(1, 32, 2)),      # k = 1,3,...,31
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}

def make_imputers():
    """Daftar skenario penanganan missing value."""
    return {
        "A. Tanpa penanganan (0 dibiarkan)": None,
        "B. Hapus baris (listwise deletion)": "DROP",
        "C. Imputasi Mean": SimpleImputer(strategy="mean"),
        "D. Imputasi Median": SimpleImputer(strategy="median"),
        "E. Imputasi KNN (k=5)": KNNImputer(n_neighbors=5),
        "F. Imputasi MICE (Iterative)": IterativeImputer(
            max_iter=10, random_state=0, sample_posterior=False
        ),
    }

def build_pipeline(imputer):
    """StandardScaler -> (Imputer) -> KNN.

    Scaler ditaruh paling depan karena KNN berbasis jarak (sensitif skala),
    dan StandardScaler mengabaikan NaN saat menghitung mean/std.
    """
    steps = [("scaler", StandardScaler())]
    if imputer is not None:
        steps.append(("imputer", imputer))
    steps.append(("knn", KNeighborsClassifier()))
    return Pipeline(steps)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    n_zero = {c: int((df[c] == 0).sum()) for c in MISSING_COLS}
    return df, n_zero

def zeros_to_nan(df):
    out = df.copy()
    out[MISSING_COLS] = out[MISSING_COLS].replace(0, np.nan)
    return out

def evaluate_scenario(name, imputer, df_raw, df_nan, seed):
    """Jalankan 1 skenario pada 1 split. Return dict hasil."""
    if imputer == "DROP":
        data = df_nan.dropna(subset=MISSING_COLS).reset_index(drop=True)
        imputer_obj = None
    elif imputer is None:
        data = df_raw          # 0 tetap sebagai angka biasa
        imputer_obj = None
    else:
        data = df_nan          # NaN akan diisi oleh imputer di dalam pipeline
        imputer_obj = imputer

    X = data.drop(columns=TARGET)
    y = data[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )

    # --- KNN default (k=5), tanpa optimasi
    base = build_pipeline(imputer_obj)
    base.fit(X_tr, y_tr)
    pred_base = base.predict(X_te)

    # --- KNN + Grid Search (5-fold CV pada data latih saja)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        build_pipeline(imputer_obj), PARAM_GRID, cv=cv,
        scoring="accuracy", n_jobs=1, refit=True,
    )
    gs.fit(X_tr, y_tr)
    pred_gs = gs.predict(X_te)

    return {
        "scenario": name,
        "seed": seed,
        "n_total": len(data),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "acc_default": accuracy_score(y_te, pred_base),
        "f1_default": f1_score(y_te, pred_base),
        "acc_gridsearch": accuracy_score(y_te, pred_gs),
        "f1_gridsearch": f1_score(y_te, pred_gs),
        "cv_acc_gridsearch": gs.best_score_,
        "best_k": gs.best_params_["knn__n_neighbors"],
        "best_weights": gs.best_params_["knn__weights"],
        "best_metric": gs.best_params_["knn__metric"],
    }

# Visualisasi
def plot_missing_counts(n_zero, out_dir):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    cols = list(n_zero.keys())
    vals = [n_zero[c] for c in cols]
    bars = ax.bar(cols, vals, color="#d1495b")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=9)
    ax.set_ylabel("Jumlah nilai 0 (missing)")
    ax.set_title("Jumlah missing value per variabel (dari 767 baris)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_missing_counts.png"), dpi=170)
    plt.close(fig)


def plot_insulin_distribution(df_raw, df_nan, out_dir):
    """Ilustrasi (fit pada seluruh data) bagaimana tiap metode mengubah sebaran Insulin."""
    X = df_nan.drop(columns=TARGET)
    sc = StandardScaler().fit(X)
    Xs = pd.DataFrame(sc.transform(X), columns=X.columns)

    def back(imp):
        Z = imp.fit_transform(Xs)
        return pd.DataFrame(sc.inverse_transform(Z), columns=X.columns)["Insulin"]

    series = {
        "Asli (nilai 0 dibiarkan)": df_raw["Insulin"],
        "Mean": back(SimpleImputer(strategy="mean")),
        "Median": back(SimpleImputer(strategy="median")),
        "KNN Imputer": back(KNNImputer(n_neighbors=5)),
        "MICE": back(IterativeImputer(max_iter=10, random_state=0)),
    }
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.9), sharey=True, sharex=True)
    for ax, (t, s) in zip(axes, series.items()):
        ax.hist(s.clip(upper=500), bins=30, color="#00798c")
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("Insulin")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Frekuensi")
    fig.suptitle("Sebaran variabel Insulin sebelum dan sesudah imputasi", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_insulin_distribution.png"), dpi=170)
    plt.close(fig)

def plot_accuracy(summary, out_dir):
    names = [s.split(".")[0] for s in summary.index]
    labels = [s for s in summary.index]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w / 2, summary["acc_default_mean"] * 100, w,
           yerr=summary["acc_default_std"] * 100, capsize=3,
           label="KNN default (k=5)", color="#edae49")
    ax.bar(x + w / 2, summary["acc_gridsearch_mean"] * 100, w,
           yerr=summary["acc_gridsearch_std"] * 100, capsize=3,
           label="KNN + Grid Search", color="#00798c")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=7)
    ax.set_ylim(60, 85)
    ax.set_ylabel("Akurasi data uji (%)")
    ax.set_title("Rata-rata akurasi pada 10 split acak (error bar = simpangan baku)")
    ax.legend(loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_accuracy.png"), dpi=170)
    plt.close(fig)

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/diabetes.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--repeats", type=int, default=len(SEEDS),
                    help="jumlah split acak (default 10)")
    args = ap.parse_args()
    if not os.path.isfile(args.csv):
        raise SystemExit(
            f"File '{args.csv}' tidak ditemukan.\n"
            "Jalankan dari folder utama repo, atau tentukan lokasi file dengan:\n"
            "    python knn_missing_values.py --csv <lokasi/diabetes.csv>"
        )
    os.makedirs(args.out, exist_ok=True)

    df_raw, n_zero = load_data(args.csv)
    df_nan = zeros_to_nan(df_raw)

    print("=" * 66)
    print(f"Dataset: {df_raw.shape[0]} baris x {df_raw.shape[1]} kolom")
    print("Jumlah nilai 0 (dianggap missing):")
    for c, v in n_zero.items():
        print(f"  {c:15s}: {v:4d} ({v / len(df_raw) * 100:5.1f}%)")
    n_complete = len(df_nan.dropna(subset=MISSING_COLS))
    print(f"Baris lengkap (tanpa missing di 4 variabel): {n_complete}")
    print("Distribusi kelas:", df_raw[TARGET].value_counts().to_dict())
    print("=" * 66)

    plot_missing_counts(n_zero, args.out)
    plot_insulin_distribution(df_raw, df_nan, args.out)

    scenarios = make_imputers()
    rows = []
    t0 = time.time()
    for seed in SEEDS[: args.repeats]:
        for name, imp in scenarios.items():
            rows.append(evaluate_scenario(name, imp, df_raw, df_nan, seed))
        print(f"[seed {seed}] selesai  (total {time.time() - t0:.0f} dtk)")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "results_per_split.csv"), index=False)

    # ---- Ringkasan rata-rata
    agg = res.groupby("scenario").agg(
        n_total=("n_total", "first"),
        acc_default_mean=("acc_default", "mean"),
        acc_default_std=("acc_default", "std"),
        acc_gridsearch_mean=("acc_gridsearch", "mean"),
        acc_gridsearch_std=("acc_gridsearch", "std"),
        f1_gridsearch_mean=("f1_gridsearch", "mean"),
        cv_acc_mean=("cv_acc_gridsearch", "mean"),
        k_median=("best_k", "median"),
    )
    base_d = agg.loc["A. Tanpa penanganan (0 dibiarkan)", "acc_default_mean"]
    base_g = agg.loc["A. Tanpa penanganan (0 dibiarkan)", "acc_gridsearch_mean"]
    agg["delta_default_vs_A"] = agg["acc_default_mean"] - base_d
    agg["delta_grid_vs_A"] = agg["acc_gridsearch_mean"] - base_g
    agg["delta_grid_vs_default"] = agg["acc_gridsearch_mean"] - agg["acc_default_mean"]
    agg.to_csv(os.path.join(args.out, "summary.csv"))

    # ---- Hasil split tunggal (seed 42)
    single = res[res["seed"] == SEEDS[0]].set_index("scenario")
    single.to_csv(os.path.join(args.out, "results_single_split_seed42.csv"))

    # ---- Berapa split yang membaik dibanding skenario A (pairwise)
    piv_g = res.pivot(index="seed", columns="scenario", values="acc_gridsearch")
    wins = {}
    for s in piv_g.columns:
        if s.startswith("A."):
            continue
        d = piv_g[s] - piv_g["A. Tanpa penanganan (0 dibiarkan)"]
        wins[s] = {"lebih_baik": int((d > 0).sum()),
                   "sama": int((d == 0).sum()),
                   "lebih_buruk": int((d < 0).sum())}

    plot_accuracy(agg, args.out)

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"n_zero": n_zero, "n_complete": n_complete,
                   "wins_vs_A_gridsearch": wins,
                   "n_splits": len(SEEDS[: args.repeats])}, f, indent=2)

    # ---- Cetak ringkasan
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== HASIL SPLIT TUNGGAL (seed 42, data uji 20%) ===")
    print((single[["n_total", "acc_default", "acc_gridsearch", "best_k",
                   "best_weights", "best_metric"]]
           .round(4)).to_string())
    print(f"\n=== RATA-RATA {len(SEEDS[: args.repeats])} SPLIT (akurasi data uji) ===")
    show = agg[["n_total", "acc_default_mean", "acc_default_std",
                "acc_gridsearch_mean", "acc_gridsearch_std",
                "delta_default_vs_A", "delta_grid_vs_A",
                "delta_grid_vs_default"]].round(4)
    print(show.to_string())
    print(f"\nJumlah split (dari {len(SEEDS[: args.repeats])}) di mana Grid Search lebih baik / sama / lebih buruk vs skenario A:")
    for k, v in wins.items():
        print(f"  {k:38s} {v}")
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik. Output di folder: {args.out}/")

if __name__ == "__main__":
    main()
