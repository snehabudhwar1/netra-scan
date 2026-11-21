"""Charts used in the reports (saved as PNG files)."""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]*100:.0f}%", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xlabel("Model said"); ax.set_ylabel("Doctor said (true grade)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_comparison(df, path):
    sets = list(df["feature_set"].unique())
    models = list(dict.fromkeys(df["model"].str.replace(r"\(.*\)", "", regex=True)))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.8 / max(len(sets), 1)
    for k, s in enumerate(sets):
        vals = []
        for m in models:
            sub = df[(df.feature_set == s) & (df.model.str.replace(r"\(.*\)", "", regex=True) == m)]
            vals.append(sub["cv_qwk"].max() if len(sub) else 0)
        ax.bar(np.arange(len(models)) + k * width, vals, width, label=s)
    ax.set_xticks(np.arange(len(models)) + width * (len(sets) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Cross-validated QWK (higher is better)")
    ax.set_title("Which features and which model work best?")
    ax.legend(title="features"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_recall(final_rep, base_rep, path):
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.bar(x - 0.2, [base_rep["recall_per_grade"][str(i)] for i in x], 0.4, label="metadata-only baseline")
    ax.bar(x + 0.2, [final_rep["recall_per_grade"][str(i)] for i in x], 0.4, label="selected model")
    ax.set_xticks(x); ax.set_xlabel("True grade"); ax.set_ylabel("Recall (share found)")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("How many photos of each grade were found? (test set)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
