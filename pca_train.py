import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_mnist_csv(path):
    df = pd.read_csv(path)
    if 'label' in df.columns:
        y = df['label'].values.astype(int)
        X = df.drop(columns=['label']).values.astype(np.float32)
    else:
        y = df.iloc[:, 0].values.astype(int)
        X = df.iloc[:, 1:].values.astype(np.float32)
    return X / 255.0, y


def analyze_pca_variance(X, n_components=300, save_plot=True):

    print(f"\nAnalyzing explained variance for top {n_components} components...")

    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    eigenvalues = (S ** 2) / (X.shape[0] - 1)
    explained_var_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_var = np.cumsum(explained_var_ratio)

    # Save variance info
    k_vals = np.arange(1, len(explained_var_ratio) + 1)
    df_var = pd.DataFrame({
        "Component": k_vals,
        "ExplainedVariance": explained_var_ratio,
        "CumulativeVariance": cumulative_var
    })
    df_var.to_csv("pca_variance_analysis.csv", index=False)
    print("✅ Saved variance data to pca_variance_analysis.csv")

    for target in [0.80, 0.90, 0.95, 0.99]:
        k_target = np.argmax(cumulative_var >= target) + 1
        print(f"→ {target*100:.0f}% variance explained by {k_target} components")

    if save_plot:
        plt.figure(figsize=(7, 5))
        plt.plot(k_vals, cumulative_var * 100, lw=2)
        plt.axhline(90, color='orange', linestyle='--', label='90% variance')
        plt.axhline(95, color='green', linestyle='--', label='95% variance')
        plt.axhline(99, color='red', linestyle='--', label='99% variance')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance (%)")
        plt.title("PCA Cumulative Explained Variance (MNIST)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("pca_variance_plot.png", dpi=300)
        plt.show()
        print("✅ Saved plot as pca_variance_plot.png")

    return df_var, cumulative_var

X_train, _ = load_mnist_csv("MNIST_train.csv")
df_var, cum_var = analyze_pca_variance(X_train, n_components=300)
