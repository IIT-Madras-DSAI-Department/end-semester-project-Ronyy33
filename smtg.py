import numpy as np
import pandas as pd
import time
from soft_en_algo import (
    pca_fit, pca_transform,
    SoftmaxSGD, confusion_matrix_and_f1
)

def load_mnist_csv(path):
    df = pd.read_csv(path)
    if 'label' in df.columns:
        y = df['label'].values.astype(int)
        X = df.drop(columns=['label']).values.astype(np.float32)
    if 'even' in df.columns:
        X = df.drop(columns=['even']).values.astype(np.float32)
        
    else:
        y = df.iloc[:, 0].values.astype(int)
        X = df.iloc[:, 1:].values.astype(np.float32)
    return X / 255.0, y


def train_softmax_ensemble_weighted_topk(
    X_train, y_train, X_val, y_val,
    n_models=15, n_comp=100, n_features_sub=28, top_k=8,
    lr=0.2, reg=1e-4, epochs=50, batch_size=64
):
    print(f"Running PCA with {n_comp} components...")
    components, mean_vec = pca_fit(X_train, n_components=n_comp)
    Xtr_pca = pca_transform(X_train, components, mean_vec)
    Xval_pca = pca_transform(X_val, components, mean_vec)
    n_total_features = Xtr_pca.shape[1]

    models, feature_subsets, weights, model_stats = [], [], [], []
    start = time.time()
    print(f"\nTraining {n_models} Softmax models (top_k={top_k}, total features each={n_features_sub})...")

    for i in range(n_models):
    
        rest_k = n_features_sub - top_k
        top_indices = np.arange(top_k)
        rest_indices = np.arange(top_k, n_total_features)
        rand_indices = np.random.choice(rest_indices, rest_k, replace=False)
        feat_idx = np.concatenate([top_indices, rand_indices])
        np.random.shuffle(feat_idx)
        feature_subsets.append(feat_idx)

      
        sm = SoftmaxSGD(n_features=n_features_sub, lr=lr, reg=reg,
                        epochs=epochs, batch_size=batch_size, seed=i)
        sm.fit(Xtr_pca[:, feat_idx], y_train)

     
        preds = sm.predict(Xval_pca[:, feat_idx])
        _, acc_i, f1_i = confusion_matrix_and_f1(y_val, preds)
        weights.append(f1_i)
        models.append(sm)
        model_stats.append({
            "model": i + 1,
            "acc": acc_i,
            "f1": f1_i,
            "features": feat_idx.tolist()
        })
        print(f"Model {i+1:02d}/{n_models}: F1={f1_i:.4f}, Acc={acc_i:.4f}")

    df_stats = pd.DataFrame(model_stats)
    df_stats.to_csv("ensemble_stats.csv", index=False)
    print("\nSaved per-model metrics to ensemble_stats.csv")

    weights = np.array(weights)
    weights /= np.sum(weights)
    print(f"\nModel weights (sum=1): {np.round(weights, 3)}")

    print("\nEvaluating weighted ensemble..")
    all_proba = []
    for i, sm in enumerate(models):
        proba = sm.predict_proba(Xval_pca[:, feature_subsets[i]])
        all_proba.append(proba * weights[i])
    mean_proba = np.sum(all_proba, axis=0)
    preds = np.argmax(mean_proba, axis=1)

    cm, acc, f1 = confusion_matrix_and_f1(y_val, preds)
    print(f"\n Weighted Ensemble accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"Total runtime: {time.time() - start:.2f} sec")
    return cm, acc, f1, weights, df_stats


def main():
    print("Loading data...")
    X_train, y_train = load_mnist_csv("MNIST_train.csv")
    X_val, y_val = load_mnist_csv("MNIST_validation.csv")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")


    cm, acc, f1, weights, df_stats = train_softmax_ensemble_weighted_topk(
        X_train, y_train, X_val, y_val,
        n_models=10, n_comp=87, n_features_sub=65, top_k=52,
        lr=0.05, reg=1e-4, epochs=60, batch_size=64
    )

    print("\nConfusion matrix:")
    print(cm)
    print(f"\nFinal Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"Average model F1: {df_stats['f1'].mean():.4f} ± {df_stats['f1'].std():.4f}")

if __name__ == "__main__":
    main()

