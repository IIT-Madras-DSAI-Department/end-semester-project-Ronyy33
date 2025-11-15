import numpy as np

def pca_fit(X, n_components=250):
    mean_vec = np.mean(X, axis=0)
    Xc = X - mean_vec
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    return components, mean_vec

def pca_transform(X, components, mean_vec):
    Xc = X - mean_vec
    return np.dot(Xc, components.T)

class SoftmaxSGD:
    def __init__(self, n_features, n_classes=10, lr=0.2, reg=1e-4,
                 batch_size=64, epochs=50, seed=0):
        rng = np.random.RandomState(seed)
        self.W = 0.01 * rng.randn(n_features + 1, n_classes)
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_classes = n_classes

    @staticmethod
    def _softmax(z):
        z -= np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        n = X.shape[0]
        Xb = np.hstack([np.ones((n, 1)), X])
        vel = np.zeros_like(self.W)
        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            Xb, y = Xb[idx], y[idx]
            for i in range(0, n, self.batch_size):
                xb = Xb[i:i+self.batch_size]
                yb = y[i:i+self.batch_size]
                logits = xb @ self.W
                probs = SoftmaxSGD._softmax(logits)
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(yb)), yb] = 1
                grad = xb.T @ (probs - onehot) / xb.shape[0]
                grad += self.reg * np.vstack([np.zeros((1, self.W.shape[1])), self.W[1:]])
                # Momentum update
                vel = 0.9 * vel - self.lr * grad
                self.W += vel
            # Learning rate decay every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.lr *= 0.5

    def predict_proba(self, X):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        logits = Xb @ self.W
        return SoftmaxSGD._softmax(logits)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def confusion_matrix_and_f1(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    eps = 1e-12
    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    acc = np.trace(cm) / np.sum(cm)
    return cm, acc, np.mean(f1s)
