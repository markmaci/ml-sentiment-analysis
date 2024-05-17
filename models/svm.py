import time
import numpy as np

class CustomSVM:
    def __init__(self, C=1.0, learning_rate=0.001, epochs=1000, batch_size=100):
        self.C = C
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = None
        self.b = None

    def hingeloss(self, w, b, X, Y):
        loss = 0.5 * np.dot(w, w.T)
        for i in range(X.shape[0]):
            ti = Y[i] * (np.dot(X[i], w.T) + b)
            loss += self.C * max(0, 1 - ti)
        return loss

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        losses = []

        for epoch in range(self.epochs):
            start_time = time.time()
            ids = np.arange(n_samples)
            np.random.shuffle(ids)
            
            grad_w = 0
            grad_b = 0

            for batch_start in range(0, n_samples, self.batch_size):
                for j in range(batch_start, min(batch_start + self.batch_size, n_samples)):
                    idx = ids[j]
                    condition = Y[idx] * (np.dot(X[idx], self.w) + self.b) >= 1
                    if not condition:
                        grad_w += self.C * Y[idx] * X[idx]
                        grad_b += self.C * Y[idx]

                self.w -= self.lr * (2 * self.w - grad_w)
                self.b += self.lr * grad_b

            loss = self.hingeloss(self.w, self.b, X, Y)
            losses.append(loss)
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss:.4f} - Time: {epoch_time:.2f} seconds")

        return self.w, self.b, losses


    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
