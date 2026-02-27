import numpy as np

from sklearn.linear_model import SGDRegressor


class SklearnSGDRegressor:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.model = SGDRegressor(
            learning_rate='adaptive',
            eta0=learning_rate,
            max_iter=n_iter,
            fit_intercept=False,
            shuffle=True,
            random_state=42
        )
        self.weights = np.array([])

    def fit(self, X, y):
        self.model.fit(X, y)
        self.weights = self.model.coef_

        return self

    def predict(self, X):
        return self.model.predict(X)
