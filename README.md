import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
grad_boost = GradientBoostingClassifier(n_estimators=100, random_state=42)
grad_boost.fit(X_train, y_train)
grad_boost_pred = grad_boost.predict(X_test)
ada_boost = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_boost.fit(X_train, y_train)
ada_boost_pred = ada_boost.predict(X_test)
quasilinear_pred = (grad_boost_pred + ada_boost_pred) / 2
grad_boost_accuracy = accuracy_score(y_test, grad_boost_pred)
ada_boost_accuracy = accuracy_score(y_test, ada_boost_pred)
quasilinear_accuracy = accuracy_score(y_test, quasilinear_pred.round())
print(f"Точность Gradient Boosting: {grad_boost_accuracy:.2f}")
print(f"Точность AdaBoost: {ada_boost_accuracy:.2f}")
print(f"Точность квазилинейной композиции: {quasilinear_accuracy:.2f}")
