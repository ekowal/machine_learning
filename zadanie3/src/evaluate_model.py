import numpy as np
from src.make_dataset import prepare_mnist
from sklearn.metrics import classification_report

_, (X_test, y_test) = prepare_mnist()

def eval_model(model):
    y_pred = model.predict(X_test)
    y_pred_int = np.argmax(y_pred, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    report = classification_report(y_pred_int, y_test_int)
    print(report)

