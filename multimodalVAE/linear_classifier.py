from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

def helper_fit_logreg(z_train, y_train, z_test, y_test):
    if z_train.shape[1] == 0:
        # if there are no latent coordinates, return zero
        return 0.0
    logreg = LogisticRegression(penalty="none").fit(z_train, y_train)
    class_pred = logreg.predict(z_test)
    accuracy = (class_pred == y_test).mean()
    auc = roc_auc_score(y_test, logreg.predict_proba(z_test)[:, 1])
    return accuracy, auc
