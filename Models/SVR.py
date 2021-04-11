from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

class SupportVectorRegressor():
  def __init__(self):
    super(SupportVectorRegressor).__init__()

  def forward(self, X_train, Y_train, X_test, Y_test):
    svr = make_pipeline(StandardScaler(), SVR(C=0.05, epsilon=0.01))
    svr.fit(X_train, Y_train.reshape(-1))
    out = svr.predict(X_test)
    out = out.reshape((out.shape[0], 1))
    return out
