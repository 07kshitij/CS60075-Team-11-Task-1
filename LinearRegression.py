from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class LinearRegressor():
  def __init__(self):
    super(LinearRegressor, self).__init__()

  def forward(self, X_train, Y_train, X_test, Y_test):
    reg = make_pipeline(StandardScaler(), LinearRegression())
    reg.fit(X_train, Y_train)
    out = reg.predict(X_test)
    out = out.reshape((out.shape[0], 1))
    return out
