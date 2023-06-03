import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
logged_model = 'runs:/d718e6779fb54feca1f4239848818c9f/model'

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

model = mlflow.sklearn.load_model(logged_model)
predictions = model.predict(X_test)
print(predictions)