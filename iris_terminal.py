from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Input from user
print("Enter flower measurements:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

# Predict
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)[0]
print(f"Predicted species: {iris.target_names[prediction]}")