from sklearn.datasets import load_iris
# Load the Iris dataset
iris = load_iris()
# Print basic info
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First 5 rows of data:")
print(iris.data[:5])
print("First 5 labels:")
print(iris.target[:5])