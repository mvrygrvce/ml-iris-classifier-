from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import gradio as gr

# Load and split the dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel="linear"),
    "Decision Tree": DecisionTreeClassifier()
}

for model in models.values():
    model.fit(X_train, y_train)
# Prediction function
def predict_species(model_name, sepal_length, sepal_width, petal_length, petal_width):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    model = models[model_name]
    prediction = model.predict(input_data)[0]
    return f"Predicted species: {iris.target_names[prediction]}"

# Gradio interface
interface = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Dropdown(choices=list(models.keys()), label="Choose Model"),
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)")
    ],
    outputs="text",
    title="🌸 Iris Classifier with Model Selector",
    description="Choose a model and enter flower measurements to predict the iris species."
)

interface.launch()