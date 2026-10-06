from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris


FEATURES = [
	"sepal length (cm)",
	"sepal width (cm)",
	"petal length (cm)",
	"petal width (cm)",
]
iris = load_iris()


@st.cache_resource
def load_model():
	model_path = Path(__file__).resolve().parent.parent / "iris_model.pkl"
	with model_path.open("rb") as model_file:
		return pickle.load(model_file)


st.title("Predict a species")
st.write("Adjust the four measurements, then ask the model for a prediction.")

measurements = st.columns(4)
values = [
	measurements[0].slider("Sepal length (cm)", 4.0, 8.0, 5.0, 0.1),
	measurements[1].slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1),
	measurements[2].slider("Petal length (cm)", 1.0, 7.0, 4.0, 0.1),
	measurements[3].slider("Petal width (cm)", 0.1, 2.5, 1.0, 0.1),
]

if st.button("Predict species", type="primary"):
	features = pd.DataFrame([values], columns=FEATURES)
	prediction = load_model().predict(features)[0]
	species = iris.target_names[int(prediction)]
	st.success(f"The model predicts **{species}**.")

st.subheader("Prediction code")
st.code(
	"""import pickle
import pandas as pd

with open("iris_model.pkl", "rb") as model_file:
	model = pickle.load(model_file)

measurements = pd.DataFrame(
	[[5.0, 3.0, 4.0, 1.0]],
	columns=[
		"sepal length (cm)",
		"sepal width (cm)",
		"petal length (cm)",
		"petal width (cm)",
	],
)
species_id = model.predict(measurements)[0]
print(["setosa", "versicolor", "virginica"][int(species_id)])""",
	language="python",
)
