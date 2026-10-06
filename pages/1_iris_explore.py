import streamlit as st
from sklearn.datasets import load_iris


iris = load_iris(as_frame=True)
iris_data = iris.frame.copy()
iris_data["species"] = iris_data["target"].map(dict(enumerate(iris.target_names)))
iris_data = iris_data.drop(columns="target")

st.title("The Iris Flower")
st.write(
	"Irises are flowering plants known for their distinctive three upright "
	"petals and three drooping sepals. Their name comes from the Greek word "
	"for rainbow, reflecting the wide range of colors found across the genus."
)

st.subheader("A classic dataset")
st.write(
	"The Iris dataset contains measurements for 150 flowers across three "
	"species: setosa, versicolor, and virginica. Sepal and petal length and "
	"width help distinguish these species and make the dataset useful for "
	"learning data analysis and machine learning."
)

metrics = st.columns(3)
metrics[0].metric("Flowers", len(iris_data))
metrics[1].metric("Species", iris_data["species"].nunique())
metrics[2].metric("Measurements per flower", 4)
