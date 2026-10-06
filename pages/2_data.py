import streamlit as st
from sklearn.datasets import load_iris


FEATURES = [
	"sepal length (cm)",
	"sepal width (cm)",
	"petal length (cm)",
	"petal width (cm)",
]

iris = load_iris(as_frame=True)
iris_data = iris.frame.copy()
iris_data["species"] = iris_data["target"].map(dict(enumerate(iris.target_names)))
iris_data = iris_data.drop(columns="target")

st.title("Explore the measurements")
st.write("Compare flower counts and measurements across the three species.")

left, right = st.columns(2)
with left:
	st.subheader("Flowers by species")
	species_counts = iris_data["species"].value_counts().rename_axis("Species")
	st.bar_chart(species_counts)
with right:
	st.subheader("Average measurements")
	average_by_species = iris_data.groupby("species", sort=False)[FEATURES].mean()
	st.bar_chart(average_by_species)

st.subheader("Compare two measurements")
x_feature, y_feature = st.columns(2)
x_axis = x_feature.selectbox("Horizontal axis", FEATURES, index=2)
y_axis = y_feature.selectbox("Vertical axis", FEATURES, index=3)
st.scatter_chart(
	iris_data,
	x=x_axis,
	y=y_axis,
	color="species",
	height=420,
)

with st.expander("View the dataset"):
	st.dataframe(iris_data, use_container_width=True, hide_index=True)
