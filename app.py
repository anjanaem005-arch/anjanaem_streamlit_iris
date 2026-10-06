import streamlit as st
st.set_page_config(page_title="Iris Field Guide", page_icon="🌸", layout="wide")

page = st.navigation(
    [
        st.Page("pages/1_iris_explore.py", title="About Iris", icon="🌸"),
        st.Page("pages/2_data.py", title="Explore the Data", icon="📊"),
        st.Page("pages/3_prediction.py", title="Prediction & Code", icon="🔮"),
    ],
    position="top",
)
page.run()