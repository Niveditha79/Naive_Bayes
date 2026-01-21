import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Naive Bayes Demo", layout="wide")

# ===================== SIDEBAR =====================
st.sidebar.title("Model Parameters")

model_type = st.sidebar.selectbox(
    "Select Naive Bayes Model",
    ["Gaussian Naive Bayes", "Bernoulli Naive Bayes"]
)

test_size = st.sidebar.slider(
    "Test Size (%)",
    min_value=10,
    max_value=50,
    value=30
)

# ===================== MAIN TITLE =====================
st.title("Naive Bayes Classification Demo 🚀")
st.write("A simple Streamlit app to demonstrate Naive Bayes classifiers")

# ===================== LOAD DATA =====================
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["Target"] = y

st.subheader("Generated Dataset")
st.dataframe(df.head())

# ===================== TRAIN TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42
)

# ===================== MODEL =====================
if model_type == "Gaussian Naive Bayes":
    model = GaussianNB()
    X_train_model = X_train
    X_test_model = X_test
else:
    # Bernoulli needs binary data
    threshold = np.mean(X_train)
    X_train_model = (X_train > threshold).astype(int)
    X_test_model = (X_test > threshold).astype(int)
    model = BernoulliNB()

model.fit(X_train_model, y_train)

# ===================== PREDICTION =====================
y_pred = model.predict(X_test_model)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)

# ===================== PERFORMANCE =====================
st.subheader("Model Performance")
st.success(f"Accuracy: {accuracy:.2f}")

# ===================== CONFUSION MATRIX =====================
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ===================== CLASSIFICATION REPORT =====================
st.subheader("Classification Report")

report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# ===================== USER PREDICTION =====================
st.subheader("Try Your Own Prediction")

col1, col2, col3, col4 = st.columns(4)

with col1:
    f1 = st.number_input("Sepal Length", min_value=0.0)
with col2:
    f2 = st.number_input("Sepal Width", min_value=0.0)
with col3:
    f3 = st.number_input("Petal Length", min_value=0.0)
with col4:
    f4 = st.number_input("Petal Width", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[f1, f2, f3, f4]])

    if model_type == "Bernoulli Naive Bayes":
        input_data = (input_data > threshold).astype(int)

    pred = model.predict(input_data)
    st.success(f"Predicted Class: {iris.target_names[pred[0]]}")
