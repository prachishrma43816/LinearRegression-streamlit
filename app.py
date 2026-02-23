import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Title
# -----------------------------
st.title("📊 Diabetes Progression Prediction using Linear Regression")
st.write("""
This app uses **Linear Regression** on the Diabetes dataset to predict disease progression and visualize the results.
""")

# -----------------------------
# Load Dataset
# -----------------------------
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Feature names for clarity
feature_names = diabetes.feature_names

# -----------------------------
# Sidebar: Train/Test Split
# -----------------------------
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20, 5)
random_state = st.sidebar.number_input("Random State", value=42)

# -----------------------------
# Train/Test Split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=random_state
)

# -----------------------------
# Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# -----------------------------
# Evaluation Metrics
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation Metrics")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R-squared (R²):** {r2:.2f}")

# -----------------------------
# Plotting
# -----------------------------
st.subheader("Visualizations")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# True vs Predicted
axs[0].scatter(y_test, y_pred, color="blue", alpha=0.5)
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
axs[0].set_title("True vs Predicted Values")
axs[0].set_xlabel("True Values")
axs[0].set_ylabel("Predicted Values")
axs[0].grid(True)

# Feature(BMI) vs Predicted Values
feature_index = feature_names.index("bmi")
axs[1].scatter(x_test[:, feature_index], y_pred, color="green", alpha=0.7)
axs[1].set_title("Feature (BMI) vs Predicted Values")
axs[1].set_xlabel("BMI (Feature 2)")
axs[1].set_ylabel("Predicted Diabetes Progression")
axs[1].grid(True)

plt.tight_layout()
st.pyplot(fig)
