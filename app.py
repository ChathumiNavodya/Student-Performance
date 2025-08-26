import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report

# --- Page Config ---
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    path = r"E:\My project\Student Performance dataset\StudentsPerformance.csv"
    df = pd.read_csv(path)
    return df

df = load_data()

st.title("📊 Student Performance Dashboard")

# --- Data Cleaning ---
st.header("🔍 Data Preview & Cleaning")

st.write("Initial Dataset Shape:", df.shape)
st.dataframe(df.head())

# Handle missing values
df = df.dropna()

# Convert categorical columns
cat_cols = df.select_dtypes(include="object").columns.tolist()

# Define target variable
df["math_passed"] = (df["math score"] >= 50).astype(int)

# --- EDA Section ---
st.header("📈 Exploratory Data Analysis (EDA)")

tab1, tab2, tab3 = st.tabs(["Distributions", "Correlation", "Group Analysis"])

with tab1:
    st.subheader("Score Distributions")
    for col in ["math score", "reading score", "writing score"]:
        fig = px.histogram(df, x=col, nbins=20, title=f"Distribution of {col}", color_discrete_sequence=["skyblue"])
        st.plotly_chart(fig)

with tab2:
    st.subheader("Correlation Heatmap")
    corr = df[["math score", "reading score", "writing score"]].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Between Scores", color_continuous_scale="Blues")
    st.plotly_chart(fig)

with tab3:
    st.subheader("Performance by Gender")
    fig = px.box(df, x="gender", y="math score", color="gender", title="Math Scores by Gender")
    st.plotly_chart(fig)

# --- Model Training ---
st.header("🤖 Model Training & Evaluation")

# Features and target
X = df.drop(columns=["math score", "math_passed"])
y = df["math_passed"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), ["reading score", "writing score"])
])

# Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# --- Metrics ---
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

st.metric("✅ Accuracy", f"{accuracy:.2f}")
st.metric("📊 AUC", f"{auc:.2f}")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
fig.update_layout(title=f"ROC Curve (AUC={auc:.2f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig)

# --- Feature Importance ---
st.subheader("🔑 Feature Importance")
coef = model.named_steps["classifier"].coef_[0]
features = model.named_steps["preprocessor"].get_feature_names_out()
importance = pd.DataFrame({"Feature": features, "Coefficient": coef}).sort_values(by="Coefficient")

fig = px.bar(importance, x="Coefficient", y="Feature", orientation="h", color="Coefficient", title="Feature Importance")
st.plotly_chart(fig)

# --- Classification Report ---
st.subheader("📑 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())
