import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Page settings
st.set_page_config(page_title="L&D Analytics Portal", layout="wide")
st.title("📊 L&D Analytics & Learning Prediction Portal")

# Load the data
try:
    df = pd.read_csv("lnd_data.csv")
except FileNotFoundError:
    st.error("❌ File 'lnd_data.csv' not found. Please place it in the same folder as app.py.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filter Data")
    dept = st.multiselect("Select Department", df['Department'].dropna().unique(), default=df['Department'].dropna().unique())
    location = st.multiselect("Select Location", df['Location'].dropna().unique(), default=df['Location'].dropna().unique())
    df = df[df['Department'].isin(dept) & df['Location'].isin(location)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", df['Employee_ID'].nunique())
col2.metric("Avg Skill Gap", round(df['Skill_Gap'].mean(), 2))
col3.metric("Training Completed (%)", f"{round(df['Training_Completion_Percentage'].mean(), 2)}%")
col4.metric("Avg Post-Training Score", round(df['Post_Training_Assessment_Score'].mean(), 2))

st.divider()

# Skill Gap Analysis
st.subheader("📉 Skill Gap by Department")
if 'Skill_Gap' in df.columns and 'Department' in df.columns:
    gap_chart = df.groupby('Department')['Skill_Gap'].mean().sort_values()
    st.bar_chart(gap_chart)
else:
    st.warning("⚠️ Skill Gap or Department column missing from dataset.")

# Training Effectiveness
st.subheader("📈 Training Effectiveness: Pre vs Post Score")
if 'Current_Skill_Level' in df.columns and 'Post_Training_Assessment_Score' in df.columns:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Current_Skill_Level', y='Post_Training_Assessment_Score', hue='Department', ax=ax)
    st.pyplot(fig)
else:
    st.warning("⚠️ Columns for skill level or post-training score missing.")

st.divider()

# Predictive Modeling
st.subheader("🤖 Predict Post-Training Assessment Score")

required_cols = ['Current_Skill_Level', 'Required_Skill_Level', 'Estimated_Training_Hours',
                 'Overall_Performance_Rating', 'Experience_Level', 'Post_Training_Assessment_Score']

# Filter for required columns and clean
if all(col in df.columns for col in required_cols):
    model_data = df[required_cols].copy()
    model_data = model_data.dropna()
    model_data = model_data.apply(pd.to_numeric, errors='coerce')
    model_data = model_data.dropna()

    # Train the model
    X = model_data.drop('Post_Training_Assessment_Score', axis=1)
    y = model_data['Post_Training_Assessment_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # User input
    st.markdown("### 📥 Input Training Info:")
    col1, col2, col3 = st.columns(3)
    csl = col1.slider("Current Skill Level", 1, 10, 5)
    rsl = col2.slider("Required Skill Level", 1, 10, 8)
    hours = col3.slider("Estimated Training Hours", 1, 50, 10)
    opr = col1.slider("Overall Performance Rating", 1, 10, 7)
    exp = col2.slider("Experience Level (Years)", 0, 30, 5)

    # Prediction
    try:
        pred_input = pd.DataFrame([[csl, rsl, hours, opr, exp]],
                                  columns=['Current_Skill_Level', 'Required_Skill_Level',
                                           'Estimated_Training_Hours', 'Overall_Performance_Rating',
                                           'Experience_Level'])
        prediction = model.predict(pred_input)[0]
        st.success(f"🎯 Predicted Post-Training Score: **{round(prediction, 2)}**")

    except Exception as e:
        st.error("⚠️ Prediction failed. Please check input values.")
else:
    st.warning("⚠️ Some required columns are missing for prediction.")
