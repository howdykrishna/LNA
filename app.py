import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Page setup
st.set_page_config(page_title="L&D Analytics & Prediction Portal", layout="wide")
st.title("📊 L&D Analytics & Prediction Dashboard")

# Load data
df = pd.read_csv("lnd_data.csv")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filter Data")
    dept = st.multiselect("Select Department", df['Department'].unique(), default=df['Department'].unique())
    location = st.multiselect("Select Location", df['Location'].unique(), default=df['Location'].unique())
    df = df[df['Department'].isin(dept) & df['Location'].isin(location)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", df['Employee_ID'].nunique())
col2.metric("Avg Skill Gap", round(df['Skill_Gap'].mean(), 2))
col3.metric("Training Completed (%)", f"{round(df['Training_Completion_Percentage'].mean(), 2)}%")
col4.metric("Avg Post-Training Score", round(df['Post_Training_Assessment_Score'].mean(), 2))

# Skill Gap Analysis
st.subheader("📉 Skill Gap Analysis by Department")
gap_chart = df.groupby('Department')['Skill_Gap'].mean().sort_values()
st.bar_chart(gap_chart)

# Training Effectiveness
st.subheader("📈 Training Effectiveness: Pre vs Post")
if 'Current_Skill_Level' in df.columns and 'Post_Training_Assessment_Score' in df.columns:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Current_Skill_Level', y='Post_Training_Assessment_Score', hue='Department', ax=ax)
    st.pyplot(fig)

# Predictive Model
st.subheader("🤖 Predict Post-Training Score")

# Prepare data
model_data = df[['Current_Skill_Level', 'Required_Skill_Level', 'Estimated_Training_Hours',
                 'Overall_Performance_Rating', 'Experience_Level', 'Post_Training_Assessment_Score']].dropna()

X = model_data.drop('Post_Training_Assessment_Score', axis=1)
y = model_data['Post_Training_Assessment_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# User input
st.markdown("### 📥 Enter details to predict outcome:")
col1, col2, col3 = st.columns(3)
csl = col1.slider("Current Skill Level", 1, 10, 5)
rsl = col2.slider("Required Skill Level", 1, 10, 8)
hours = col3.slider("Estimated Training Hours", 1, 50, 10)
opr = col1.slider("Overall Performance Rating", 1, 10, 7)
exp = col2.slider("Experience Level", 0, 30, 5)

pred_input = pd.DataFrame([[csl, rsl, hours, opr, exp]], 
                          columns=['Current_Skill_Level', 'Required_Skill_Level', 
                                   'Estimated_Training_Hours', 'Overall_Performance_Rating', 
                                   'Experience_Level'])

score = model.predict(pred_input)[0]
st.success(f"🎯 Predicted Post-Training Score: **{round(score, 2)}**")
