import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Analytics Walkthrough")

df = pd.read_csv("lnd_data.csv")

with st.sidebar:
    st.header("Filter")
    departments = st.multiselect("Select Department", df["Department"].dropna().unique(), default=df["Department"].dropna().unique())
    locations = st.multiselect("Select Location", df["Location"].dropna().unique(), default=df["Location"].dropna().unique())
    methods = st.multiselect("Training Method", df["Training_Method"].dropna().unique(), default=df["Training_Method"].dropna().unique())

    df = df[df["Department"].isin(departments) & df["Location"].isin(locations) & df["Training_Method"].isin(methods)]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", df["Employee_ID"].nunique())
col2.metric("Avg Skill Gap", round(df["Skill_Gap"].mean(), 2))
col3.metric("Training Completion %", f"{round(df['Training_Completion_Percentage'].mean(), 1)}%")

# Charts
st.subheader("🧑‍💼 Employees Trained by Department")
st.bar_chart(df.groupby("Department")["Employee_ID"].nunique())

st.subheader("📈 Avg. Post-Training Score by Method")
st.bar_chart(df.groupby("Training_Method")["Post_Training_Assessment_Score"].mean())

st.subheader("📊 Business Impact Distribution")
st.write(df["Business_Impact"].value_counts().plot.pie(autopct='%1.1f%%', figsize=(5,5)))
st.pyplot()
