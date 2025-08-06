import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧠 Skill Gap Analysis")

df = pd.read_csv("lnd_data.csv")
df = df.dropna(subset=["Skill_Name", "Skill_Gap"])

# Bar chart
st.subheader("📉 Avg. Skill Gap by Skill Name")
skill_gap = df.groupby("Skill_Name")["Skill_Gap"].mean().sort_values(ascending=False)
st.bar_chart(skill_gap)

# Heatmap
st.subheader("📊 Heatmap: Skill Gap by Department & Skill Category")
heatmap_df = df.pivot_table(values="Skill_Gap", index="Department", columns="Skill_Category", aggfunc="mean")
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu")
st.pyplot()
