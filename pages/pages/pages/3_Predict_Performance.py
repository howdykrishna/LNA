import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("🤖 Predict Learning Outcome")

df = pd.read_csv("lnd_data.csv")

# Required columns check
cols_needed = ['Current_Skill_Level', 'Required_Skill_Level', 'Estimated_Training_Hours',
               'Overall_Performance_Rating', 'Experience_Level', 'Post_Training_Assessment_Score']

if not all(col in df.columns for col in cols_needed):
    st.error("Dataset missing required columns.")
    st.stop()

# Clean data
model_df = df[cols_needed].dropna()
model_df = model_df.apply(pd.to_numeric, errors='coerce').dropna()

# Model
X = model_df.drop('Post_Training_Assessment_Score', axis=1)
y = model_df['Post_Training_Assessment_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# User input
st.subheader("📥 Input Employee Training Info:")
col1, col2, col3 = st.columns(3)
csl = col1.slider("Current Skill Level", 1, 10, 5)
rsl = col2.slider("Required Skill Level", 1, 10, 8)
hours = col3.slider("Training Hours", 1, 50, 12)
opr = col1.slider("Performance Rating", 1, 10, 7)
exp = col2.slider("Experience (Years)", 0, 30, 5)

input_df = pd.DataFrame([[csl, rsl, hours, opr, exp]],
                        columns=['Current_Skill_Level', 'Required_Skill_Level',
                                 'Estimated_Training_Hours', 'Overall_Performance_Rating',
                                 'Experience_Level'])

prediction = model.predict(input_df)[0]
st.success(f"🎯 Predicted Post-Training Score: **{round(prediction, 2)}**")
