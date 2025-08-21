import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Incident Data Insights", layout="wide")

st.title("📊 Incident Data Insights & Exploration")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your incident CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Data Cleaning ---
    if "incident_date" in df.columns:
        df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")

    if "incident_time" in df.columns:
        df["incident_time"] = pd.to_datetime(df["incident_time"], errors="coerce").dt.time

    st.subheader("🔎 Raw Data Preview")
    st.dataframe(df.head())

    # --- KPI Metrics ---
    st.subheader("📌 Key Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reports", len(df))
    col2.metric("Unique Departments", df["department"].nunique() if "department" in df else 0)
    col3.metric("Unique Locations", df["location"].nunique() if "location" in df else 0)

    # --- Incidents Over Time ---
    if "incident_date" in df.columns:
        st.subheader("📅 Incidents Over Time")
        incidents_by_date = df.groupby("incident_date").size().reset_index(name="count")
        fig = px.line(incidents_by_date, x="incident_date", y="count", title="Incidents Trend")
        st.plotly_chart(fig, use_container_width=True)

    # --- Department Breakdown ---
    if "department" in df.columns:
        st.subheader("🏢 Incidents by Department")
        fig = px.bar(df, x="department", title="Incidents by Department")
        st.plotly_chart(fig, use_container_width=True)

    # --- Injury Breakdown ---
    if "was_injured" in df.columns:
        st.subheader("🩹 Injury Breakdown")
        fig = px.pie(df, names="was_injured", title="Injured vs Not Injured")
        st.plotly_chart(fig, use_container_width=True)

    # --- Word Cloud of Incident Descriptions ---
    if "incident_description" in df.columns:
        st.subheader("☁️ Common Words in Incident Descriptions")
        text = " ".join(str(desc) for desc in df["incident_description"].dropna())
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No text available for word cloud.")

else:
    st.info("👆 Upload a CSV file to begin exploring.")
