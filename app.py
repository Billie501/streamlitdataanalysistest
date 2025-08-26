import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced Incident Analytics", layout="wide")

st.title("📊 Advanced Incident Analytics & Safety Intelligence")
st.markdown("*Comprehensive insights to improve workplace safety and prevent future incidents*")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your incident CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- Enhanced Data Cleaning ---
    # Handle various date formats
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Standardize time columns
    time_columns = [col for col in df.columns if 'time' in col.lower()]
    for col in time_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.time
    
    # Extract hour from datetime for time analysis
    if 'incident_date' in df.columns:
        df['hour'] = pd.to_datetime(df['incident_date']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['incident_date']).dt.day_name()
        df['month'] = pd.to_datetime(df['incident_date']).dt.month
        df['year'] = pd.to_datetime(df['incident_date']).dt.year
    
    # Clean text columns
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    st.subheader("🔎 Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head())
    with col2:
        st.write("**Data Quality Summary:**")
        missing_data = df.isnull().sum()
        st.write(f"Total Records: {len(df)}")
        st.write(f"Columns: {len(df.columns)}")
        if missing_data.sum() > 0:
            st.write("Missing Values:")
            for col, missing in missing_data[missing_data > 0].items():
                st.write(f"  • {col}: {missing} ({missing/len(df)*100:.1f}%)")

    # --- Enhanced KPI Dashboard ---
    st.subheader("📈 Executive Dashboard")
    
    # Calculate advanced metrics
    total_incidents = len(df)
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    # Filter for current period if date exists
    if 'incident_date' in df.columns:
        df_current_month = df[(df['month'] == current_month) & (df['year'] == current_year)]
        df_last_month = df[(df['month'] == current_month-1) & (df['year'] == current_year)]
        
        # Calculate trends
        current_month_incidents = len(df_current_month)
        last_month_incidents = len(df_last_month)
        
        if last_month_incidents > 0:
            incident_trend = ((current_month_incidents - last_month_incidents) / last_month_incidents) * 100
        else:
            incident_trend = 0
    else:
        incident_trend = 0
        current_month_incidents = 0
    
    # Injury rate calculation
    injury_rate = 0
    if 'was_injured' in df.columns:
        injury_rate = (df['was_injured'].sum() / len(df)) * 100 if len(df) > 0 else 0
    
    # Display KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Incidents", total_incidents)
    col2.metric("This Month", current_month_incidents, f"{incident_trend:+.1f}%")
    col3.metric("Injury Rate", f"{injury_rate:.1f}%")
    col4.metric("Departments", df['department'].nunique() if 'department' in df.columns else 0)
    col5.metric("Locations", df['location'].nunique() if 'location' in df.columns else 0)

    # --- PREDICTIVE ANALYTICS ---
    st.subheader("🔮 Predictive Analytics & Forecasting")
    
    if 'incident_date' in df.columns and len(df) > 10:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Incident Trend Forecast**")
            
            # Create monthly aggregation for forecasting
            df_monthly = df.groupby([df['incident_date'].dt.year, df['incident_date'].dt.month]).size().reset_index()
            df_monthly['date'] = pd.to_datetime(df_monthly[['incident_date', 'level_1']].rename(columns={'incident_date': 'year', 'level_1': 'month'}))
            df_monthly = df_monthly.sort_values('date')
            
            if len(df_monthly) >= 3:
                # Simple moving average forecast
                window = min(3, len(df_monthly))
                df_monthly['forecast'] = df_monthly[0].rolling(window=window).mean().shift(1)
                
                # Create next 3 months prediction
                last_avg = df_monthly[0].tail(window).mean()
                future_dates = pd.date_range(start=df_monthly['date'].max() + pd.DateOffset(months=1), 
                                           periods=3, freq='M')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_monthly['date'], y=df_monthly[0], 
                                       mode='lines+markers', name='Actual Incidents'))
                fig.add_trace(go.Scatter(x=future_dates, y=[last_avg]*3, 
                                       mode='lines+markers', name='Predicted', 
                                       line=dict(dash='dash')))
                fig.update_layout(title="3-Month Incident Forecast", xaxis_title="Date", yaxis_title="Incidents")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**🎯 Risk Indicators**")
            
            # Calculate risk scores
            risk_factors = []
            
            # Department risk
            if 'department' in df.columns:
                dept_incidents = df['department'].value_counts()
                high_risk_dept = dept_incidents.index[0] if len(dept_incidents) > 0 else "Unknown"
                risk_factors.append(f"🔴 High Risk Department: {high_risk_dept} ({dept_incidents.iloc[0]} incidents)")
            
            # Time-based risk
            if 'hour' in df.columns:
                hour_risk = df['hour'].value_counts()
                peak_hour = hour_risk.index[0] if len(hour_risk) > 0 else 0
                risk_factors.append(f"⏰ Peak Risk Time: {peak_hour}:00 ({hour_risk.iloc[0]} incidents)")
            
            # Day of week risk
            if 'day_of_week' in df.columns:
                day_risk = df['day_of_week'].value_counts()
                risky_day = day_risk.index[0] if len(day_risk) > 0 else "Unknown"
                risk_factors.append(f"📅 Highest Risk Day: {risky_day} ({day_risk.iloc[0]} incidents)")
            
            # Injury severity prediction
            if 'was_injured' in df.columns and 'department' in df.columns:
                injury_by_dept = df.groupby('department')['was_injured'].mean().sort_values(ascending=False)
                if len(injury_by_dept) > 0:
                    high_injury_dept = injury_by_dept.index[0]
                    injury_pct = injury_by_dept.iloc[0] * 100
                    risk_factors.append(f"🏥 Injury Hotspot: {high_injury_dept} ({injury_pct:.1f}% injury rate)")
            
            for factor in risk_factors:
                st.write(factor)

    # --- ADVANCED TIME ANALYSIS ---
    if 'incident_date' in df.columns:
        st.subheader("⏰ Time Pattern Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Heatmap by hour and day
            if 'hour' in df.columns and 'day_of_week' in df.columns:
                heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='incidents')
                heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='incidents').fillna(0)
                
                # Order days correctly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_pivot = heatmap_pivot.reindex(day_order)
                
                fig = px.imshow(heatmap_pivot, 
                              title="Incident Heatmap: Day vs Hour",
                              labels=dict(x="Hour of Day", y="Day of Week", color="Incidents"),
                              color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Seasonal trends
            if 'month' in df.columns:
                monthly_incidents = df.groupby('month').size().reset_index(name='incidents')
                monthly_incidents['month_name'] = monthly_incidents['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
                
                fig = px.bar(monthly_incidents, x='month_name', y='incidents',
                           title="Seasonal Incident Patterns")
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Weekly patterns
            if 'day_of_week' in df.columns:
                daily_incidents = df['day_of_week'].value_counts().reindex(
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                ).reset_index()
                daily_incidents.columns = ['day', 'incidents']
                
                fig = px.line(daily_incidents, x='day', y='incidents', 
                            markers=True, title="Weekly Incident Pattern")
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

    # --- DEPARTMENT & LOCATION INSIGHTS ---
    st.subheader("🏢 Department & Location Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'department' in df.columns:
            dept_analysis = df.groupby('department').agg({
                df.columns[0]: 'count',  # incident count
                'was_injured': 'mean' if 'was_injured' in df.columns else lambda x: 0
            }).reset_index()
            
            if 'was_injured' in df.columns:
                dept_analysis.columns = ['department', 'incident_count', 'injury_rate']
                dept_analysis['injury_rate'] *= 100
                
                # Create bubble chart
                fig = px.scatter(dept_analysis, x='incident_count', y='injury_rate', 
                               size='incident_count', hover_name='department',
                               title="Department Risk Matrix",
                               labels={'incident_count': 'Total Incidents', 
                                     'injury_rate': 'Injury Rate (%)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(dept_analysis, x='department', y=dept_analysis.columns[1],
                           title="Incidents by Department")
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'location' in df.columns:
            location_incidents = df['location'].value_counts().head(10).reset_index()
            location_incidents.columns = ['location', 'incidents']
            
            fig = px.bar(location_incidents, x='incidents', y='location', 
                        orientation='h', title="Top 10 Incident Locations")
            st.plotly_chart(fig, use_container_width=True)

    # --- INCIDENT CATEGORIZATION & ANALYSIS ---
    if 'label' in df.columns or 'category' in df.columns:
        st.subheader("🏷️ Incident Category Analysis")
        
        category_col = 'label' if 'label' in df.columns else 'category'
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = df[category_col].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            
            fig = px.pie(category_counts, names='category', values='count',
                        title="Incident Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category trends over time if date is available
            if 'incident_date' in df.columns:
                category_trends = df.groupby([df['incident_date'].dt.date, category_col]).size().reset_index(name='count')
                
                fig = px.line(category_trends, x='incident_date', y='count', 
                            color=category_col, title="Category Trends Over Time")
                st.plotly_chart(fig, use_container_width=True)

    # --- ENHANCED TEXT ANALYSIS ---
    if 'incident_description' in df.columns:
        st.subheader("📝 Advanced Text Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**☁️ Incident Description Word Cloud**")
            text = " ".join(str(desc) for desc in df['incident_description'].dropna())
            if text.strip():
                # Remove common stop words
                stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
                words = text.lower().split()
                filtered_text = " ".join([word for word in words if word not in stop_words and len(word) > 2])
                
                wordcloud = WordCloud(width=400, height=300, background_color="white").generate(filtered_text)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No text available for analysis.")
        
        with col2:
            st.write("**🔍 Key Terms Frequency**")
            if text.strip():
                # Extract key terms
                words = text.lower().split()
                word_freq = pd.Series(words).value_counts().head(10)
                word_freq = word_freq[word_freq.index.str.len() > 3]  # Filter short words
                
                fig = px.bar(x=word_freq.values, y=word_freq.index, 
                           orientation='h', title="Most Common Terms")
                st.plotly_chart(fig, use_container_width=True)

    # --- BUSINESS RECOMMENDATIONS ---
    st.subheader("💡 AI-Driven Recommendations")
    
    recommendations = []
    
    # Time-based recommendations
    if 'hour' in df.columns:
        peak_hours = df['hour'].value_counts().head(2)
        if len(peak_hours) > 0:
            recommendations.append(f"🕐 **Peak Risk Hours**: Increase safety supervision during {peak_hours.index[0]}:00-{peak_hours.index[0]+1}:00 (highest incident time)")
    
    # Department recommendations
    if 'department' in df.columns:
        high_risk_depts = df['department'].value_counts().head(2)
        if len(high_risk_depts) > 0:
            recommendations.append(f"🏢 **Focus Area**: Prioritize safety training in {high_risk_depts.index[0]} department ({high_risk_depts.iloc[0]} incidents)")
    
    # Injury prevention
    if 'was_injured' in df.columns and df['was_injured'].sum() > 0:
        injury_rate = (df['was_injured'].sum() / len(df)) * 100
        if injury_rate > 10:
            recommendations.append(f"🏥 **Critical**: {injury_rate:.1f}% injury rate requires immediate safety protocol review")
        
        # Department-specific injury recommendations
        if 'department' in df.columns:
            dept_injury_rates = df.groupby('department')['was_injured'].agg(['sum', 'count', 'mean'])
            dept_injury_rates = dept_injury_rates[dept_injury_rates['count'] >= 3]  # Only departments with 3+ incidents
            if len(dept_injury_rates) > 0:
                worst_dept = dept_injury_rates['mean'].idxmax()
                worst_rate = dept_injury_rates.loc[worst_dept, 'mean'] * 100
                recommendations.append(f"🎯 **Targeted Intervention**: {worst_dept} has {worst_rate:.1f}% injury rate - implement enhanced safety measures")
    
    # Trend-based recommendations
    if 'incident_date' in df.columns and len(df) > 30:
        recent_30d = df[df['incident_date'] >= (datetime.now() - timedelta(days=30))]
        if len(recent_30d) > len(df) * 0.3:  # If 30% of incidents in last 30 days
            recommendations.append("📈 **Trend Alert**: Recent surge in incidents detected - conduct immediate safety audit")
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations[:5], 1):  # Limit to top 5
            st.write(f"{i}. {rec}")
    else:
        st.write("📊 **Data Analysis**: Upload more comprehensive data for personalized recommendations")

    # --- EXPORT INSIGHTS ---
    st.subheader("📤 Export Analytics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Report"):
            st.success("Analytics report generated! (Feature in development)")
    
    with col2:
        if st.button("📧 Email Insights"):
            st.success("Insights email sent! (Feature in development)")
    
    with col3:
        if st.button("📅 Schedule Reports"):
            st.success("Report scheduling configured! (Feature in development)")

else:
    st.info("👆 Upload your incident CSV file to unlock powerful safety analytics and predictive insights")
    st.write("**Expected columns for optimal analysis:**")
    st.write("• Reporter Name, Person Involved, Incident Date & Time")
    st.write("• Department & Location, Incident Description")
    st.write("• Label/Category, Injury Information (was_injured: boolean)")