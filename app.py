import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz
import os

st.set_page_config(page_title="Google Play Store Analytics", layout="wide")
st.title("📱 Google Play Store Analytics Dashboard")

# Sidebar for testing & live control
st.sidebar.header("Dashboard Settings")
ist_tz = pytz.timezone('Asia/Kolkata')
current_ist_time = datetime.now(ist_tz)
st.sidebar.write(f"**Current IST Time:** {current_ist_time.strftime('%I:%M %p')}")

bypass_time = st.sidebar.checkbox("Demo Mode (Show all charts regardless of IST window)", value=True)

def is_active(start_hour, end_hour):
    if bypass_time:
        return True
    current_hour = datetime.now(ist_tz).hour
    return start_hour <= current_hour < end_hour

@st.cache_data
def load_data():
    apps_file = 'google_play_store_dataset.csv'
    reviews_file = 'googleplaystore_user_reviews.csv' if os.path.exists('googleplaystore_user_reviews.csv') else '17-googleplaystore_user_reviews.csv'
    
    if not os.path.exists(apps_file):
        st.error(f"Missing file: `{apps_file}` in repository root!")
        return pd.DataFrame()

    df_apps = pd.read_csv(apps_file)
    df_reviews = pd.read_csv(reviews_file) if os.path.exists(reviews_file) else pd.DataFrame(columns=['App', 'Sentiment_Subjectivity'])
    
    df_apps = df_apps[df_apps['Category'] != '1.9'].copy()
    df_apps['Reviews'] = pd.to_numeric(df_apps['Reviews'], errors='coerce').fillna(0)
    df_apps['Rating'] = pd.to_numeric(df_apps['Rating'], errors='coerce')
    df_apps['Installs_Numeric'] = pd.to_numeric(df_apps['Installs'].astype(str).str.replace(r'[+,]', '', regex=True), errors='coerce').fillna(0)
    
    def parse_size(val):
        val = str(val).strip()
        if 'M' in val: return pd.to_numeric(val.replace('M', ''), errors='coerce')
        elif 'k' in val or 'K' in val: return pd.to_numeric(val.replace('k', '').replace('K', ''), errors='coerce') / 1024
        return np.nan
    
    df_apps['Size_MB'] = df_apps['Size'].apply(parse_size)
    df_apps['Price_Numeric'] = pd.to_numeric(df_apps['Price'].astype(str).str.replace('$', '', regex=False), errors='coerce').fillna(0)
    df_apps['Revenue'] = df_apps['Price_Numeric'] * df_apps['Installs_Numeric']
    df_apps['Last Updated'] = pd.to_datetime(df_apps['Last Updated'], errors='coerce')
    
    if not df_reviews.empty and 'Sentiment_Subjectivity' in df_reviews.columns:
        user_sentiments = df_reviews.groupby('App')['Sentiment_Subjectivity'].mean().reset_index()
        return pd.merge(df_apps, user_sentiments, on='App', how='left')
    return df_apps

df = load_data()

if df.empty:
    st.warning("No data loaded. Please ensure your CSV files are uploaded to your GitHub repository.")
    st.stop()

# --- Task 1 (5 PM - 7 PM IST) ---
st.header("Task 1: Size vs Rating (Bubble Chart)")
if is_active(17, 19):
    cats_t1 = ['GAME', 'BEAUTY', 'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'ENTERTAINMENT', 'SOCIAL', 'EVENTS']
    mask_t1 = (
        (df['Rating'] > 3.5) &
        (df['Category'].str.upper().isin(cats_t1)) &
        (df['Reviews'] > 500) &
        (~df['App'].str.contains('s', case=False, na=False)) &
        (df['Sentiment_Subjectivity'] > 0.5) &
        (df['Installs_Numeric'] > 50000) &
        (df['Size_MB'].notna())
    )
    f1 = df[mask_t1].copy()
    f1['Category'] = f1['Category'].replace({'BEAUTY': 'सौंदर्य', 'BUSINESS': 'வணிகம்', 'DATING': 'Partnersuche'})
    fig1 = px.scatter(f1, x='Size_MB', y='Rating', size='Installs_Numeric', color='Category', color_discrete_map={'GAME': '#FF1493'}, hover_name='App', size_max=45)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Task 1 chart is active only between 5:00 PM and 7:00 PM IST.")

# --- Task 2 (6 PM - 8 PM IST) ---
st.header("Task 2: Global Installs Map")
if is_active(18, 20):
    mask_t2 = ~df['Category'].str.upper().str.startswith(('A', 'C', 'G', 'S'))
    f2 = df[mask_t2].copy()
    top_5 = f2.groupby('Category')['Installs_Numeric'].sum().nlargest(5).reset_index()
    top_5['Exceeds_1M'] = top_5['Installs_Numeric'] > 1_000_000
    top_5['Formatted_Installs'] = top_5['Installs_Numeric'].apply(lambda x: f"{x:,.0f}")
    fig2 = px.choropleth(top_5, locations=['USA', 'IND', 'BRA', 'IDN', 'RUS'], locationmode='ISO-3', color='Installs_Numeric', hover_name='Category', color_continuous_scale='Plasma')
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Task 2 chart is active only between 6:00 PM and 8:00 PM IST.")

# --- Task 3 (6 PM - 9 PM IST) ---
st.header("Task 3: Time Series Trend with MoM Growth")
if is_active(18, 21):
    mask_t3 = (~df['App'].str.lower().str.startswith(('x', 'y', 'z'))) & (~df['App'].str.contains('s', case=False, na=False)) & (df['Category'].str.upper().str.startswith(('E', 'C', 'B'))) & (df['Reviews'] > 500)
    f3 = df[mask_t3].copy()
    f3['Category'] = f3['Category'].replace({'BEAUTY': 'सौंदर्य', 'BUSINESS': 'வணிகம்', 'DATING': 'Partnersuche'})
    f3['YearMonth'] = f3['Last Updated'].dt.to_period('M').dt.to_timestamp()
    m3 = f3.groupby(['YearMonth', 'Category'])['Installs_Numeric'].sum().reset_index().sort_values('YearMonth')
    m3['MoM_Growth'] = m3.groupby('Category')['Installs_Numeric'].pct_change() * 100
    fig3 = go.Figure()
    for cat, grp in m3.groupby('Category'):
        fig3.add_trace(go.Scatter(x=grp['YearMonth'], y=grp['Installs_Numeric'], mode='lines+markers', name=cat))
        for _, row in grp[grp['MoM_Growth'] > 20].iterrows():
            fig3.add_vrect(x0=row['YearMonth'] - pd.DateOffset(days=14), x1=row['YearMonth'] + pd.DateOffset(days=14), fillcolor="rgba(46, 204, 113, 0.25)", layer="below", line_width=0)
    fig3.update_layout(title="Total Installs Trend Over Time", xaxis_title="Date", yaxis_title="Total Installs")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Task 3 chart is active only between 6:00 PM and 9:00 PM IST.")

# --- Task 4 (4 PM - 6 PM IST) ---
st.header("Task 4: Cumulative Installs (Stacked Area)")
if is_active(16, 18):
    mask_t4 = (df['Rating'] >= 4.2) & (~df['App'].str.contains(r'\d', na=False)) & (df['Category'].str.upper().str.startswith(('T', 'P'))) & (df['Reviews'] > 1000) & (df['Size_MB'].between(20, 80))
    f4 = df[mask_t4].copy()
    f4['Category'] = f4['Category'].replace({'TRAVEL_AND_LOCAL': 'Voyages et destinations', 'PRODUCTIVITY': 'Productividad', 'PHOTOGRAPHY': '写真'})
    f4['YearMonth'] = f4['Last Updated'].dt.to_period('M').dt.to_timestamp()
    m4 = f4.groupby(['YearMonth', 'Category'])['Installs_Numeric'].sum().unstack(fill_value=0).cumsum().stack().reset_index(name='Cumulative_Installs')
    fig4 = px.area(m4, x='YearMonth', y='Cumulative_Installs', color='Category')
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Task 4 chart is active only between 4:00 PM and 6:00 PM IST.")

# --- Task 5 (3 PM - 5 PM IST) ---
st.header("Task 5: Grouped Bar Chart (Avg Rating vs Reviews)")
if is_active(15, 17):
    mask_t5 = (df['Size_MB'] >= 10) & (df['Last Updated'].dt.month == 1)
    f5 = df[mask_t5].copy()
    stats5 = f5.groupby('Category').agg(Avg_Rating=('Rating', 'mean'), Total_Reviews=('Reviews', 'sum'), Total_Installs=('Installs_Numeric', 'sum')).reset_index()
    top10 = stats5[stats5['Avg_Rating'] >= 4.0].nlargest(10, 'Total_Installs')
    fig5 = go.Figure(data=[
        go.Bar(name='Average Rating', x=top10['Category'], y=top10['Avg_Rating'], yaxis='y1', marker_color='#2980b9'),
        go.Bar(name='Total Reviews', x=top10['Category'], y=top10['Total_Reviews'], yaxis='y2', marker_color='#e67e22')
    ])
    fig5.update_layout(barmode='group', yaxis=dict(title="Average Rating", range=[0, 5]), yaxis2=dict(title="Total Reviews", overlaying='y', side='right'))
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Task 5 chart is active only between 3:00 PM and 5:00 PM IST.")

# --- Task 6 (1 PM - 2 PM IST) ---
st.header("Task 6: Dual-Axis Comparison (Free vs Paid)")
if is_active(13, 14):
    def parse_ver(v):
        try: return float('.'.join(str(v).split(' ')[0].split('W')[0].split('.')[:2]))
        except: return np.nan
    df['Android_Ver_Num'] = df['Android Ver'].apply(parse_ver)
    mask_t6 = (df['Size_MB'] > 15) & (df['Content Rating'] == 'Everyone') & (df['App'].str.len() <= 30) & (df['Android_Ver_Num'] > 4.0) & (df['Installs_Numeric'] >= 10000)
    f6 = df[mask_t6].copy()
    top3 = f6.groupby('Category')['Installs_Numeric'].sum().nlargest(3).index
    f6 = f6[f6['Category'].isin(top3)]
    s6 = f6.groupby(['Category', 'Type']).agg(Avg_Installs=('Installs_Numeric', 'mean'), Avg_Revenue=('Revenue', 'mean')).reset_index()
    s6['Category_Type'] = s6['Category'] + " (" + s6['Type'] + ")"
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(x=s6['Category_Type'], y=s6['Avg_Installs'], name='Avg Installs', marker_color='#3498db', yaxis='y1'))
    fig6.add_trace(go.Scatter(x=s6['Category_Type'], y=s6['Avg_Revenue'], name='Avg Revenue ($)', mode='lines+markers', marker_color='#e74c3c', yaxis='y2'))
    fig6.update_layout(yaxis=dict(title="Average Installs"), yaxis2=dict(title="Average Revenue ($)", overlaying='y', side='right'))
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("Task 6 chart is active only between 1:00 PM and 2:00 PM IST.")
