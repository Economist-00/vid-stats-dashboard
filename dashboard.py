#import 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px 
import streamlit as st
from datetime import datetime

# functions
def style_negative(v, props=''):
    """Style negative values in dataframe"""
    try:
        return props if v < 0 else None
    except:
        pass

def style_positive(v, props=''):
    """Style positive values in dataframe"""
    try:
        return props if v > 0 else None
    except:
        pass

def audience_simple(country):
    """Show top countries"""
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'
    
#load data
@st.cache_data
def load_data():
    df_ag = pd.read_csv("video-data/Aggregated_Metrics_By_Video.csv", encoding='utf-8').iloc[1:, :]
    df_ag_sub = pd.read_csv("video-data/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv")
    df_comments = pd.read_csv("video-data/All_Comments_Final.csv")
    df_time = pd.read_csv("video-data/Video_Performance_Over_Time.csv")
    df_ag.columns = ['Video', 'Video title', 'Video publish time', 'Comments added', 'Shares', 'Dislikes', 'Likes', 'Subscribers lost', 'Subscribers gained', 
                    'RPM(USD)', 'CPM(USD)', 'Average percentage viewed(%)', 'Average view duration', 'Views', 'Watch time(hours)', 'Subscribers', 
                    'Your estimated revenue(USD)', 'Impressions', 'Impressions click-through rate(%)']
    df_ag['Video publish time'] = pd.to_datetime(df_ag['Video publish time'], format='mixed')
    df_ag['Average view duration'] = df_ag['Average view duration'].apply(lambda x: datetime.strptime(x, '%H:%M:%S'))
    df_ag['Avg_duration_sec'] = df_ag['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)
    df_ag['Engagement_ratio'] = (df_ag['Comments added'] + df_ag['Shares'] + df_ag['Dislikes'] + df_ag['Likes']) / df_ag['Views']
    df_ag['Views / sub_gained'] = df_ag['Views'] / df_ag['Subscribers gained']
    df_ag.sort_values('Video publish time', ascending=False, inplace=True)
    df_time['Date'] = pd.to_datetime(df_time['Date'], format='mixed')
    return df_ag, df_ag_sub, df_comments, df_time

df_ag, df_ag_sub, df_comments, df_time = load_data()

#data engineering
df_ag_diff = df_ag.copy()
metric_date_12mo = df_ag_diff['Video publish time'].max() - pd.DateOffset(months=12)
numeric_cols = np.array((df_ag_diff.dtypes == 'float64') | (df_ag_diff.dtypes == 'int64'))
median_ag = df_ag_diff[df_ag_diff['Video publish time'] >= metric_date_12mo].iloc[:, numeric_cols].median()
df_ag_diff.iloc[:, numeric_cols] = (df_ag_diff.iloc[:, numeric_cols] - median_ag).div(median_ag)

#merge daily data with publish data to get delta
df_time_diff = pd.merge(df_time, df_ag.loc[:, ['Video','Video publish time']], left_on='External Video ID', right_on='Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

#get last 12 months of data
date_12mo = df_ag['Video publish time'].max() - pd.DateOffset(months=12)
try:
    df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]
except:
    pass

#get daily view data
views_days = pd.pivot_table(df_time_diff_yr, index='days_published', values='Views', aggfunc=[np.mean, np.median, lambda x: np.percentile(x, 80), lambda x: np.percentile(x, 20)]).reset_index()
views_days.columns = ['days_published', 'mean_views', 'median_views', '80pct_views', '20pct_views']
views_days = views_days[views_days['days_published'].between(0,30)]
views_cumulative = views_days.loc[:, ['days_published', 'median_views', '80pct_views', '20pct_views']]
views_cumulative.loc[:,['median_views', '80pct_views', '20pct_views']] = views_cumulative.loc[:,['median_views', '80pct_views', '20pct_views']].cumsum()

#adding widgets to dashboard
add_sidebar = st.sidebar.selectbox('Aggregate or Individual Video', ('Aggregate Metrics', 'Individual Video Analysis'))

if add_sidebar == 'Aggregate Metrics':
    df_ag_metrics = df_ag[['Video publish time', 'Views', 'Likes', 'Subscribers', 'Shares', 'Comments added', 'RPM(USD)', 
                            'Average percentage viewed(%)', 'Avg_duration_sec', 'Engagement_ratio', 'Views / sub_gained']]
    metric_date_6mo = df_ag_metrics['Video publish time'].max() - pd.DateOffset(months=6)
    metric_date_12mo = df_ag_metrics['Video publish time'].max() - pd.DateOffset(months=12)
    numeric_cols_metrics = np.array((df_ag_metrics.dtypes == 'float64') | (df_ag_metrics.dtypes == 'int64'))
    metric_medians6mo = df_ag_metrics[df_ag_metrics['Video publish time'] >= metric_date_6mo].iloc[:, numeric_cols_metrics].median()
    metric_medians12mo = df_ag_metrics[df_ag_metrics['Video publish time'] >= metric_date_12mo].iloc[:, numeric_cols_metrics].median()

    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]

    count = 0
    for i in metric_medians6mo.index:
        with columns[count]:
            delta = (metric_medians6mo[i] - metric_medians12mo[i])/metric_medians12mo[i]
            st.metric(label=i, value=round(metric_medians6mo[i], 1), delta="{:.2%}".format(delta))
            count += 1
            if count >= 5:
                count = 0
    
    df_ag_diff['Publish_date'] = df_ag_diff['Video publish time'].apply(lambda x: x.date())
    df_ag_diff_final = df_ag_diff.loc[:, ['Video title', 'Publish_date', 'Views', 'Likes', 'Subscribers', 'Shares','Comments added', 'RPM(USD)', 'Average percentage viewed(%)',  
                                          'Avg_duration_sec', 'Engagement_ratio', 'Views / sub_gained']]
    
    numeric_cols_metrics_fin = np.array((df_ag_diff_final.dtypes == 'float64') | (df_ag_diff_final.dtypes == 'int64'))
    
    df_ag_numeric_list = df_ag_diff_final.iloc[:, numeric_cols_metrics_fin].median().index.to_list()
    df_to_pct = {}
    for i in df_ag_numeric_list:
        df_to_pct[i] = '{:.1%}'.format

    st.dataframe(df_ag_diff_final.style.hide().applymap(style_negative, props='color:red;').applymap(style_positive, props='color:green;').format(df_to_pct))

if add_sidebar == 'Individual Video Analysis':
    videos = tuple(df_ag['Video title'])
    video_select = st.selectbox('Pick a video:', videos)

    ag_filtered = df_ag[df_ag['Video title'] == video_select]
    ag_sub_filtered = df_ag_sub[df_ag_sub['Video Title'] == video_select]
    ag_sub_filtered['Country'] = ag_sub_filtered['Country Code'].apply(audience_simple)
    ag_sub_filtered.sort_values('Is Subscribed', inplace=True)

    fig = px.bar(ag_sub_filtered, x='Views', y='Is Subscribed', color='Country', orientation='h')
    st.plotly_chart(fig)

    ag_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
    first_30 = ag_time_filtered[ag_time_filtered['days_published'].between(0,30)]
    first_30 = first_30.sort_values('days_published')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'], mode='lines',
                              name='20th_percentile', line=dict(color='purple', dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'], mode='lines+markers',
                              name='50th_percentile', line=dict(color='black', dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'], mode='lines',
                              name='80th_percentile', line=dict(color='royalblue', dash='dash')))
    fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(), mode='lines',
                              name='current video', line=dict(color='firebrick', width=8)))
    fig2.update_layout(title='View Comparison first 30 days',
                       xaxis_title='Days since published',
                       yaxis_title='Cumulative views')
    st.plotly_chart(fig2)