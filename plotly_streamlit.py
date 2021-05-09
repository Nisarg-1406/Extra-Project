# -*- coding: utf-8 -*-
"""
Created on Sat May  8 14:50:04 2021

@author: NISARG MEHTA
"""

import streamlit as st
import pandas as pd
import plotly.figure_factory as ff

st.title('EMDAT Dataset Graphs Using Streamlit And Plotly')

@st.cache
def load_data(nrows):
    data = pd.read_csv(r"C:\Users\NISARG MEHTA\Downloads\emdat_cleaned_data.csv",nrows=nrows)
    return data

weekly_data = load_data(1000)

#WeeklyDemand Data
st.subheader('Weekly Demand Data')
st.write(weekly_data)

st.bar_chart(weekly_data['Total_Deaths'])
df = pd.DataFrame(weekly_data[:200], columns = ['Total_Deaths','No_Affected','Total_Affected','Total_Damages'])
df.hist()
st.pyplot()

st.line_chart(df)

chart_data = pd.DataFrame(weekly_data[:100], columns=['Total_Deaths', 'Total_Damages'])
st.area_chart(chart_data)

# st.bar_chart(center_info_data['region_code'])
# st.bar_chart(center_info_data['center_type'])

hist_data = [weekly_data['Total_Deaths'],weekly_data['Total_Damages']]
group_labels = ['Center Id', 'Region Code']
fig = ff.create_distplot(hist_data, group_labels, bin_size=[10, 25])
st.plotly_chart(fig, use_container_width=True)