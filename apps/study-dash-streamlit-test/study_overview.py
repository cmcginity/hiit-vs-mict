### Imports and environment
# Imports
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Environment
pfileabs = Path(__file__).resolve()
pchk = pfileabs
chk = 0
while not pchk.name == 'hiit-vs-mict' or chk >= 10:
    pchk = pchk.parent
    chk += 1

proot = pchk.parent
pdata = os.path.join(proot, 'data')
pdata_myphd = os.path.join(pdata, 'myphd')
ptest = os.path.join(pdata_myphd,'_processed','sourcetype_device','WearableFitbit-Fitbit','006_qtz1b13893369732763681_hr_WearableFitbit_Fitbit.csv')


### Functions and Parameters
@st.cache_resource
def load_data(path):
    df = pd.read_csv(path)
    df = df[df['Value'] >= df['target_hr_45']]
    return df

@st.cache_data
def split_dataframe_by_time_gap(df, time_field='_time', gap=pd.Timedelta(minutes=5)):
    """
    Splits the dataframe into separate dataframes whenever a gap greater than the given threshold is detected.

    Args:
    - df (pd.DataFrame): Input dataframe sorted on time_field.
    - time_field (str): Name of the time column in the dataframe.
    - gap (pd.Timedelta): Time gap to detect separate events.

    Returns:
    - List[pd.DataFrame]: A list of dataframes for separate events.
    """

    # Ensure the time field is in datetime format
    df[time_field] = pd.to_datetime(df[time_field])

    # Identify the indices where the difference between consecutive rows is greater than the given gap
    gap_indices = df[df[time_field].diff().gt(gap)].index.tolist()

    # Add the start and end indices for easier splitting
    start_indices = [df.index[0]] + gap_indices
    end_indices = gap_indices + [df.index[-1] + 1]

    dataframes = []
    for start, end in zip(start_indices, end_indices):
        dfchk = df.loc[start:end-1]
        hr_max = dfchk['Value'].max()
        if hr_max >= dfchk['target_hr_70'].iloc[0]:
            dataframes.append(df.loc[start:end-1])
        
    return dataframes


### Load data
dfwo = load_data(ptest)
# dfwo = df[df['Value'] >= df['target_hr_45']]

### Identify workouts
dfwos = split_dataframe_by_time_gap(dfwo,'_time',gap=pd.Timedelta(minutes=5))

ppt_id = dfwo["ppt_id"].apply(lambda x: f"{x:03}").iloc[0]



### ### ### ### ### ### Dashboard ### ### ### ### ### ###
st.title(f'🏃‍♂️ HIIT-vs-MICT Project 🏃‍♀️')
# st.markdown(f'path root: {proot}')
# st.markdown(f'path pchk: {pchk}')
# st.markdown(f'path myphd: {pdata_myphd}') 
# st.markdown(f'path file absolute: {pfileabs}') 
# st.write('Here\'s some text and stuff:')
st.markdown(f'## Participant: {ppt_id}')

st.metric(label='🏃‍♂️🏃‍♀️Workouts',value=len(dfwos),delta=None)
# Slider to select the index
selected_index = st.slider("Select Workout:", 0, len(dfwos) - 1)
dfselected = dfwos[selected_index]

st_wo = dfselected['_time'].min()
et_wo = dfselected['_time'].max()
dur_wo = et_wo - st_wo
dur_wo_min1 = round(dur_wo.total_seconds() / 60,1)
dow_a = dfselected['dow_abbr'].iloc[0]

hr_min = dfselected['Value'].min()
hr_max = dfselected['Value'].max()

col11, col12, col13 = st.columns(3)
# col12.metric("⏰ Start", dfselected['_time'].min().strftime('%Y-%m-%d %H:%M:%S'), None)
# col13.metric("End ⏰", dfselected['_time'].max().strftime('%Y-%m-%d %H:%M:%S'), None)

col11.metric(f"📆 Weekday",f"{dow_a}",None)
col12.metric(f"⏰ Start", f"{st_wo.strftime('%H:%M:%S')}")
col13.metric(f"⏰ Stop", f"{et_wo.strftime('%H:%M:%S')}")
# col12.markdown(f"⏰ Start: {st_wo.strftime('%Y-%m-%d %H:%M:%S')}")
# col13.markdown(f"⏰ End: {et_wo.strftime('%Y-%m-%d %H:%M:%S')}")
# col13.markdown(f":stopwatch: Duration: {dur_wo_min1} min")


col21, col22, col23= st.columns(3)
col21.metric(f":stopwatch: Duration", f"{dur_wo_min1} min")
col22.metric(':green_heart: $HR_{min}$(bpm)', f'{hr_min}', None)
col23.metric('❤️‍🔥 $HR_{max}$(bpm)', hr_max, None)

st.line_chart(dfselected[['_time','Value', 'target_hr_70', 'target_hr_90']],
              x='_time',
              y=['Value', 'target_hr_70', 'target_hr_90'],
              color=["#0668C9","#83C9FF","#F92B2C"])
            #   color=["#0668C9","#83C9FF","#F92B2C"])
            #   color=["#FF5722","#FFC3A0","#FF6B6B"])
            #   color=["#3EC1D3","#FF9A00","#FF165D"])
            #   color=["#035e7b","#fce38a","#f38181"])
            #   color=['#007BFF', '#FFA500', '#FF4136'])
            #   ["95e1d3","519872","fce38a","f38181","035e7b"]
            
st.write(dfwo.head(100))