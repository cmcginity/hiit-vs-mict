### Imports and environment
# Imports
# import os
# import io
import streamlit as st
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# from google.oauth2.service_account import Credentials
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from googleapiclient.http import MediaIoBaseDownload
# import requests
import custom_utility.api_data_utils as apihelper




# @st.cache_data
def get_data_from_preloaded(file_id):
    return preloaded_data.get(file_id)

### Functions and Parameters
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
    # If df is empty, return an empty list
    if df.empty:
        return []
    
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
        hr_max = dfchk['value'].max()
        if hr_max >= dfchk['target_hr_70'].iloc[0]:
            dataframes.append(df.loc[start:end-1])
        
    return dataframes

# @st.cache_data
def split_dataframe_by_wk_wo(df):
    # Group by wk_id and wo_id and then create a list of dataframes
    grouped = df.groupby(['wk_id', 'wo_id'])
    df_list = [group for _, group in grouped]
    return df_list

def clean_int(df,list_int):
    for x in list_int:
        df[x] = df[x].astype('Int64')
    return df

def clean_float(df,list_float):
    for x in list_float:
        df[x] = df[x].astype('Float64')
    return df

def clean_str(df,list_str):
    for x in list_str:
        df[x] = df[x].astype('string')
    return df

def parse_time(df, time_col):
    """
    Parse the given time column in the DataFrame to extract:
    - Day of the week in numerical form (0 = Monday, 6 = Sunday)
    - Day of the week in abbreviated form (e.g., 'Mon', 'Tue')
    - Month of the year in numerical form (01 = January, 12 = December)
    - Month of the year in abbreviated form (e.g., 'Jan', 'Feb')

    Args:
    - df (pd.DataFrame): The input DataFrame
    - time_col (str): The column containing datetime64 values

    Returns:
    - pd.DataFrame: DataFrame with added columns
    """

    # Extract day of the week
    df['dow'] = df[time_col].dt.dayofweek
    df['dow_abbr'] = df[time_col].dt.strftime('%a')

    # Extract month of the year
    df['moy'] = df[time_col].dt.strftime('%m')
    df['moy_abbr'] = df[time_col].dt.strftime('%b')

    return df


def clean_ppt_df(df):
    df['ppt_id'] = df['record_id'].apply(lambda x: f'{x:03}')
    df = df.drop('redcap_event_name',axis=1)
    # dfppt['myphd_date_shift'] = dfppt['myphd_date_shift'].fillna(0)
    df_str = [
        'myphd_id'
    ]
    df_int = [
        'ppt_id',
        'record_id',
        'enrollment_status',
        'randomization_group',
        'myphd_date_shift'
    ]
    df = clean_int(df,df_int)
    df = clean_str(df,df_str)
    return df

def merge_rc(df,dfppt):
    df = df.merge(dfppt, on='ppt_id', how='left')
    return df

def enrollment_status(esnum):
    esnum = str(esnum)
    enrollcode = {
        '1': 'Prescreened',
        '2': 'Screened',
        '3': 'Qualified',
        '4': 'Ongoing',
        '5': 'Completed',
        '6': 'Withdrawn',
        '7': 'Lost',
        '8': 'Unconsenting'
    }
    enrollcode2 = {
        '1': '👋Prescreened',
        '2': '🔎Screened',
        '3': '👍Qualified',
        '4': '🏃‍♀️Ongoing',
        '5': '🏅Completed',
        '6': '❤️‍🩹Withdrawn',
        '7': '👻Lost',
        '8': '🛑Unconsenting'
    }
    return enrollcode2[esnum]

def rand_group(rgnum):
    rgnum = str(rgnum)
    groupcode = {
        '1': 'HIIT',
        '2': 'MICT',
        '3': 'Control'
    }
    groupcode2 = {
        '1': '🏃‍♂️ HIIT',
        '2': '🚶‍♂️ MICT',
        '3': '🧍‍♂️ CTL'
    }
    groupcode3 = {
        '1': '⚡ HIIT',
        '2': '🔥 MICT',
        '3': '❄ CTL'
    }
    return groupcode3[rgnum]

cohort1 = '🏃‍♂️🚶‍♂️🧍‍♂️'
cohort2 = '❤️🧡💙'


### Initialize APIs and load data
# Google Drive API Initialization
drive_service = apihelper.setup_drive()
pworkout_fname_id = apihelper.get_file_ids_from_dir(st.secrets['gdrive_id_workout'])
# st.write(pworkout_fname_id["012_qtz1b17269433695528197_workout_allevents.csv"])

# Use the preload function to load data
preloaded_data = apihelper.preload_data_from_drive(st.secrets["gdrive_id_workout"])

# Load metadata
pworkout = st.secrets["gdrive_id_workout"]
dfmetadata = apihelper.get_dfmetadata(drive_service, pworkout)

# Load PPT data
dfppt = apihelper.read_redcap_report(st.secrets['redcap']['api_url'],st.secrets['redcap']['api_key_curtis'],st.secrets['redcap']['ppt_meta_master_id'])
dfppt = clean_ppt_df(dfppt)



### ### ### ### ### ### Dashboard ### ### ### ### ### ###
# st.title(f'🏃‍♂️ HIIT-vs-MICT Project 🏃‍♀️')
st.markdown(f'# Adherence View 🏃‍♂️🔬👩‍🔬')

### Select a PPT
ppt_list = dfmetadata['ppt_id'].sort_values()
selected_ppt = st.selectbox(
    'Select a participant to review.',
    ppt_list
)

### Load PPT workout data
dfwo = preloaded_data.get(pworkout_fname_id[dfmetadata['fname'][dfmetadata['ppt_id'] == selected_ppt].iloc[0]])

### Merge data
dfwo['_time'] = pd.to_datetime(dfwo['_time'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
dfwo['_realtime'] = pd.to_datetime(dfwo['_realtime'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
dfwo = parse_time(dfwo,'_realtime')
dfwo = merge_rc(dfwo,dfppt)
### Identify workouts
# dfwos = split_dataframe_by_time_gap(dfwo,'_time',gap=pd.Timedelta(minutes=5))
dfwos = split_dataframe_by_wk_wo(dfwo)
### 
# ppt_id = dfwo["ppt_id"].apply(lambda x: f"{x:03}").iloc[0]





st.markdown(f'## Participant: {selected_ppt}')
col_ppt1, col_ppt2, col_ppt3 = st.columns(3)

col_ppt1.metric(label='🏃‍♂️🏃‍♀️Workouts',value=len(dfwos),delta=None)
col_ppt2.metric('🎟 Status',enrollment_status(dfwo['enrollment_status'].iloc[0]),None)
col_ppt3.metric(label=f'{cohort1} Cohort',value=rand_group(dfwo['randomization_group'].iloc[0]),delta=None)


# All workouts
def preprocess(df):
    # Ensure datetime format
    df['_realtime'] = pd.to_datetime(df['_realtime'])

    # Focus on polar data
    df = df.loc[df['device'] == "Polar"]
    

    # Compute workout start time
    df['_wotime_base'] = df.groupby(['wk_id', 'wo_id'])['_realtime'].transform('min')

    # Calculate a new field _wotime as difference in a resolution (e.g., minutes) from _wotime_base
    df['_wotime'] = (df.loc[:,'_realtime'] - df.loc[:,'_wotime_base']).dt.total_seconds() / 60  # Here, we'll show time in minutes

    # Give a time 

    return df

def plot_workouts(df):
    # ppt = df.loc[0,'ppt_id']
    ppt = df['ppt_id'].apply(lambda x: f'{x:03}')
    ppt = ppt.iloc[0]
    workouts = df.groupby(['wk_id', 'wo_id'])

    # Setup the color map
    n_colors = workouts.ngroups
    cmap = plt.get_cmap('viridis', n_colors) 

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, ((wk_id, wo_id), wo) in enumerate(workouts):
        ax.plot(wo['_wotime'], wo['value'], label=f'{i:02}: Week {wk_id}, Workout {wo_id}', color=cmap(i), alpha=0.8)
    
    ax.set_xlabel('Workout Duration (Minutes)')
    ax.set_ylabel('HR')
    ax.set_title(f'{ppt}: Workout Performance Over Time')
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    fig.tight_layout()
    st.pyplot(fig)
dfwodur = preprocess(dfwo)
plot_workouts(dfwodur)



# Slider to select the workout
if len(dfwos) <= 0:
    st.markdown('## No qualifying workouts! 💩')
    st.stop()
selected_index = st.slider("Select Workout:", 0, len(dfwos) - 1)
dfselected = dfwos[selected_index]
dfselected['_realtime'] = pd.to_datetime(dfselected['_realtime'])

# Convert device column to lowercase
dfselected['device'] = dfselected['device'].str.lower()

# Extract KPIs for presentation
st_wo = dfselected['_realtime'].min()
et_wo = dfselected['_realtime'].max()
dur_wo = et_wo - st_wo
dur_wo_min1 = round(dur_wo.total_seconds() / 60,1)
dow_a = dfselected['dow_abbr'].iloc[0]

hr_min = dfselected['value'].min()
hr_max = dfselected['value'].max()


col11, col12, col13 = st.columns(3)
col11.metric(f"📆 Weekday",f"{dow_a}",None)
col12.metric(f"⏰ Start", f"{st_wo.strftime('%H:%M:%S')}")
col13.metric(f"⏰ Stop", f"{et_wo.strftime('%H:%M:%S')}")

col21, col22, col23= st.columns(3)
col21.metric(f":stopwatch: Duration", f"{dur_wo_min1} min")
col22.metric(':green_heart: $HR_{min}$(bpm)', f'{hr_min}', None)
col23.metric('❤️‍🔥 $HR_{max}$(bpm)', hr_max, None)

col31, col32, col33= st.columns(3)
timescale = col31.selectbox(
    "Timescale:",
    ["1S","2S","5S","10S"]
)

### Group by 5-second intervals and the device, then calculate the aggregations
result = (dfselected
          .groupby([pd.Grouper(key='_realtime', freq=timescale), 'device'])
          .agg(value_avg=('value', 'mean'),
               value_med=('value', 'median'),
               value_min=('value', 'min'),
               value_max=('value', 'max'),
               value_ct=('value', 'size'))
          ).reset_index()
result = result.pivot(index='_realtime', columns='device').sort_index(axis=1,level=1)

# Rename columns to match the desired format {device}_{agg}
result.columns = [f'{agg}_{device}' for agg, device in result.columns]
result.reset_index(inplace=True)  # Resetting the index after the pivot to make _realtime a column again

col_list = ['_realtime'] + [f'value_med_{col}' for col in dfselected['device'].unique()]
result = result[col_list]
result['target_hr_45'] = dfppt[dfppt['ppt_id'] == int(selected_ppt)]['target_hr_45'].iloc[0]
result['target_hr_55'] = dfppt[dfppt['ppt_id'] == int(selected_ppt)]['target_hr_55'].iloc[0]
result['target_hr_70'] = dfppt[dfppt['ppt_id'] == int(selected_ppt)]['target_hr_70'].iloc[0]
result['target_hr_90'] = dfppt[dfppt['ppt_id'] == int(selected_ppt)]['target_hr_90'].iloc[0]
# result.rename(columns={'value_med_fitbit': 'fitbit','value_med_polar': 'polar'}, inplace=True)


### Plotting
def get_plot_fields(x):
    if x == 1:
        return ['_time', 'value', 'target_hr_70', 'target_hr_90']
    elif x == 2:
        return ['_time', 'value', 'target_hr_45', 'target_hr_55']
    else:
        return ['_time', 'value']

def get_target_hr_list(x):
    if x == 1:
        return ['target_hr_70', 'target_hr_90']
    elif x == 2:
        return ['target_hr_45', 'target_hr_55']
    else:
        return None

randgroup = dfppt[dfppt['ppt_id'] == int(selected_ppt)]['randomization_group'].iloc[0]
plot_fields = get_plot_fields(dfwo['randomization_group'].iloc[0])
ylist = col_list[1:]+get_target_hr_list(randgroup)
st.line_chart(result,
              x=col_list[0],
              y=ylist,
              color=["#83C9FF","#F92B2C","#F99417","#0668C9"][0:len(ylist)])
            #   color=["#0668C9","#83C9FF","#F92B2C"])
            #   color=["#FF5722","#FFC3A0","#FF6B6B"])
            #   color=["#3EC1D3","#FF9A00","#FF165D"])
            #   color=["#035e7b","#fce38a","#f38181"])
            #   color=['#007BFF', '#FFA500', '#FF4136'])
            #   ["95e1d3","519872","fce38a","f38181","035e7b"]



# st.pyplot(fig)
# st.write(dfselected.head(100))


