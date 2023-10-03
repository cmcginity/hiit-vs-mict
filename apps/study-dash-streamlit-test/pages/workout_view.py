### Imports and environment
# Imports
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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
pdata_fitbit = os.path.join(pdata_myphd,'_processed','sourcetype_device','WearableFitbit-Fitbit')
ptest = os.path.join(pdata_myphd,'_processed','sourcetype_device','WearableFitbit-Fitbit','006_qtz1b13893369732763681_hr_WearableFitbit_Fitbit.csv')


# Google Drive API Initialization
def setup_drive():
    SERVICE_ACCOUNT_FILE = '/Users/curtismcginity/Downloads/hiit-vs-mict-50dbc00f450b.json'
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

drive_service = setup_drive()
            


### Functions and Parameters
@st.cache_resource
def load_data(path):
    df = pd.read_csv(path)
    df = df[df['Value'] >= 0.8 * df['target_hr_45']]
    return df

@st.cache_data
def get_dfmetadata(_drive_service, folder_id=None):
    pathfitbit = st.secrets["gdrive_id_myphd__processed_hr_fitbit"]
    dfmetadata_list = []

    try:
        # If folder_id is provided, list files from that folder. Otherwise, list from entire Drive.
        if folder_id:
            response = drive_service.files().list(corpora='drive', 
                                                driveId=st.secrets["gdrive_id_root"],
                                                q=f"'{pathfitbit}' in parents",
                                                includeItemsFromAllDrives=True, 
                                                supportsAllDrives=True).execute()
        else:
            response = drive_service.files().list(corpora='drive', 
                                                driveId=st.secrets["gdrive_id_root"],
                                                includeItemsFromAllDrives=True, 
                                                supportsAllDrives=True).execute()
        
        for file in response.get('files', []):
            fname = file.get('name')
            if fname.startswith("0"):
                meta = fname.split("_")
                metadata = {
                    'ppt_id': meta[0],
                    'myphd_id': meta[1],
                    'datatype': meta[2],
                    'sourcetype': meta[3],
                    'device': meta[4],
                    'fname': fname
                }
                dfmetadata_list.append(metadata)
                
    except HttpError as error:
        print(f"An error occurred: {error}")

    dfmetadata = pd.DataFrame(dfmetadata_list)
    return dfmetadata
# def get_dfmetadata(path_to_data):
#     dfmetadata_list = []
#     # dfmetadata = pd.DataFrame(columns=['ppt_id', 'myphd_id', 'datatype', 'sourcetype', 'device', 'fname'])
#     # st.write(f'dfmetadata {dfmetadata}')
#     # st.write(f'pandas {pd.__version__}')
#     # st.write(f'streamlit {st.__version__}')
#     for subdir, dirs, files in os.walk(pdata_fitbit):
#         for f in files:
#             if f.startswith("0"):
#                 meta = f.split("_")

#                 # fpath = os.path.join(subdir, f)
#                 # df = pd.read_csv(fpath)
#                 # df = df[df['Value'] >= df['target_hr_45']]

#                 metadata = {
#                     'ppt_id': meta[0],
#                     'myphd_id': meta[1],
#                     'datatype': meta[2],
#                     'sourcetype': meta[3],
#                     'device': meta[4],
#                     'fname': f
#                     # 'df': df
#                 }
#                 dfmetadata_list.append(metadata)
#     dfmetadata = pd.DataFrame(dfmetadata_list)
#     return dfmetadata

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
        hr_max = dfchk['Value'].max()
        if hr_max >= dfchk['target_hr_70'].iloc[0]:
            dataframes.append(df.loc[start:end-1])
        
    return dataframes

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







### ### ### ### ### ### Dashboard ### ### ### ### ### ###
# st.title(f'🏃‍♂️ HIIT-vs-MICT Project 🏃‍♀️')
st.markdown(f'# Workout View 🏃‍♂️🔬👩‍🔬')
# st.markdown(f'path root: {proot}')
# st.markdown(f'path pchk: {pchk}')
# st.markdown(f'path myphd: {pdata_myphd}') 
# st.markdown(f'path file absolute: {pfileabs}') 
# st.write('Here\'s some text and stuff:')

### Load metadata
pathfitbit = st.secrets["gdrive_id_myphd__processed_hr_fitbit"]
# qstring = f"'{pathfitbit}' in parents"
# fitbitfiles = drive_service.files().list(corpora='drive', 
#                                       driveId=st.secrets["gdrive_id_root"],
#                                       q=qstring,
#                                       includeItemsFromAllDrives=True, 
#                                       supportsAllDrives=True).execute().get('files',[])
# st.write(f'q string: {qstring}')
# st.write(f'drive service: {fitbitfiles}')
# st.write(f'secret: {pathfitbit}')
dfmetadata = get_dfmetadata(drive_service, pathfitbit)
# dfmetadata = get_dfmetadata(pdata_fitbit)
print(dfmetadata)
ppt_list = dfmetadata['ppt_id'].sort_values()
selected_ppt = st.selectbox(
    'Select a participant to review.',
    ppt_list
)

### Load data
pdata_fitbit_file = os.path.join(pdata_fitbit,dfmetadata['fname'][dfmetadata['ppt_id'] == selected_ppt].iloc[0])
dfwo = load_data(pdata_fitbit_file)
# dfwo = df[df['Value'] >= df['target_hr_45']]

### Identify workouts
dfwos = split_dataframe_by_time_gap(dfwo,'_time',gap=pd.Timedelta(minutes=5))

# ppt_id = dfwo["ppt_id"].apply(lambda x: f"{x:03}").iloc[0]


st.markdown(f'## Participant: {selected_ppt}')
col_ppt1, col_ppt2, col_ppt3 = st.columns(3)

col_ppt1.metric(label='🏃‍♂️🏃‍♀️Workouts',value=len(dfwos),delta=None)
col_ppt2.metric('🎟 Status',enrollment_status(dfwo['enrollment_status'].iloc[0]),None)
col_ppt3.metric(label=f'{cohort1} Cohort',value=rand_group(dfwo['randomization_group'].iloc[0]),delta=None)
# Slider to select the index
if len(dfwos) <= 0:
    st.markdown('## No qualifying workouts! 💩')
    st.stop()
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
            
st.write(dfselected.head(100))