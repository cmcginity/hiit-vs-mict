### Imports and environment
# Imports
import os
import io
import streamlit as st
# st.write(f"root id: {st.secrets['gdrive_id_root']}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import requests



# Google Drive API Initialization
@st.cache_resource
def setup_drive():
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

drive_service = setup_drive()
# st.write("About to setup drive")
# try:
#     drive_service = setup_drive()
#     st.write(f'drive service {drive_service}')
# except Exception as e:
#     st.write(f'Failed to set up drive service: {e}')

# @st.cache_data 
# def get_file_id_from_name(_drive_service, filename, parent_id):
#     results = _drive_service.files().list(
#         corpora='drive',
#         driveId=st.secrets["gdrive_id_root"],
#         q=f"name='{filename}' and '{parent_id}' in parents",
#         includeItemsFromAllDrives=True,
#         supportsAllDrives=True
#     ).execute()
#     files = results.get('files', [])
#     if not files:
#         raise Exception(f"File {filename} not found!")
#     return files[0]['id']

@st.cache_data
def get_file_ids_from_dir(parent_id):
    drive_service = setup_drive()
    results = drive_service.files().list(
        corpora='drive',
        driveId=st.secrets["gdrive_id_root"],
        q=f"'{parent_id}' in parents",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()
    files = results.get('files', [])
    if not files:
        raise Exception(f"Folder {parent_id} has no files!")
    # id_name = [{x['name'] : x['id']} for x in files]
    id_name = {}
    for x in files:
        id_name[x['name']] = x['id']
    return id_name

pworkout_fname_id = get_file_ids_from_dir(st.secrets['gdrive_id_workout'])
# st.write(pworkout_fname_id["012_qtz1b17269433695528197_workout_allevents.csv"])


def load_data_from_drive(_drive_service, file_id):
    request = _drive_service.files().get_media(fileId=file_id)
    io_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(io_buffer, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    io_buffer.seek(0)
    df = pd.read_csv(io_buffer, parse_dates=['_realtime','_time'])
    # df = df[df['value'] >= 0.8 * df['target_hr_45']]
    return df

# def load_data_from_drive_rc(_drive_service, file_id):
#     request = _drive_service.files().get_media(fileId=file_id)
#     io_buffer = io.BytesIO()
#     downloader = MediaIoBaseDownload(io_buffer, request)
#     done = False
#     while done is False:
#         status, done = downloader.next_chunk()
#     io_buffer.seek(0)
#     df = pd.read_csv(io_buffer)
#     # df = df[df['value'] >= 0.8 * df['target_hr_45']]
#     return df

# st.write("About to fetch")
@st.cache_data
def preload_data_from_drive(parent_id):
    drive_service = setup_drive()
    preloaded_data = {}
    
    # Get list of files in the specified folder
    response = drive_service.files().list(corpora='drive', 
                                          driveId=st.secrets["gdrive_id_root"],
                                          q=f"'{parent_id}' in parents",
                                          includeItemsFromAllDrives=True, 
                                          supportsAllDrives=True).execute()
    # Iterate over each file and load its data
    for file in response.get('files', []):
        if file.get('name').startswith("workout"):
            continue
        file_id = file['id']
        df = load_data_from_drive(drive_service, file_id)
        preloaded_data[file_id] = df
    
    return preloaded_data

# Use the preload function to load data
preloaded_data = preload_data_from_drive(st.secrets["gdrive_id_workout"])

# @st.cache_data
def get_data_from_preloaded(file_id):
    return preloaded_data.get(file_id)

### REDCap data extraction
@st.cache_data
def read_redcap_report(api_url, api_key, report_id):
    """
    Reads a specific report from REDCap into a pandas DataFrame.

    :param api_url: URL to the REDCap API endpoint.
    :param api_key: API key for authentication.
    :param report_id: ID of the report to be fetched.
    :return: DataFrame containing the report data or None if an error occurs.
    """

    # Define the payload for the REDCap API request
    data = {
        'token': api_key,
        'content': 'report',
        'format': 'csv',
        'report_id': report_id,
        'rawOrLabel': 'raw',
        'rawOrLabelHeaders': 'raw',
        'exportCheckboxLabel': 'false',
        'returnFormat': 'csv'
    }

    # Make the POST request to the REDCap API
    response = requests.post(api_url, data=data)

    # Check if the request was successful
    if response.status_code != 200:
        st.error(f"Error fetching data from REDCap: {response.text}")
        return None

    # Convert the CSV response to a pandas DataFrame
    try:
        df = pd.read_csv(io.StringIO(response.text))
        return df
    except pd.errors.ParserError:
        st.error("Error parsing REDCap response as CSV")
        return None

# xxx = read_redcap_report(st.secrets['redcap']['api_url'],st.secrets['redcap']['api_key_curtis'],st.secrets['redcap']['ppt_meta_master_id'])


### Functions and Parameters
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # df = df[df['value'] >= 0.8 * df['target_hr_45']]
    return df

@st.cache_data
def get_dfmetadata(_drive_service, folder_id=None):
    pworkout = st.secrets["gdrive_id_workout"]
    dfmetadata_list = []

    try:
        # If folder_id is provided, list files from that folder. Otherwise, list from entire Drive.
        if folder_id:
            response = drive_service.files().list(corpora='drive', 
                                                driveId=st.secrets["gdrive_id_root"],
                                                q=f"'{pworkout}' in parents",
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
                    # 'datatype': meta[2],
                    # 'sourcetype': meta[3],
                    # 'device': meta[4],
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
#                 # df = df[df['value'] >= df['target_hr_45']]

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

# @st.cache_data
# def get_rc_data(fname):
#     fileid = get_file_id_from_name(drive_service,fname,st.secrets["gdrive_id_rc_keys"])
#     dfppt = load_data_from_drive_rc(drive_service,fileid)
#     dfppt = clean_ppt_df(dfppt)
#     return dfppt

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

def adh_score_mict(wk,group):
    mict_dur = {
        '1': 20,
        '2': 20,
        '3': 30,
        '4': 30,
        '5': 40,
        '6': 40,
        '7': 40,
        '8': 40,
        '9': 40,
        '10': 40,
        '11': 40,
        '12': 40,
        '13': 40,
        '14': 40,
        '15': 40
    }
    






### ### ### ### ### ### Dashboard ### ### ### ### ### ###
# st.title(f'🏃‍♂️ HIIT-vs-MICT Project 🏃‍♀️')
st.markdown(f'# Adherence Dashboard 🏃‍♂️🔬👩‍🔬')
# st.markdown(f'path root: {proot}')
# st.markdown(f'path pchk: {pchk}')
# st.markdown(f'path myphd: {pdata_myphd}') 
# st.markdown(f'path file absolute: {pfileabs}') 
# st.write('Here\'s some text and stuff:')

### Load metadata
pworkout = st.secrets["gdrive_id_workout"]
# qstring = f"'{pworkout}' in parents"
# fitbitfiles = drive_service.files().list(corpora='drive', 
#                                       driveId=st.secrets["gdrive_id_root"],
#                                       q=qstring,
#                                       includeItemsFromAllDrives=True, 
#                                       supportsAllDrives=True).execute().get('files',[])
# st.write(f'q string: {qstring}')
# st.write(f'drive service: {fitbitfiles}')
# st.write(f'secret: {pworkout}')
dfmetadata = get_dfmetadata(drive_service, pworkout)
# dfmetadata = get_dfmetadata(pdata_fitbit)
# print(dfmetadata)
ppt_list = dfmetadata['ppt_id'].sort_values()
selected_ppt = st.selectbox(
    'Select a participant to review.',
    ppt_list
)

dfadh = pd.DataFrame()


### Load data
# dfwo = preloaded_data.get(pworkout_fname_id[dfmetadata['fname'][dfmetadata['ppt_id'] == selected_ppt].iloc[0]])
# dfwo = load_data(pdata_fitbit_file)
# dfwo = df[df['value'] >= df['target_hr_45']]
dfppt = read_redcap_report(st.secrets['redcap']['api_url'],st.secrets['redcap']['api_key_curtis'],st.secrets['redcap']['ppt_meta_master_id'])

dfppt = clean_ppt_df(dfppt)
dfppt.drop('target_hr_35', axis=1, inplace=True)
# st.write(dfppt.dtypes)
# st.write(dfppt)
# st.write(dfppt)

# List of target heart rate columns to aggregate
target_hr_columns = [col for col in dfppt.columns if col.startswith('target_hr_')]

ctr=0
dftest = []
for x in preloaded_data.values():
    # st.write(x)
    if ctr == 0:
        dftest.append(x)
    ctr+=1

for dfwo in dftest:
    ### Merge data
    dfwo['_time'] = pd.to_datetime(dfwo['_time'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
    dfwo['_realtime'] = pd.to_datetime(dfwo['_realtime'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
    dfwo = parse_time(dfwo,'_realtime')
    dfwo = merge_rc(dfwo,dfppt)
    ### Identify workouts
    dfwos = split_dataframe_by_wk_wo(dfwo)
    # st.write(dfwos[22])
    for dfselected in [dfwos[22]]:
        dfselected['_realtime'] = pd.to_datetime(dfselected['_realtime'])
        # Convert device column to lowercase
        dfselected['device'] = dfselected['device'].str.lower()
        st.write(dfselected)
        pptid = dfselected['ppt_id'].iloc[0]
        wkid = dfselected['wk_id'].iloc[0]
        woid = dfselected['wo_id'].iloc[0]
        thr45 = dfselected['target_hr_45'].iloc[0]
        thr55 = dfselected['target_hr_55'].iloc[0]
        thr70 = dfselected['target_hr_70'].iloc[0]
        thr90 = dfselected['target_hr_90'].iloc[0]
        ### Group by 5-second intervals and the device, then calculate the aggregations
        result = (dfselected
                .groupby([pd.Grouper(key='_realtime', freq='5S'), 'device'])
                .agg(value_avg=('value', 'mean'),
                    value_med=('value', 'median'),
                    value_min=('value', 'min'),
                    value_max=('value', 'max'),
                    value_ct=('value', 'size'))
                ).reset_index()
        result = result.pivot(index='_realtime', columns='device').sort_index(axis=1,level=1)
        # Flatten MultiIndex columns and create column names in {device}_{agg} format
        result.columns = [f'{agg}_{device}' for agg, device in result.columns]
        result.reset_index(inplace=True)  # Resetting the index after the pivot to make _realtime a column again
        col_list = ['_realtime'] + [f'value_med_{col}' for col in dfselected['device'].unique()]
        result = result[col_list]
        result['target_hr_45'] = thr45
        result['target_hr_55'] = thr55
        result['target_hr_70'] = thr70
        result['target_hr_90'] = thr90
        result_plot = result
        result['ppt_id'] = pptid
        result['wk_id'] = wkid
        result['wo_id'] = woid
        # st.write('result1:')
        st.write(result)


        # # Group by 5-second intervals and the device, then calculate the aggregations
        # aggregations = {'value': ['mean', 'median', 'min', 'max', 'size']}
        # # Add constant columns with 'first' aggregation to keep their names and values
        # for col in target_hr_columns:
        #     aggregations[col] = 'first'
        # st.write(aggregations)
        # result = (dfselected
        #         .groupby([pd.Grouper(key='_realtime', freq='5S'), 'device'])
        #         .agg(**aggregations)
        #         ).reset_index()

        
        # result = (dfselected
        #         .groupby([pd.Grouper(key='_realtime', freq='5S'), 'device'])
        #         .agg(value_avg=('value', 'mean'),
        #             value_med=('value', 'median'),
        #             value_min=('value', 'min'),
        #             value_max=('value', 'max'),
        #             value_ct=('value', 'size'),
        #             target_hr_45=('target_hr_45','mean'),
        #             target_hr_55=('target_hr_55','mean'),
        #             target_hr_70=('target_hr_70','mean'),
        #             target_hr_90=('target_hr_90','mean'))
        #         ).reset_index()
        # # After grouping, pivot the DataFrame
        # result = result.pivot(index='_realtime', columns='device').sort_index(axis=1, level=1)

        # # Flatten MultiIndex columns and create column names in {device}_{agg} format
        # new_columns = []
        # for col_info in result.columns:
        #     if isinstance(col_info, tuple) and len(col_info) > 1:
        #         # Handling device-value columns
        #         agg, device = col_info
        #         new_columns.append(f'{agg}_{device}')
        #     else:
        #         # Handling constant columns
        #         new_columns.append(col_info[0])
        # result.columns = new_columns

        # result.reset_index(inplace=True)  # Resetting the index after the pivot to make _realtime a column again
        
        # # consolidating columns
        # for col in target_hr_columns:
        #     colpolar = col + '_polar'
        #     colfitbit = col + '_fitbit'
        #     result[col] = result[colpolar]
        #     result.drop(colpolar, axis=1, inplace=True)
        #     result.drop(colfitbit, axis=1, inplace=True)
        # # Selecting only median columns for devices and the target heart rate columns
        # col_list = ['_realtime'] + [f'value_med_{col}' for col in dfselected['device'].unique()] + target_hr_columns
        # result = result[col_list]
        # # st.write('result2:')
        # # st.write(result)








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
# Flatten MultiIndex columns and create column names in {device}_{agg} format
result.columns = [f'{agg}_{device}' for agg, device in result.columns]
result.reset_index(inplace=True)  # Resetting the index after the pivot to make _realtime a column again
col_list = ['_realtime'] + [f'value_med_{col}' for col in dfselected['device'].unique()]
result = result[col_list]
result['target_hr_45'] = dfppt[dfppt['ppt_id'] == int(selected_ppt)]['target_hr_45'].iloc[0]
result['target_hr_55'] = dfppt[dfppt['ppt_id'] == int(selected_ppt)]['target_hr_55'].iloc[0]
result['target_hr_70'] = dfppt[dfppt['ppt_id'] == int(selected_ppt)]['target_hr_70'].iloc[0]
result['target_hr_90'] = dfppt[dfppt['ppt_id'] == int(selected_ppt)]['target_hr_90'].iloc[0]
# result.rename(columns={'value_med_fitbit': 'fitbit','value_med_polar': 'polar'}, inplace=True)

# st.write(dfwos[1])

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
def wo_dur_mict(week):
    if week <= 2:
        return 20
    elif week <= 4:
        return 30
    elif week > 4:
        return 40

def adh_mict_01(df):
    total_time_buckets_5s = wo_dur_mict(df['wk']) * 60 / 5
    auc_lb = total_time_buckets_5s * df['target_hr_70']
    auc_ub = total_time_buckets_5s * df['target_hr_90']
    df['adh_01'] = df['target_hr_70'] + df['target_hr_90']

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

dfselectedgps = dfselected.groupby(['ppt_id', 'wk_id', 'wo_id', 'device', pd.Grouper(key='_realtime', freq='5S')]).agg({'value': 'median','target_hr_70': 'max','target_hr_90': 'max'})
# plt.figure(figsize=(12, 6))
dev_color = {
    'fitbit':           "#F99417", #"#EA5455",#"#F92B2C",
    'polar':            "#2E4374", #"#0668C9"
    'target_hr_45':     "#83C9FF",
    'target_hr_55':     "#F92B2C",
    'target_hr_70':     "#83C9FF",
    'target_hr_90':     "#F92B2C"
}
fig, ax = plt.subplots()#gca()  # Get the current axis
plot_fields = plot_fields[1:]
devices = dfselected['device'].unique()
#todo add layer around target_hr_*
# dfselected.plot(x='_realtime', y='target_hr_70', label='target_hr_70', linestyle='-', alpha=0.85, color="#83C9FF", ax=ax)
# dfselected.plot(x='_realtime', y='target_hr_90', label='target_hr_90', linestyle='-', alpha=0.85, color="#F92B2C", ax=ax)
for yy in get_target_hr_list(randgroup):
    plotcolor = dev_color.get(yy)
    dfselected.plot(x='_realtime', y=yy, label=yy, linestyle='-', alpha=0.85, color=dev_color.get(yy), ax=ax)
for dev in devices:
    # dev = str.lower(dev)
    subset = dfselectedgps.xs(key=dev, level='device', axis=0)  # Extract data for the specific device
    # st.write(subset)
    if str.lower(dev) == 'fitbit':
        dev_short = 'fb'
        lstyle = '--'
    elif str.lower(dev) == 'polar':
        dev_short = 'pl'
        lstyle = '-'
    subset.reset_index().plot(x='_realtime', y='value', label=str.lower(dev), linestyle=lstyle, alpha=0.99, color=dev_color.get(str.lower(dev)), ax=ax)
    
# ax.legend()

st.pyplot(fig)
st.write(dfselected.head(100))