### Imports and environment
# Imports
# import os
import io
import streamlit as st
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
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

# @st.cache_data
def get_dfmetadata(_drive_service, folder_id=None):
    pworkout = st.secrets["gdrive_id_workout"]
    dfmetadata_list = []

    try:
        # If folder_id is provided, list files from that folder. Otherwise, list from entire Drive.
        if folder_id:
            response = _drive_service.files().list(corpora='drive', 
                                                driveId=st.secrets["gdrive_id_root"],
                                                q=f"'{pworkout}' in parents",
                                                includeItemsFromAllDrives=True, 
                                                supportsAllDrives=True).execute()
        else:
            response = _drive_service.files().list(corpora='drive', 
                                                driveId=st.secrets["gdrive_id_root"],
                                                includeItemsFromAllDrives=True, 
                                                supportsAllDrives=True).execute()
        
        for file in response.get('files', []):
            fname = file.get('name')
            if not fname.startswith("q"): #fname.startswith("0"): # todo: this will be a breaking error soon!!!
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
