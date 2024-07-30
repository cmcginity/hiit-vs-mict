### Imports and environment
# Imports
# import os
import io
import streamlit as st
import pandas as pd
# import numpy as np
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




### ### ### ### ### ### Dashboard ### ### ### ### ### ###
st.title(f'🏃‍♂️ HIIT-vs-MICT Project 🏃‍♀️')

# TODO: update pipeline...
#       initiate from here?

