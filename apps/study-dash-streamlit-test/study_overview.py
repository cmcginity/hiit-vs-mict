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
# proot = pfileabs.parent.parent.parent.parent
pchk = pfileabs
chk = 0
while not pchk.name == 'hiit-vs-mict' or chk >= 10:
    pchk = pchk.parent
    chk += 1
proot = pchk.parent
pdata = os.path.join(proot, 'data')
pdata_myphd = os.path.join(pdata, 'myphd')
# proot = '../../../'


### Functions and Parameters

# Create a sample dataset
np.random.seed(42)
n_data_points = 100
date_range = pd.date_range(start='1/1/2020', periods=n_data_points, freq='D')
sample_data = {
    'date': date_range,
    'value': np.random.randn(n_data_points).cumsum(),
    'field_x': np.random.choice(['A', 'B', 'C', 'D'], n_data_points)
}
df = pd.DataFrame(sample_data)

# def main():
#     st.title('Interactive Time Series Dashboard')

#     # Dropdown for field_x
#     field_x_options = df['field_x'].unique().tolist()
#     selected_field = st.sidebar.selectbox('Select Field X:', field_x_options)

#     # Date range selector
#     min_date = min(filtered_df['date'])
#     max_date = max(filtered_df['date'])
#     date_range = st.sidebar.date_input("Date range", [min_date, max_date])


#     # Filter by field_x
#     filtered_df = df[df['field_x'] == selected_field]

#     # Filter by date range
#     mask = (filtered_df['date'] >= pd.Timestamp(date_range[0])) & (filtered_df['date'] <= pd.Timestamp(date_range[1]))
#     final_df = filtered_df[mask]


#     # Plot
#     st.line_chart(final_df.set_index('date')['value'])

# if __name__ == "__main__":
#     main()

### ### ### ### ### ### Dashboard ### ### ### ### ### ### 
st.markdown(f'path root: {proot}')
st.markdown(f'path pchk: {pchk}')
st.markdown(f'path myphd: {pdata_myphd}') 
st.markdown(f'path file absolute: {pfileabs}') 
st.write('Here\'s some text and stuff:')
st.write(df)