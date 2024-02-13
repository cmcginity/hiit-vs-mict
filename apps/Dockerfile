# Use the specified Python version
FROM python:3.10.12-slim

# Set the working directory in the container to 'apps'
ENV APP_HOME /apps
WORKDIR $APP_HOME
# WORKDIR /apps

# Copy the content of the local apps directory to the working directory inside the container
COPY . .

# Expose the port
# EXPOSE 8501
# EXPOSE 8080
# Install the dependencies for each app using pip
# For study-dash-streamlit-test app
RUN pip3 install -r study-dash-streamlit-test/requirements.txt

# For study-dash app (assuming it has its requirements.txt)
# RUN pip install --no-cache-dir -r study-dash/requirements.txt

# You'd repeat the above line for each app's requirements

RUN find /usr/local/lib/python3.10/site-packages/streamlit -type f \( -iname \*.py -o -iname \*.js \) -print0 | xargs -0 sed -i 's/healthz/health-check/g'


# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the main Streamlit app
# CMD ["streamlit", "run", "--server.port=$PORT", "--server.address=0.0.0.0", "study-dash-streamlit-test/study_overview.py"]
# CMD streamlit run --server.port=$PORT --browser.serverAddress=0.0.0.0 --server.enableCORS=False --server.enableXsrfProtection False study-dash-streamlit-test/study_overview.py
CMD ["sh", "-c", "streamlit run --server.port=${PORT} --browser.serverAddress=0.0.0.0 --server.enableCORS=False --server.enableXsrfProtection False study-dash-streamlit-test/study_overview.py"]

# ENTRYPOINT ["streamlit", "run", "study-dash-streamlit-test/study_overview.py", "--server.port=8501", "--server.address=0.0.0.0"]