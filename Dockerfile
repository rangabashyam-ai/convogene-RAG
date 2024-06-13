# Use an official Python runtime as a parent image
FROM python:3.9-alpine

# Set environment variables
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_PROJECT=rag-infobell
ENV LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# Set the working directory in the container
WORKDIR /app

# Copy only requirements.txt first to leverage caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
#ENV NAME StreamlitApp

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]

