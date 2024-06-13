FROM python:3.12-slim

# Update package list and install required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
    
# Set the working directory in the container
WORKDIR /app

# Copy only requirements.txt to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV COHERE_API_KEY=${COHERE_API_KEY}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV PINECONE_API_KEY=${PINECONE_API_KEY}

# Expose the port the app runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "side_by_side_app.py"]
