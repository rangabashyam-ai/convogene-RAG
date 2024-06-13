FROM python:3.9-slim

ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_PROJECT=rag-infobell
ENV LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

WORKDIR /app

# Update pip first
RUN python -m pip install --upgrade pip

# Copy and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
