FROM python:3.9-alpine

ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_PROJECT=rag-infobell
ENV LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

WORKDIR /app

# Update pip and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
