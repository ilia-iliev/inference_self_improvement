FROM gemma4-challenge:latest

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
