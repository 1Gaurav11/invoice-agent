FROM python:3.12-slim
WORKDIR /app
COPY server.py store.py decide.py inspect_db.py ./
ENV PYTHONUNBUFFERED=1 DB_PATH=/data/agent.sqlite PORT=8080
VOLUME ["/data"]
EXPOSE 8080
CMD ["python3", "server.py"]
