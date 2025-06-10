from fastapi import FastAPI
from prometheus_client import start_http_server
from app.api.endpoints import router

app = FastAPI(title="Faster R-CNN API")
app.include_router(router)

start_http_server(8001)  # Prometheus metrics on a different port
