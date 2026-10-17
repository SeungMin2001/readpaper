FROM node:24-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -f /run/secrets/build_ca ]; then \
      NODE_EXTRA_CA_CERTS=/run/secrets/build_ca npm ci --no-audit --no-fund; \
    else \
      npm ci --no-audit --no-fund; \
    fi
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    READPAPER_DATA_DIR=/data \
    HF_HOME=/data/models/huggingface
COPY backend/requirements.txt /app/backend/requirements.txt
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -f /run/secrets/build_ca ]; then \
      PIP_CERT=/run/secrets/build_ca pip install --no-cache-dir -r /app/backend/requirements.txt; \
    else \
      pip install --no-cache-dir -r /app/backend/requirements.txt; \
    fi
COPY backend/readpaper /app/backend/readpaper
COPY backend/scripts /app/backend/scripts
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist
RUN mkdir -p /data && useradd --system --uid 10001 --home /data readpaper && chown -R readpaper:readpaper /app /data
USER readpaper
EXPOSE 8000
CMD ["uvicorn", "readpaper.main:app", "--host", "0.0.0.0", "--port", "8000"]
