# Deployment Guide - Well Log Classifier

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5008

CMD ["python", "app.py"]
```

### Build and Run

```bash
docker build -t well-log-classifier .
docker run -p 5008:5008 well-log-classifier
```

## Docker Compose

```yaml
version: '3.8'
services:
  well-log-classifier:
    build: .
    ports:
      - "5008:5008"
    environment:
      - FLASK_ENV=production
    volumes:
      - model-data:/app/outputs

volumes:
  model-data:
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PORT | Server port | 5008 |

## Production Considerations

- Use gunicorn for production serving:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5008 app:app
  ```
- Set `debug=False` in `app.py` (already set)
- Configure reverse proxy (nginx) for SSL termination
- Set up health check monitoring on `/api/health`
- Use a process manager (systemd, supervisor) for auto-restart

## Training Pipeline

1. `python train.py` generates synthetic data and trains models
2. Models are saved to `outputs/models/`
3. `python app.py` loads models and starts the API server

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push to main
- Installs dependencies
- Runs training pipeline
- Executes API tests

---

*Elaborado por Ing. Kelvin Cabrera*
