# Well Log Classifier

ML-based lithology classification and porosity estimation from well log data using deep learning.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | **TensorFlow/Keras** - neural network models |
| Data Processing | pandas, numpy, joblib |
| Web Server | **FastAPI** + uvicorn |
| Monitoring | prometheus-fastapi-instrumentator |
| Validation | pydantic v2 |
| Visualization | matplotlib, seaborn |

### Key Libraries
- TensorFlow/Keras - Deep learning for lithology classification
- FastAPI - Modern async web framework
- pandas / numpy - Data processing

## Overview

This project provides deep learning models for analyzing well log measurements:

- **Lithology Classifier**: TensorFlow/Keras neural network that classifies rock type from log curves (gamma ray, resistivity, neutron porosity, density porosity, sonic, caliper).
- **Porosity Estimator**: Keras regressor that predicts density porosity from log features.

## Lithology Classes

- Sandstone
- Limestone
- Shale
- Dolomite
- Anhydrite

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train models

```bash
python train.py
```

### Run API server

```bash
python app.py
```

The server starts on port 5008. Open `http://localhost:5008` for the web UI.

### Run tests

```bash
python test_api.py
```

## API Endpoints

| Method | Path             | Description                        |
|--------|------------------|------------------------------------|
| GET    | `/`              | Web UI                             |
| POST   | `/api/classify`  | Classify lithology from features   |
| POST   | `/api/porosity`  | Estimate porosity from features    |
| GET    | `/api/models`    | Get model info                     |
| GET    | `/api/health`    | Health check                       |

### POST /api/classify

```json
{
  "features": [40, 50, 25, 22, 75, 8.5]
}
```

Features order: gamma_ray, resistivity, neutron_porosity, density_porosity, sonic, caliper.

### POST /api/porosity

```json
{
  "features": [40, 50, 25, 22, 75, 8.5]
}
```

## Project Structure

```
well-log-classifier/
├── well_log_classifier/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lithology_classifier.py
│   │   └── porosity_estimator.py
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py
├── templates/
│   └── index.html
├── outputs/models/
├── .github/workflows/ci.yml
├── train.py
├── app.py
├── test_api.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

Elaborado por Ing. Kelvin Cabrera
