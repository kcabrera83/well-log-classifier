# User Guide - Well Log Classifier

## Overview

The Well Log Classifier is a machine learning system for lithology classification and porosity estimation from well log measurements. It provides a web dashboard and REST API for real-time predictions.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd well-log-classifier
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

This generates 5,000 synthetic well log samples and trains:
- Lithology Classifier (RandomForest + GradientBoosting ensemble)
- Porosity Estimator (GradientBoosting)

Models are saved to `outputs/models/`.

### Run the Server

```bash
python app.py
```

Open `http://localhost:5008` in your browser.

## Dashboard Features

- **Lithology Classification Panel** - Enter well log values and get rock type predictions with probability breakdown
- **Porosity Estimation Panel** - Enter well log values and get density porosity predictions
- **Model Status** - Shows loaded model types and feature columns
- **Results Visualization** - Probability bar charts for lithology predictions

## API Usage

### Classify Lithology (curl)

```bash
curl -X POST http://localhost:5008/api/classify \
  -H "Content-Type: application/json" \
  -d '{"features": [40, 50, 25, 22, 75, 8.5]}'
```

### Classify Lithology (Python)

```python
import requests

response = requests.post("http://localhost:5008/api/classify", json={
    "features": [40, 50, 25, 22, 75, 8.5]
})
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probabilities: {result['probabilities']}")
```

### Estimate Porosity (curl)

```bash
curl -X POST http://localhost:5008/api/porosity \
  -H "Content-Type: application/json" \
  -d '{"features": [40, 50, 25, 22, 75, 8.5]}'
```

### Estimate Porosity (Python)

```python
import requests

response = requests.post("http://localhost:5008/api/porosity", json={
    "features": [40, 50, 25, 22, 75, 8.5]
})
result = response.json()
print(f"Porosity: {result['porosity']}%")
```

### Check Health

```bash
curl http://localhost:5008/api/health
```

### Get Model Info

```bash
curl http://localhost:5008/api/models
```

## Feature Values Guide

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| gamma_ray | Natural gamma radiation | 10 - 200 API |
| resistivity | Electrical resistivity | 5 - 300 ohm-m |
| neutron_porosity | Neutron porosity log | 5 - 50 % |
| density_porosity | Density-derived porosity | 5 - 40 % |
| sonic | Sonic travel time | 40 - 100 us/ft |
| caliper | Borehole diameter | 6 - 12 inches |

## Running Tests

```bash
python test_api.py
```

## Troubleshooting

- **Models not loaded**: Run `python train.py` before starting the server
- **Feature count error**: Ensure exactly 6 features are provided in the correct order
- **Port in use**: Change the port in `app.py` or stop other services on port 5008

---

*Elaborado por Ing. Kelvin Cabrera*
