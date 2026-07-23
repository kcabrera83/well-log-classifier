# Architecture - Well Log Classifier

## System Overview

```
+------------------+     +-------------------+     +------------------+
|   Data Layer     | --> |   Model Layer     | --> |   API Layer      |
| (Data Generator) |     | (ML Models)       |     | (Flask REST)     |
+------------------+     +-------------------+     +------------------+
                                                          |
                                                          v
                                                 +------------------+
                                                 | Dashboard Layer  |
                                                 | (HTML/CSS/JS)    |
                                                 +------------------+
```

## Components

### Data Layer

- **Source**: Synthetic data generator (`data_generator.py`)
- **Samples**: 5,000 well log records across 5 lithologies
- **Preprocessing**: StandardScaler for numeric features, LabelEncoder for lithology labels
- **Features**: gamma_ray, resistivity, neutron_porosity, density_porosity, sonic, caliper

### Model Layer

#### Lithology Classifier
- **Algorithm**: VotingClassifier (RandomForest + GradientBoosting)
- **Task**: Multi-class classification (5 rock types)
- **Input**: 6 well log features
- **Output**: Lithology class + probability distribution
- **Serialization**: pickle (`.pkl`)

#### Porosity Estimator
- **Algorithm**: GradientBoostingRegressor
- **Task**: Regression (density porosity prediction)
- **Input**: 6 well log features
- **Output**: Porosity value (%)
- **Serialization**: pickle (`.pkl`)

### Preprocessing Pipeline

1. Raw well log values received via API
2. Reshaped to (1, 6) feature array
3. Label encoded (classification) or direct features (regression)
4. Passed to trained model

### API Layer

- **Framework**: Flask
- **Port**: 5008
- **Format**: JSON request/response
- **Endpoints**: 6 (classify, porosity, health, models, docs, index)

### Dashboard Layer

- **Frontend**: HTML/CSS/JS (Jinja2 templates)
- **Charts**: Plotly.js for probability visualization
- **Theme**: Dark theme UI

## Data Flow

1. **User Input** -> Dashboard form or API POST request
2. **Validation** -> Check required fields and feature count
3. **Preprocessing** -> Feature array construction
4. **Prediction** -> Model inference (classification or regression)
5. **Post-processing** -> Label decoding (classification), rounding
6. **Response** -> JSON with predictions and probabilities

## Project Structure

```
well-log-classifier/
├── well_log_classifier/
│   ├── __init__.py
│   ├── data_generator.py          # Synthetic data generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lithology_classifier.py  # VotingClassifier wrapper
│   │   └── porosity_estimator.py    # GradientBoosting wrapper
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py          # Scaling, encoding, splitting
├── templates/
│   └── index.html                   # Dashboard UI
├── outputs/models/                  # Saved model artifacts
├── train.py                         # Training pipeline
├── app.py                           # Flask API server
├── test_api.py                      # API test suite
├── requirements.txt
└── setup.py
```

## Model Evaluation

### Lithology Classifier
- Train accuracy: ~0.90+
- Test accuracy: ~0.85+
- Metrics: Precision, Recall, F1-Score per class

### Porosity Estimator
- Train RMSE: ~2.0
- Test RMSE: ~2.5
- R2 score: ~0.85+

---

*Elaborado por Ing. Kelvin Cabrera*
