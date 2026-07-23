# API Documentation - Well Log Classifier

## Base URL

```
http://localhost:5008
```

## Endpoints

### GET /

Serve the main web dashboard UI.

**Response:** HTML page

---

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0.0"
}
```

**Status Codes:**
- 200: Service is healthy

---

### GET /api/models

Return information about loaded models and available features.

**Response:**
```json
{
  "lithology_classifier": {
    "loaded": true,
    "type": "RandomForest + GradientBoosting (VotingClassifier)"
  },
  "porosity_estimator": {
    "loaded": true,
    "type": "GradientBoostingRegressor"
  },
  "features": ["gamma_ray", "resistivity", "neutron_porosity", "density_porosity", "sonic", "caliper"],
  "lithology_classes": ["sandstone", "limestone", "shale", "dolomite", "anhydrite"]
}
```

---

### POST /api/classify

Classify rock lithology from well log features.

**Request:**
```json
{
  "features": [40, 50, 25, 22, 75, 8.5]
}
```

**Feature order:** `gamma_ray`, `resistivity`, `neutron_porosity`, `density_porosity`, `sonic`, `caliper`

**Response:**
```json
{
  "prediction": "sandstone",
  "probabilities": {
    "sandstone": 0.8542,
    "limestone": 0.0521,
    "shale": 0.0312,
    "dolomite": 0.0412,
    "anhydrite": 0.0213
  }
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | Missing `features` field | `{"error": "Missing 'features' field"}` |
| 400 | Wrong feature count | `{"error": "Expected 6 features: [...]"}` |
| 500 | Prediction error | `{"error": "<details>"}` |

---

### POST /api/porosity

Estimate density porosity from well log features.

**Request:**
```json
{
  "features": [40, 50, 25, 22, 75, 8.5]
}
```

**Feature order:** `gamma_ray`, `resistivity`, `neutron_porosity`, `density_porosity`, `sonic`, `caliper`

**Response:**
```json
{
  "porosity": 22.3456
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | Missing `features` field | `{"error": "Missing 'features' field"}` |
| 400 | Wrong feature count | `{"error": "Expected 6 features: [...]"}` |
| 500 | Prediction error | `{"error": "<details>"}` |

---

### GET /api/docs

Return OpenAPI 3.0 specification.

**Response:**
```json
{
  "openapi": "3.0.0",
  "info": {"title": "Well Log Classifier", "version": "1.0.0"},
  "paths": { ... }
}
```

---

## Feature Reference

| Index | Feature | Unit | Typical Range |
|-------|---------|------|---------------|
| 0 | gamma_ray | API | 10 - 200 |
| 1 | resistivity | ohm-m | 5 - 300 |
| 2 | neutron_porosity | % | 5 - 50 |
| 3 | density_porosity | % | 5 - 40 |
| 4 | sonic | us/ft | 40 - 100 |
| 5 | caliper | inches | 6 - 12 |

## Lithology Classes

| Class | Description |
|-------|-------------|
| sandstone | Clastic reservoir rock |
| limestone | Carbonate reservoir rock |
| shale | Seal/non-reservoir rock |
| dolomite | Carbonate reservoir rock |
| anhydrite | Evaporite seal rock |

## Error Codes

- **200**: Success
- **400**: Bad request (missing or invalid parameters)
- **500**: Internal server error

---

*Elaborado por Ing. Kelvin Cabrera*
