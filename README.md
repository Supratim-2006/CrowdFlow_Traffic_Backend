---
title: CrowdFlow AI Backend
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Traffic Final Backend

This repository contains the backend API for the Traffic Final project. It exposes a unified FastAPI service that performs road scene analysis, congestion estimation, road closure prediction, disruption prediction, and routing over a Bengaluru road graph.

## Features

- Image-based road scene analysis with object detection and congestion scoring
- Lane occupancy-based congestion percentage
- Road closure prediction using a Hugging Face model
- Traffic disruption severity prediction
- Event-aware routing with live event weights
- Local bypass and nearest main road route suggestions

## Repository structure

- `main_api.py` — FastAPI application entrypoint
- `crowdflow_detector2.py` — detector and congestion scoring logic
- `routing.py` — routing and event-weighted graph functions
- `road_closure_app.py` — road closure model utilities
- `models.py` — request/response schemas used by the API
- `requirements.txt` — Python dependencies
- `Traffic_routing/` — graph and event CSV data for routing

## Prerequisites

- Python 3.10 or newer
- Docker (for containerised deployment)

## Local run (without Docker)

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Start the API server:
   ```bash
   uvicorn main_api:app --host 0.0.0.0 --port 7860 --reload
   ```

4. Open the interactive docs: `http://127.0.0.1:7860/docs`

## Local run (with Docker)

```bash
docker build -t crowdflow-local .
docker run --rm -p 7860:7860 crowdflow-local
```

Then visit `http://localhost:7860/docs`.

## Usage

### Analyze endpoint

`POST /analyze`

Form-data fields:

| Field | Type | Description |
|-------|------|-------------|
| `file` | image | Road scene image |
| `road_block_reason` | string | Incident type (e.g. `accident`) |
| `latitude` | float | Incident latitude |
| `longitude` | float | Incident longitude |
| `zone` | string | Zone label |
| `corridor` | string | Corridor label |
| `junction` | string | Optional junction label |

### Routing endpoints

- `POST /routing/nearest-main-road` — find nearest main roads
- `POST /routing/local-bypass` — find bypass around an accident location
- `POST /routing/event-aware-route` — compute a route weighted by live events

Use the Swagger UI at `/docs` to explore request models and try the endpoints interactively.

## Notes

- The app downloads model bundles from Hugging Face Hub at startup for road closure and traffic disruption prediction.
- Routing uses a local `Traffic_routing/bengaluru_crowdflow.graphml` graph and event CSV data.
- CPU inference is used by default; update `device="cuda"` in `main_api.py` if a GPU is available.

## Troubleshooting

- If the API cannot load the graph, verify that `Traffic_routing/bengaluru_crowdflow.graphml` exists and is readable.
- If HF Hub model downloads fail, ensure the `HF_TOKEN` secret is set in Space Settings if the model repos are private.