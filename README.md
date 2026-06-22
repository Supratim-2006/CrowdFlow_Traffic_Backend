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
- Git (optional, for cloning)
- [Optional] GPU with CUDA for faster YOLO inference, but CPU mode works too

## Setup and local run

1. Open a terminal in the repository folder:
   ```powershell
   cd C:\Users\supra\OneDrive\Documents\Traffic_Final_Backend
   ```

2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the API server with Uvicorn:
   ```powershell
   uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
   ```

5. Open the API docs in your browser:
   - `http://127.0.0.1:8000/docs`

## Usage

### Analyze endpoint

POST `/analyze`

Form-data fields:

- `file` — image file upload
- `road_block_reason` — text describing the incident (e.g. `accident`)
- `latitude` — incident latitude
- `longitude` — incident longitude
- `zone` — zone label
- `corridor` — corridor label
- `junction` — optional junction label

### Routing endpoints

The API includes `/routing` endpoints:

- `POST /routing/nearest-main-road` — find nearest main roads
- `POST /routing/local-bypass` — find bypass around an accident location
- `POST /routing/event-aware-route` — compute a route weighted by live events

Use the built-in Swagger UI at `/docs` to explore request models and try the endpoints interactively.

## Notes

- The app downloads model bundles from Hugging Face at runtime for road closure and traffic disruption prediction.
- Routing uses a local `Traffic_routing/bengaluru_crowdflow.graphml` graph and event CSV data.
- If you want to change the model device, update `detector = CrowdFlowDetector(model_name=..., device="cpu")` in `main_api.py`.

## Troubleshooting

- If dependency installation fails, ensure `pip` is using the correct Python interpreter from the activated virtual environment.
- If the API cannot load the graph, verify that `Traffic_routing/bengaluru_crowdflow.graphml` exists and is readable.
