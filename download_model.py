from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="SupratimKukri/crowdflow-model",
    filename="yolov8x-seg.pt"
)

print(MODEL_PATH)