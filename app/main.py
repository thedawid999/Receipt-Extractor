from app.io import save_to_file, resolve_image_paths
from pipeline_processor import predict

paths = resolve_image_paths("./samples")

results = []

for path in paths:
    results.append(predict(path))

save_to_file("outputs", results)

