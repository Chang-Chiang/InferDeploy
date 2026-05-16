# https://docs.ultralytics.com/

from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
# default: 1 * 84 * 8400
# output: 1 * 8400(max 8400 boxes per image) * 84(80 classes + 4 coordinates[cx, cy, w, h])
model = YOLO("yolov8x.pt")

# Export the model to ONNX format
success = model.export(format="onnx")