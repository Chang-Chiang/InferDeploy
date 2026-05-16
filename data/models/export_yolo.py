from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")

# Export the model to ONNX format
success = model.export(format="onnx")