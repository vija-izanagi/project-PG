from ultralytics import YOLO

# Load the Nano Pose model (downloads the standard .pt structure)
model = YOLO("yolo11n-pose.pt")

# Convert it to open ONNX format
model.export(format="onnx") 
print("🚀 Successfully created 'yolo11n-pose.onnx'!")