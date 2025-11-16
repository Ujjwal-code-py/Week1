from ultralytics import YOLO
import torch
import os

def create_yaml_and_train():
    print("=== CREATING YAML AND TRAINING ===")
    
    # Create YAML file
    yaml_content = """path: ./dataset
train: train
val: test
test: test

nc: 1
names: ['pothole']
"""
    with open("pothole.yaml", "w") as f:
        f.write(yaml_content)
    print("✅ YAML file created: pothole.yaml")
    
    # Verify GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model and train
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='pothole.yaml',  # Use the YAML file
        epochs=50,
        imgsz=640,
        batch=8,
        workers=2,
        patience=10,
        lr0=0.01,
        save=True,
        device=0,
        name='pothole_detection_v1',
        exist_ok=True
    )
    
    print("✅ Training completed!")
    return results

if __name__ == "__main__":
    create_yaml_and_train()