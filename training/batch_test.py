from ultralytics import YOLO
import os

def batch_test_model():
    # Load model
    model = YOLO('runs/detect/pothole_detection_v1/weights/best.pt')
    
    # Test on all test images
    test_images_path = "dataset/test"
    image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"🧪 Batch testing on {len(image_files)} images...")
    
    total_potholes = 0
    saved_count = 0
    
    # Create results directory
    os.makedirs("test_results", exist_ok=True)
    
    for image_file in image_files:
        image_path = os.path.join(test_images_path, image_file)
        
        # Run inference
        results = model(image_path, save=False, conf=0.5)
        
        # Save results
        for r in results:
            potholes_detected = len(r.boxes)
            total_potholes += potholes_detected
            
            if potholes_detected > 0:
                # Save image with detections
                r.save(filename=f"test_results/{image_file}")
                saved_count += 1
    
    print(f"\n📊 BATCH TEST RESULTS:")
    print(f"Total images processed: {len(image_files)}")
    print(f"Images with potholes: {saved_count}")
    print(f"Total potholes detected: {total_potholes}")
    print(f"Average potholes per image: {total_potholes/len(image_files):.2f}")
    print(f"Results saved in: test_results/ folder")

if __name__ == "__main__":
    batch_test_model()