from ultralytics import YOLO
import cv2
import os

def test_pothole_detection():
    # Load your trained model
    model = YOLO('runs/detect/pothole_detection_v1/weights/best.pt')
    
    print("✅ Model loaded successfully!")
    print(f"Model mAP50: 0.744")  # Your achieved accuracy
    
    # Test on sample images from test set
    test_images_path = "dataset/test"
    image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Test on first 3 images
    for i, image_file in enumerate(image_files[:3]):
        image_path = os.path.join(test_images_path, image_file)
        print(f"\n🔍 Testing on: {image_file}")
        
        # Run inference
        results = model(image_path, conf=0.5)  # 50% confidence threshold
        
        # Show results
        for r in results:
            # Display image with detections
            im_array = r.plot()  # BGR numpy array
            cv2.imshow(f'Pothole Detection - {image_file}', im_array)
            cv2.waitKey(0)  # Press any key to continue
            cv2.destroyAllWindows()
            
            # Print detection info
            print(f"Potholes detected: {len(r.boxes)}")
            for j, box in enumerate(r.boxes):
                print(f"  Pothole {j+1}: Confidence {box.conf.item():.3f}")
    
    print("\n🎯 Testing completed!")

if __name__ == "__main__":
    test_pothole_detection()