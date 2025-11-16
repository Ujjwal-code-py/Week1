import os
from PIL import Image

def verify_dataset_structure():
    dataset_path = "./dataset"  # Adjust if needed
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset path doesn't exist!")
        return False
    
    # Check train folder
    train_path = os.path.join(dataset_path, "train")
    if not os.path.exists(train_path):
        print("❌ Train folder doesn't exist!")
        return False
    
    # Check test folder
    test_path = os.path.join(dataset_path, "test")
    if not os.path.exists(test_path):
        print("❌ Test folder doesn't exist!")
        return False
    
    # Count files
    train_images = [f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    train_labels = [f for f in os.listdir(train_path) if f.endswith('.txt')]
    
    test_images = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_labels = [f for f in os.listdir(test_path) if f.endswith('.txt')]
    
    print(f"✅ Train images: {len(train_images)}")
    print(f"✅ Train labels: {len(train_labels)}")
    print(f"✅ Test images: {len(test_images)}")
    print(f"✅ Test labels: {len(test_labels)}")
    
    # Check if image and label names match
    if train_images and train_labels:
        # Get base names without extensions
        image_bases = {os.path.splitext(f)[0] for f in train_images}
        label_bases = {os.path.splitext(f)[0] for f in train_labels}
        
        # Find matching pairs
        matches = image_bases.intersection(label_bases)
        print(f"✅ Matching image-label pairs in train: {len(matches)}")
        
        if matches:
            sample_match = list(matches)[0]
            print(f"✅ Sample match: {sample_match}")
    
    return True

if __name__ == "__main__":
    verify_dataset_structure()