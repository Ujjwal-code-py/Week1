import os
import yaml

def analyze_dataset():
    dataset_path = "dataset"
    
    print("=== DATASET ANALYSIS ===")
    
    # Check train folder
    train_path = os.path.join(dataset_path, "train")
    train_files = os.listdir(train_path)
    train_images = [f for f in train_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    train_labels = [f for f in train_files if f.endswith('.txt')]
    
    print(f"Train images: {len(train_images)}")
    print(f"Train labels: {len(train_labels)}")
    
    # Check test folder
    test_path = os.path.join(dataset_path, "test")
    test_files = os.listdir(test_path)
    test_images = [f for f in test_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_labels = [f for f in test_files if f.endswith('.txt')]
    
    print(f"Test images: {len(test_images)}")
    print(f"Test labels: {len(test_labels)}")
    
    # Check label format
    if train_labels:
        sample_label = os.path.join(train_path, train_labels[0])
        with open(sample_label, 'r') as f:
            lines = f.readlines()
            print(f"\nSample label format (first 2 lines):")
            for i, line in enumerate(lines[:2]):
                print(f"  Line {i+1}: {line.strip()}")
    
    # Check image dimensions
    if train_images:
        from PIL import Image
        sample_image = os.path.join(train_path, train_images[0])
        img = Image.open(sample_image)
        print(f"\nSample image dimensions: {img.size}")

if __name__ == "__main__":
    analyze_dataset()