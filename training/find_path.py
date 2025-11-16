import os

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Check if dataset exists in current location
dataset_path = "dataset"
if os.path.exists(dataset_path):
    print(f"✅ Dataset found at: {os.path.abspath(dataset_path)}")
    
    # List contents
    print("\nDataset contents:")
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            print(f"📁 {item}/")
            sub_items = os.listdir(item_path)[:5]  # Show first 5 items
            for sub_item in sub_items:
                print(f"   📄 {sub_item}")
            if len(os.listdir(item_path)) > 5:
                print(f"   ... and {len(os.listdir(item_path)) - 5} more")
        else:
            print(f"📄 {item}")
else:
    print(f"❌ Dataset not found at: {os.path.abspath(dataset_path)}")
    
    # Search for dataset in parent directories
    print("\nSearching for dataset in parent directories...")
    current_dir = os.getcwd()
    for root, dirs, files in os.walk(current_dir):
        if "dataset" in dirs:
            found_path = os.path.join(root, "dataset")
            print(f"✅ Found dataset at: {found_path}")
            break
    else:
        print("❌ Dataset not found in any parent directories")