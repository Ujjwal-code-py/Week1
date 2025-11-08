import os, random, shutil

# Source folders
IMAGES_DIR = "pothole_dataset/images"
LABELS_DIR = "pothole_dataset/labels"

# Target folders
TRAIN_DIR = "pothole_dataset/train"
VAL_DIR = "pothole_dataset/val"

# Create directories
for folder in [TRAIN_DIR, VAL_DIR]:
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder, "labels"), exist_ok=True)

# List all images
images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(images)

# 80% train, 20% validation split
split_index = int(0.8 * len(images))
train_files = images[:split_index]
val_files = images[split_index:]

def copy_files(files, subset_folder):
    for img in files:
        label = img.rsplit('.', 1)[0] + ".txt"
        src_img = os.path.join(IMAGES_DIR, img)
        src_label = os.path.join(LABELS_DIR, label)

        if os.path.exists(src_label):
            shutil.copy(src_img, os.path.join(subset_folder, "images", img))
            shutil.copy(src_label, os.path.join(subset_folder, "labels", label))

copy_files(train_files, TRAIN_DIR)
copy_files(val_files, VAL_DIR)

print(f"✅ Done! {len(train_files)} images in train, {len(val_files)} in val.")
