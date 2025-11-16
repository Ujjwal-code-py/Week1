import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def configure_cloudinary():
    cloudinary.config(
        cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
        api_key=os.getenv('CLOUDINARY_API_KEY'),
        api_secret=os.getenv('CLOUDINARY_API_SECRET'),
        secure=True
    )
    print("✅ Cloudinary configured successfully")

def upload_to_cloudinary(file_path, folder="uploads", resource_type="image"):
    """Upload file to Cloudinary and return URL"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(file_path)
        public_id = f"pothole-detection/{folder}/{timestamp}_{os.path.splitext(filename)[0]}"
        
        upload_result = cloudinary.uploader.upload(
            file_path,
            public_id=public_id,
            resource_type=resource_type,
            folder=f"pothole-detection/{folder}"
        )
        
        return {
            'success': True,
            'url': upload_result['secure_url'],
            'public_id': upload_result['public_id'],
            'format': upload_result['format'],
            'bytes': upload_result['bytes']
        }
    except Exception as e:
        print(f"❌ Cloudinary upload failed: {e}")
        return {'success': False, 'error': str(e)}

def upload_annotated_image(image_array, original_filename, folder="results"):
    """Upload annotated image (numpy array) to Cloudinary"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        public_id = f"pothole-detection/{folder}/annotated_{timestamp}_{os.path.splitext(original_filename)[0]}"
        
        # Convert numpy array to temporary file
        import tempfile
        import cv2
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, image_array)
            upload_result = cloudinary.uploader.upload(
                temp_file.name,
                public_id=public_id,
                resource_type="image",
                folder=f"pothole-detection/{folder}"
            )
        
        # Clean up temp file
        os.unlink(temp_file.name)
        
        return {
            'success': True,
            'url': upload_result['secure_url'],
            'public_id': upload_result['public_id'],
            'format': upload_result['format'],
            'bytes': upload_result['bytes']
        }
    except Exception as e:
        print(f"❌ Cloudinary annotated image upload failed: {e}")
        return {'success': False, 'error': str(e)}