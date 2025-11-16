import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class PotholeDepthEstimator:
    def __init__(self):
        self.detection_model = YOLO('D:/Ujjwal/Pothole detection/runs/detect/pothole_detection_v1/weights/best.pt')
        
        # Camera calibration (can be adjusted by user)
        self.pixels_per_meter = 2000  # Default: 2000 pixels = 1 meter
        self.reference_object_width = 0.0  # Known width in meters
        self.reference_pixel_width = 0.0   # Width in pixels
        
        # Depth estimation parameters (based on road studies)
        self.depth_estimation_methods = {
            'geometric': 0.6,    # Weight for geometric method
            'texture': 0.3,      # Weight for texture analysis
            'shadow': 0.1        # Weight for shadow analysis
        }
        
        print("🎯 Pothole Depth Estimator Initialized")
    
    def calibrate_using_reference(self, image, reference_width_meters, reference_pixel_width):
        """Calibrate the scale using a known object in the image"""
        self.reference_object_width = reference_width_meters
        self.reference_pixel_width = reference_pixel_width
        self.pixels_per_meter = reference_pixel_width / reference_width_meters
        
        print(f"✅ Calibrated: {self.pixels_per_meter:.0f} pixels/meter")
        print(f"   1 pixel = {100/self.pixels_per_meter:.2f} cm")
    
    def estimate_depth_geometric(self, width_meters):
        """Estimate depth based on pothole geometry and empirical data"""
        # Based on road maintenance studies and pothole geometry
        if width_meters < 0.15:    # Very small potholes (<15cm)
            return width_meters * 0.25  # Shallow
        elif width_meters < 0.30:  # Small potholes (15-30cm)
            return width_meters * 0.35  # Medium depth
        elif width_meters < 0.60:  # Medium potholes (30-60cm)
            return width_meters * 0.30  # Standard depth
        else:                       # Large potholes (>60cm)
            return width_meters * 0.25  # Relatively shallower
    
    def analyze_texture_depth(self, image_region):
        """Analyze texture to estimate depth (rougher texture = deeper)"""
        if image_region.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture metrics
        texture_std = np.std(gray)  # Standard deviation (roughness)
        texture_range = np.ptp(gray)  # Peak-to-peak (contrast)
        
        # Normalize and combine texture features
        texture_score = (texture_std / 50 + texture_range / 100) / 2
        texture_score = min(max(texture_score, 0), 1)  # Clamp to 0-1
        
        # Convert to depth (0-10cm based on texture)
        depth_from_texture = texture_score * 0.10  # Max 10cm from texture
        
        return depth_from_texture
    
    def analyze_shadow_depth(self, image_region):
        """Analyze shadows to estimate depth (darker = deeper)"""
        if image_region.size == 0:
            return 0.0
        
        # Convert to HSV for better shadow detection
        hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
        
        # Calculate average brightness in Value channel
        avg_brightness = np.mean(hsv[:,:,2])
        
        # Darker regions indicate deeper potholes
        darkness_factor = (255 - avg_brightness) / 255  # 0=light, 1=dark
        
        # Convert to depth (0-5cm based on darkness)
        depth_from_shadow = darkness_factor * 0.05  # Max 5cm from shadows
        
        return depth_from_shadow
    
    def estimate_combined_depth(self, width_meters, image_region):
        """Combine multiple methods for best depth estimation"""
        # Method 1: Geometric estimation
        depth_geo = self.estimate_depth_geometric(width_meters)
        
        # Method 2: Texture analysis
        depth_texture = self.analyze_texture_depth(image_region)
        
        # Method 3: Shadow analysis
        depth_shadow = self.analyze_shadow_depth(image_region)
        
        # Weighted combination
        weights = self.depth_estimation_methods
        combined_depth = (
            depth_geo * weights['geometric'] +
            depth_texture * weights['texture'] + 
            depth_shadow * weights['shadow']
        )
        
        # Ensure depth is reasonable (not more than 50% of width)
        max_reasonable_depth = width_meters * 0.5
        final_depth = min(combined_depth, max_reasonable_depth)
        
        # Minimum depth for any pothole
        final_depth = max(final_depth, 0.02)  # At least 2cm
        
        return {
            'final_depth_m': final_depth,
            'final_depth_cm': final_depth * 100,
            'methods': {
                'geometric': depth_geo * 100,
                'texture': depth_texture * 100,
                'shadow': depth_shadow * 100
            },
            'width_m': width_meters
        }
    
    def calculate_pothole_dimensions(self, image_path, reference_width_meters=0, reference_pixel_width=0):
        """Complete pothole analysis with best depth estimation"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        print(f"\n🔍 Analyzing: {image_path}")
        
        # Calibrate if reference provided
        if reference_width_meters > 0 and reference_pixel_width > 0:
            self.calibrate_using_reference(image, reference_width_meters, reference_pixel_width)
        else:
            print("ℹ️ Using default calibration (2000 pixels/meter)")
        
        # Detect potholes
        results = self.detection_model(image, conf=0.5)
        
        if len(results[0].boxes) == 0:
            print("❌ No potholes detected")
            return None
        
        pothole_data = []
        
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            width_px = x2 - x1
            
            # Convert to real-world measurements
            width_meters = width_px / self.pixels_per_meter
            width_cm = width_meters * 100
            
            # Extract pothole region for analysis
            pothole_region = image[y1:y2, x1:x2]
            
            # Estimate depth using combined methods
            depth_info = self.estimate_combined_depth(width_meters, pothole_region)
            
            # Calculate volume (cylindrical approximation)
            radius_m = width_meters / 2
            volume_m3 = 3.14159 * (radius_m ** 2) * depth_info['final_depth_m']
            volume_liters = volume_m3 * 1000
            
            pothole_info = {
                'id': i + 1,
                'confidence': box.conf.item(),
                'width_px': width_px,
                'width_cm': width_cm,
                'depth_cm': depth_info['final_depth_cm'],
                'volume_liters': volume_liters,
                'bbox': (x1, y1, x2, y2),
                'depth_analysis': depth_info
            }
            
            pothole_data.append(pothole_info)
            
            print(f"\n🕳️ Pothole {i+1} Analysis:")
            print(f"   Confidence: {box.conf.item():.3f}")
            print(f"   Width: {width_cm:.1f} cm")
            print(f"   Depth: {depth_info['final_depth_cm']:.1f} cm")
            print(f"     - Geometric: {depth_info['methods']['geometric']:.1f} cm")
            print(f"     - Texture: {depth_info['methods']['texture']:.1f} cm")
            print(f"     - Shadow: {depth_info['methods']['shadow']:.1f} cm")
            print(f"   Volume: {volume_liters:.2f} liters")
        
        return pothole_data, image
    def calculate_pothole_dimensions_from_array(self, image_array):
        """Complete pothole analysis directly from numpy array (much faster)"""
        if image_array is None:
            return None
        
        # Detect potholes directly from array
        results = self.detection_model(image_array, conf=0.5)
        
        if len(results[0].boxes) == 0:
            # Return empty list instead of None to continue counting frames
            return [], image_array
        
        pothole_data = []
        
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            width_px = x2 - x1
            
            # Convert to real-world measurements
            width_meters = width_px / self.pixels_per_meter
            width_cm = width_meters * 100
            
            # Extract pothole region for analysis
            pothole_region = image_array[y1:y2, x1:x2]
            
            # Estimate depth using combined methods
            depth_info = self.estimate_combined_depth(width_meters, pothole_region)
            
            # Calculate volume (cylindrical approximation)
            radius_m = width_meters / 2
            volume_m3 = 3.14159 * (radius_m ** 2) * depth_info['final_depth_m']
            volume_liters = volume_m3 * 1000
            
            pothole_info = {
                'id': i + 1,
                'confidence': box.conf.item(),
                'width_px': width_px,
                'width_cm': width_cm,
                'depth_cm': depth_info['final_depth_cm'],
                'volume_liters': volume_liters,
                'bbox': (x1, y1, x2, y2),
                'depth_analysis': depth_info
            }
            
            pothole_data.append(pothole_info)
        
        return pothole_data, image_array
    
    def visualize_results(self, image, pothole_data, output_path):
        """Visualize detection results with dimensions"""
        result_image = image.copy()
        
        for pothole in pothole_data:
            x1, y1, x2, y2 = pothole['bbox']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add info text
            info_lines = [
                f"Pothole {pothole['id']}",
                f"W:{pothole['width_cm']:.1f}cm D:{pothole['depth_cm']:.1f}cm",
                f"V:{pothole['volume_liters']:.1f}L"
            ]
            
            y_offset = y1 - 10
            for line in info_lines:
                cv2.putText(result_image, line, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_offset -= 15
        
        # Save and show
        cv2.imwrite(output_path, result_image)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Pothole Detection with Dimensions')
        plt.axis('off')
        plt.show()
        
        print(f"💾 Results saved: {output_path}")

def main():
    estimator = PotholeDepthEstimator()
    
    # Test on sample image
    test_images_path = "dataset/test"
    import os
    image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:
        image_path = os.path.join(test_images_path, image_files[0])
        
        print("="*70)
        print("POTHLE DEPTH ESTIMATION WITH COMBINED METHODS")
        print("="*70)
        
        # Analyze with default calibration
        results = estimator.calculate_pothole_dimensions(image_path)
        
        if results:
            pothole_data, image = results
            estimator.visualize_results(image, pothole_data, "depth_estimation_result.jpg")
            return pothole_data

if __name__ == "__main__":
    main()