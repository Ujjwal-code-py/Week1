from training.depth_estimation import PotholeDepthEstimator
from training.cost_estimation import CostEstimator
import os

def main():
    # Run depth estimation
    depth_estimator = PotholeDepthEstimator()
    test_images_path = "dataset/test"
    image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:
        image_path = os.path.join(test_images_path, image_files[0])
        results = depth_estimator.calculate_pothole_dimensions(image_path)
        
        if results:
            pothole_data, image = results
            
            # Run cost estimation
            cost_estimator = CostEstimator()
            cost_estimator.get_user_inputs()
            costs = cost_estimator.calculate_repair_cost(pothole_data)
            cost_estimator.print_cost_report(costs)

if __name__ == "__main__":
    main()