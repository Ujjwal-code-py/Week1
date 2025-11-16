class CostEstimator:
    def __init__(self):
        # Default values (will be updated by user)
        self.material_cost_per_liter = 40.0  # ₹ per liter
        self.labor_cost_per_hour = 300.0     # ₹ per hour per worker
        self.team_size = 2                   # Number of workers
        self.overhead_percentage = 15        # 15% overhead
        
        print("💰 Cost Estimator (User Configurable)")
        print("All costs in Indian Rupees (₹)")
    
    def get_user_inputs(self):
        """Get cost inputs from user"""
        print("\n📝 ENTER COST PARAMETERS:")
        print("-" * 40)
        
        try:
            self.material_cost_per_liter = float(input("Enter material cost per liter (₹): ") or "40.0")
            self.labor_cost_per_hour = float(input("Enter labor cost per hour per worker (₹): ") or "300.0")
            self.team_size = int(input("Enter number of workers in team: ") or "2")
            self.overhead_percentage = float(input("Enter overhead percentage (%): ") or "15.0")
            
            print("✅ Cost parameters updated successfully!")
            
        except ValueError:
            print("❌ Invalid input! Using default values.")
    
    def calculate_repair_time(self, pothole_data):
        """Calculate realistic repair time based on pothole volumes"""
        total_volume = sum(pothole['volume_liters'] for pothole in pothole_data)
        total_potholes = len(pothole_data)
        
        # Realistic time estimates (based on road maintenance standards)
        base_setup_time = 0.5  # hours (equipment setup)
        base_cleanup_time = 0.25  # hours
        
        # Time per pothole depends on size
        prep_time_per_pothole = 0.08  # hours (5 minutes) for cleaning/prep
        fill_time_per_liter = 0.003   # hours per liter
        compact_time_per_pothole = 0.05  # hours (3 minutes) for compaction
        
        # Calculate total time
        total_prep_time = total_potholes * prep_time_per_pothole
        total_fill_time = total_volume * fill_time_per_liter
        total_compact_time = total_potholes * compact_time_per_pothole
        
        total_hours = (base_setup_time + total_prep_time + 
                      total_fill_time + total_compact_time + base_cleanup_time)
        
        return {
            'total_hours': total_hours,
            'setup_time': base_setup_time,
            'prep_time': total_prep_time,
            'fill_time': total_fill_time,
            'compact_time': total_compact_time,
            'cleanup_time': base_cleanup_time
        }
    
    # Update the calculate_repair_cost method to optionally include location
    def calculate_repair_cost(self, pothole_data, location_data=None):
        """Calculate total costs based on user inputs"""
        if not pothole_data:
            return None
        
        total_volume = sum(pothole['volume_liters'] for pothole in pothole_data)
        total_potholes = len(pothole_data)
        
        # Calculate repair time
        time_breakdown = self.calculate_repair_time(pothole_data)
        
        # Material cost (with 10% waste factor)
        material_required = total_volume * 1.10
        material_cost = material_required * self.material_cost_per_liter
        
        # Labor cost
        labor_cost = time_breakdown['total_hours'] * self.labor_cost_per_hour * self.team_size
        
        # Equipment and transport (fixed costs)
        equipment_cost = 500.0  # ₹ per job
        transport_cost = 300.0  # ₹ per job
        
        # Subtotal
        subtotal = material_cost + labor_cost + equipment_cost + transport_cost
        
        # Overhead
        overhead_cost = subtotal * (self.overhead_percentage / 100)
        
        # Total cost
        total_cost = subtotal + overhead_cost
        
        cost_breakdown = {
            'total_potholes': total_potholes,
            'total_volume_liters': total_volume,
            'material_required_liters': material_required,
            'material_cost_per_liter': self.material_cost_per_liter,
            'labor_cost_per_hour': self.labor_cost_per_hour,
            'team_size': self.team_size,
            'overhead_percentage': self.overhead_percentage,
            
            'material_cost': material_cost,
            'labor_cost': labor_cost,
            'equipment_cost': equipment_cost,
            'transport_cost': transport_cost,
            'overhead_cost': overhead_cost,
            'total_cost': total_cost,
            'cost_per_pothole': total_cost / total_potholes,
            
            'time_breakdown': time_breakdown,
            'location_data': location_data  # Include location data in cost breakdown
        }
        
        return cost_breakdown
    
    def print_cost_report(self, cost_breakdown):
        """Print comprehensive cost report"""
        if not cost_breakdown:
            print("❌ No cost data available")
            return
        
        print("\n" + "="*70)
        print("💰 POTHLE REPAIR COST ESTIMATION (₹)")
        print("="*70)
        
        # Project Summary
        print(f"📊 PROJECT SUMMARY:")
        print(f"   Total Potholes: {cost_breakdown['total_potholes']}")
        print(f"   Total Volume: {cost_breakdown['total_volume_liters']:.1f} liters")
        print(f"   Material Required: {cost_breakdown['material_required_liters']:.1f} liters (with 10% waste)")
        
        # Time Breakdown
        time_info = cost_breakdown['time_breakdown']
        print(f"\n⏰ TIME ESTIMATION:")
        print(f"   Total Time: {time_info['total_hours']:.2f} hours")
        print(f"   - Setup: {time_info['setup_time']:.2f} hours")
        print(f"   - Preparation: {time_info['prep_time']:.2f} hours")
        print(f"   - Filling: {time_info['fill_time']:.2f} hours")
        print(f"   - Compaction: {time_info['compact_time']:.2f} hours")
        print(f"   - Cleanup: {time_info['cleanup_time']:.2f} hours")
        
        # Cost Breakdown
        print(f"\n💵 COST BREAKDOWN:")
        print(f"   Material: ₹{cost_breakdown['material_cost']:.2f}")
        print(f"     ({cost_breakdown['material_required_liters']:.1f}L × ₹{cost_breakdown['material_cost_per_liter']:.2f}/L)")
        
        print(f"   Labor: ₹{cost_breakdown['labor_cost']:.2f}")
        print(f"     ({time_info['total_hours']:.2f}h × {cost_breakdown['team_size']} workers × ₹{cost_breakdown['labor_cost_per_hour']:.2f}/h)")
        
        print(f"   Equipment: ₹{cost_breakdown['equipment_cost']:.2f}")
        print(f"   Transport: ₹{cost_breakdown['transport_cost']:.2f}")
        print(f"   Overhead ({cost_breakdown['overhead_percentage']}%): ₹{cost_breakdown['overhead_cost']:.2f}")
        
        print(f"\n💰 TOTAL COST: ₹{cost_breakdown['total_cost']:.2f}")
        print(f"   Cost per Pothole: ₹{cost_breakdown['cost_per_pothole']:.2f}")
        print("="*70)

# Test function
def main():
    estimator = CostEstimator()
    
    # Get user inputs
    estimator.get_user_inputs()
    
    # Sample pothole data (from detection)
    sample_potholes = [
        {'volume_liters': 8.5},
        {'volume_liters': 15.2},
        {'volume_liters': 22.7},
        {'volume_liters': 12.1},
    ]
    
    print(f"\n🧪 Testing with sample data:")
    print(f"   {len(sample_potholes)} potholes, {sum(p['volume_liters'] for p in sample_potholes):.1f} liters total")
    
    # Calculate costs
    costs = estimator.calculate_repair_cost(sample_potholes)
    estimator.print_cost_report(costs)

if __name__ == "__main__":
    main()