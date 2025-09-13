# test_thermal_model.py - Local testing script
import joblib
import numpy as np
import cv2
from PIL import Image
import argparse
import sys
import os

class LocalThermalTester:
    def __init__(self, model_path='thermal_model.pkl'):
        """Initialize the tester with model path"""
        self.model_path = model_path
        self.model_data = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model_data = joblib.load(self.model_path)
            print(f"✓ Model loaded successfully from {self.model_path}")
            self.feature_names = self.model_data['feature_names']
            print(f"✓ Features: {self.feature_names}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def extract_thermal_features(self, image_path):
        """Extract features from thermal image"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image from {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path
            
            print(f"✓ Image loaded: {image.shape}")
            
            # Convert to grayscale for temperature analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Basic temperature statistics (must match training semantics)
            mean_temp = float(np.mean(gray))
            std_temp = float(np.std(gray))
            min_temp = float(np.min(gray))
            max_temp = float(np.max(gray))
            temp_range = float(max_temp - min_temp)
            
            print(f"  - Mean temperature (brightness): {mean_temp:.1f}")
            print(f"  - Temperature variation (std): {std_temp:.1f}")
            
            # Color analysis for RGB thermal images
            if len(image.shape) == 3:
                h, w, c = image.shape
                total_pixels = h * w
                # Color ratio analysis (simple proxy consistent with training intent)
                blue_pixels = np.sum(image[:, :, 2] > image[:, :, 0])  # More blue than red
                red_pixels = np.sum(image[:, :, 0] > image[:, :, 2])   # More red than blue
                green_pixels = np.sum(image[:, :, 1] > np.maximum(image[:, :, 0], image[:, :, 2]))

                blue_ratio = float(blue_pixels) / float(total_pixels)
                red_ratio = float(red_pixels) / float(total_pixels)
                green_ratio = float(green_pixels) / float(total_pixels)
                yellow_ratio = float(max(0.0, 1.0 - blue_ratio - red_ratio - green_ratio))
                
                print(f"  - Blue ratio: {blue_ratio:.3f} (indicates cooler regions)")
                print(f"  - Red ratio: {red_ratio:.3f} (indicates warmer regions)")
                print(f"  - Green ratio: {green_ratio:.3f} (indicates moderate regions)")
                print(f"  - Yellow ratio: {yellow_ratio:.3f} (indicates hot regions)")
            else:
                blue_ratio = red_ratio = green_ratio = yellow_ratio = 0.25
                print("  - Grayscale image: using default color ratios")
            
            # Gradient analysis for uniformity
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))
            
            print(f"  - Gradient magnitude: {gradient_magnitude:.1f} (uniformity indicator)")
            
            # Uniformity/regularity (training used 1 / (1 + std_temp))
            uniformity_score = 1.0 / (1.0 + std_temp)
            regularity_est = uniformity_score  # keep printed value for continuity
            print(f"  - Estimated regularity: {regularity_est:.3f}")
            
            # Hotspot detection
            threshold = mean_temp + 1.5 * std_temp
            hotspots = gray > threshold
            hotspot_count = int(min(cv2.connectedComponents(hotspots.astype(np.uint8))[0] - 1, 10))
            
            print(f"  - Hotspot count: {hotspot_count}")
            
            # Build features in the exact order the model expects
            if not self.feature_names:
                raise ValueError("Model feature_names missing in saved model data.")

            feats_dict = {
                'mean_temp': mean_temp,
                'std_temp': std_temp,
                'min_temp': min_temp,
                'max_temp': max_temp,
                'temp_range': temp_range,
                'blue_ratio': blue_ratio,
                'green_ratio': green_ratio,
                'red_ratio': red_ratio,
                'yellow_ratio': yellow_ratio,
                'gradient_magnitude': gradient_magnitude,
                'uniformity_score': uniformity_score,
                'hotspot_count': float(hotspot_count),
            }

            features = np.array([feats_dict[name] for name in self.feature_names], dtype=float)
            thermal_variation_pct = (std_temp / 255.0) * 100.0
            aux = {
                'thermal_variation': std_temp,
                'thermal_variation_pct': thermal_variation_pct
            }
            return features, aux
            
        except Exception as e:
            print(f"✗ Error processing image: {e}")
            return None
    
    def predict_properties(self, image_path):
        """Predict material properties from image"""
        print(f"\n🔬 Analyzing thermal image: {image_path}")
        print("=" * 50)
        
        # Extract features
        out = self.extract_thermal_features(image_path)
        if out is None:
            return None
        features, aux = out
        
        # Scale features
        features_scaled = self.model_data['scaler'].transform(features.reshape(1, -1))
        
        # Make prediction
        predictions = self.model_data['model'].predict(features_scaled)[0]
        
        # Ensure realistic bounds
        regularity = max(0, min(1, predictions[0]))
        tensile = max(0, predictions[1])
        elongation = max(0, predictions[2])
        abrasion = max(0, int(predictions[3]))
        
        return {
            'regularity_score': regularity,
            'tensile_strength_mpa': tensile,
            'elongation_at_break_percent': elongation,
            'abrasion_resistance_cycles': abrasion,
            'thermal_variation': aux.get('thermal_variation'),
            'thermal_variation_pct': aux.get('thermal_variation_pct')
        }
    
    def format_results(self, results):
        """Format prediction results for display"""
        if results is None:
            return "Analysis failed"
        
        # Composite quality assessment: tensile, elongation, thermal variation %
        tensile = results['tensile_strength_mpa']
        elong = results['elongation_at_break_percent']
        var_pct = results.get('thermal_variation_pct')

        if (var_pct is not None and var_pct < 10.0) and (tensile > 70.0) and (elong > 5.0):
            quality = "Excellent"
            quality_icon = "🟢"
            recommendation = "Production-ready for high-stress use; maintain controls."
        elif (var_pct is not None and var_pct < 16.0) and (tensile > 50.0) and (elong > 7.0):
            quality = "Good"
            quality_icon = "🟡"
            recommendation = "Moderate-stress use; improve uniformity to advance."
        else:
            quality = "Poor"
            quality_icon = "🔴"
            recommendation = "Reduce variation and increase tensile/elongation."

        # Format output
        output = f"""
📊 MATERIAL PROPERTY PREDICTIONS
================================

🔍 Thermal Analysis Results:
• Regularity Score: {results['regularity_score']:.3f}/1.000
• Tensile Strength: {results['tensile_strength_mpa']:.1f} MPa
• Elongation at Break: {results['elongation_at_break_percent']:.1f}%
• Abrasion Resistance: {results['abrasion_resistance_cycles']:,} cycles
• Thermal Variation (std): {results.get('thermal_variation', float('nan')):.1f}
• Thermal Variation (% of 255): {var_pct:.1f}%

{quality_icon} Quality Assessment: {quality}

💡 Recommendations:
{recommendation}

📐 Quality Rules:
• Excellent: variation < 10%, tensile > 70.0 MPa, elongation > 5%
• Good: variation < 16%, tensile > 50.0 MPa, elongation > 7%
• Otherwise: Poor

📝 Analysis Notes:
• Lower thermal variation indicates more uniform material (better repeatability)
• Tensile strength > 70 MPa indicates superior structural properties  
• Higher elongation suggests better flexibility/ductility
"""
        return output
    
    def test_synthetic_sample(self):
        """Test with synthetic features (no image required)"""
        print("\n🧪 Testing with synthetic sample...")
        print("=" * 50)
        
        # Create synthetic thermal features aligned to model feature_names (12 features)
        syn = {
            'mean_temp': 180.0,
            'std_temp': 20.0,
            'min_temp': 120.0,
            'max_temp': 240.0,
            'temp_range': 120.0,
            'blue_ratio': 0.40,
            'green_ratio': 0.30,
            'red_ratio': 0.20,
            'yellow_ratio': 0.10,
            'gradient_magnitude': 12.0,
            'uniformity_score': 1.0 / (1.0 + 20.0),
            'hotspot_count': 2.0,
        }
        synthetic_features = np.array([[syn[name] for name in self.feature_names]], dtype=float)

        print("Synthetic features:")
        for name, value in zip(self.feature_names, synthetic_features[0]):
            print(f"  - {name}: {value}")

        # Scale and predict
        features_scaled = self.model_data['scaler'].transform(synthetic_features)
        predictions = self.model_data['model'].predict(features_scaled)[0]
        
        results = {
            'regularity_score': max(0, min(1, predictions[0])),
            'tensile_strength_mpa': max(0, predictions[1]),
            'elongation_at_break_percent': max(0, predictions[2]),
            'abrasion_resistance_cycles': max(0, int(predictions[3])),
            'thermal_variation': syn['std_temp'],
            'thermal_variation_pct': (syn['std_temp']/255.0)*100.0
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Test thermal analysis model locally')
    parser.add_argument('--image', '-i', type=str, help='Path to thermal image')
    parser.add_argument('--model', '-m', type=str, default='thermal_model.pkl', help='Path to model file')
    parser.add_argument('--synthetic', '-s', action='store_true', help='Test with synthetic sample')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"✗ Model file not found: {args.model}")
        print("Please run 'python train_model.py' first to create the model.")
        sys.exit(1)
    
    # Initialize tester
    tester = LocalThermalTester(args.model)
    
    if args.synthetic or not args.image:
        # Test with synthetic sample
        results = tester.test_synthetic_sample()
        print(tester.format_results(results))
    
    if args.image:
        # Test with real image
        if not os.path.exists(args.image):
            print(f"✗ Image file not found: {args.image}")
            sys.exit(1)
        
        results = tester.predict_properties(args.image)
        print(tester.format_results(results))

if __name__ == "__main__":
    main()