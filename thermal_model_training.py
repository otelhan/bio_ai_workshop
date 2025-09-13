# thermal_model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import cv2
from PIL import Image
import os

class ThermalImageAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'mean_temp', 'std_temp', 'min_temp', 'max_temp', 'temp_range',
            'blue_ratio', 'green_ratio', 'red_ratio', 'yellow_ratio',
            'gradient_magnitude', 'uniformity_score', 'hotspot_count'
        ]
        
    def extract_thermal_features(self, image_path_or_array):
        """Extract thermal features from thermal image"""
        if isinstance(image_path_or_array, str):
            # Load image from path
            img = cv2.imread(image_path_or_array)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path_or_array
            
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Calculate basic temperature statistics (using brightness as proxy)
        brightness = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_temp = np.mean(brightness)
        std_temp = np.std(brightness)
        min_temp = np.min(brightness)
        max_temp = np.max(brightness)
        temp_range = max_temp - min_temp
        
        # Color ratio analysis
        blue_mask = (hsv[:,:,0] >= 100) & (hsv[:,:,0] <= 130)  # Blue hues
        green_mask = (hsv[:,:,0] >= 40) & (hsv[:,:,0] <= 80)   # Green hues  
        red_mask = ((hsv[:,:,0] >= 0) & (hsv[:,:,0] <= 20)) | (hsv[:,:,0] >= 160)  # Red hues
        yellow_mask = (hsv[:,:,0] >= 20) & (hsv[:,:,0] <= 40)  # Yellow hues
        
        total_pixels = img.shape[0] * img.shape[1]
        blue_ratio = np.sum(blue_mask) / total_pixels
        green_ratio = np.sum(green_mask) / total_pixels
        red_ratio = np.sum(red_mask) / total_pixels
        yellow_ratio = np.sum(yellow_mask) / total_pixels
        
        # Gradient analysis for uniformity
        grad_x = cv2.Sobel(brightness, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(brightness, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Uniformity score (inverse of standard deviation)
        uniformity_score = 1 / (1 + std_temp)
        
        # Hotspot detection
        threshold = mean_temp + 1.5 * std_temp
        hotspots = brightness > threshold
        hotspot_count = cv2.connectedComponents(hotspots.astype(np.uint8))[0] - 1
        
        return np.array([
            mean_temp, std_temp, min_temp, max_temp, temp_range,
            blue_ratio, green_ratio, red_ratio, yellow_ratio,
            gradient_magnitude, uniformity_score, hotspot_count
        ])
    
    def create_training_data(self, csv_path):
        """Create training data from CSV and generate synthetic features"""
        df = pd.read_csv(csv_path)
        
        # Generate synthetic thermal features based on the patterns we established
        X = []
        y = []
        
        for _, row in df.iterrows():
            # Create synthetic features based on our analysis logic
            regularity = row['Regularity_Score']
            tensile = row['Tensile_Strength_MPa']
            elongation = row['Elongation_at_Break_Percent']
            abrasion = row['Abrasion_Resistance_Cycles']
            
            # Synthetic feature generation based on correlations
            mean_temp = 255 - (tensile / 90) * 200  # Higher tensile = cooler colors
            std_temp = (1 - regularity) * 50  # Lower regularity = higher variation
            temp_range = std_temp * 4
            min_temp = mean_temp - temp_range/2
            max_temp = mean_temp + temp_range/2
            
            # Color ratios based on properties
            blue_ratio = (tensile / 90) * 0.4  # More blue = higher tensile
            red_ratio = (elongation / 16) * 0.4  # More red = higher elongation
            green_ratio = (abrasion / 1300) * 0.4  # More green = higher abrasion
            yellow_ratio = max(0, 1 - blue_ratio - red_ratio - green_ratio)
            
            gradient_magnitude = (1 - regularity) * 30
            uniformity_score = regularity
            hotspot_count = int((1 - regularity) * 10)
            
            features = [
                mean_temp, std_temp, min_temp, max_temp, temp_range,
                blue_ratio, green_ratio, red_ratio, yellow_ratio,
                gradient_magnitude, uniformity_score, hotspot_count
            ]
            
            X.append(features)
            y.append([regularity, tensile, elongation, abrasion])
        
        return np.array(X), np.array(y)
    
    def train_model(self, csv_path):
        """Train the multi-output regression model"""
        print("Creating training data...")
        X, y = self.create_training_data(csv_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multi-output model
        print("Training model...")
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Mean Absolute Error: {mae:.3f}")
        print(f"R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        for i, importance in enumerate(feature_importance):
            print(f"{self.feature_names[i]}: {importance:.3f}")
    
    def predict_properties(self, image_path_or_features):
        """Predict material properties from thermal image or features"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if isinstance(image_path_or_features, str) or isinstance(image_path_or_features, np.ndarray):
            # Extract features from image
            features = self.extract_thermal_features(image_path_or_features).reshape(1, -1)
        else:
            features = np.array(image_path_or_features).reshape(1, -1)
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)[0]
        
        return {
            'regularity_score': round(predictions[0], 3),
            'tensile_strength_mpa': round(predictions[1], 1),
            'elongation_at_break_percent': round(predictions[2], 1),
            'abrasion_resistance_cycles': int(predictions[3])
        }
    
    def save_model(self, model_path='thermal_model.pkl'):
        """Save trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='thermal_model.pkl'):
        """Load trained model and scaler"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {model_path}")

# Training script
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ThermalImageAnalyzer()
    
    # Train model (assumes thermal_materials_dataset.csv exists)
    analyzer.train_model('thermal_materials_dataset.csv')
    
    # Save model
    analyzer.save_model('thermal_model.pkl')
    
    print("\nModel training complete! You can now use it for predictions.")
    
    # Example usage
    print("\nExample prediction (using synthetic features):")
    sample_features = [150, 25, 100, 200, 100, 0.3, 0.2, 0.4, 0.1, 15, 0.8, 2]
    result = analyzer.predict_properties(sample_features)
    print(result)