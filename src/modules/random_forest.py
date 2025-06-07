import pickle
import os
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def get_model_path():
    """Get the path to the random forest model file."""
    # Try multiple possible locations for the model file
    possible_paths = [
        # If running as executable
        os.path.join(getattr(sys, '_MEIPASS', ''), 'models', 'random_forest_model.pkl'),
        # If running in development
        os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'random_forest_model.pkl'),
        # If running from root directory
        os.path.join('models', 'random_forest_model.pkl'),
        # If running from src directory
        os.path.join('..', 'models', 'random_forest_model.pkl')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found random forest model at: {path}")
            return path
            
    logger.warning("Random forest model not found in any of the expected locations")
    return None

# Initialize model variables
model = None
MODEL_AVAILABLE = False

def initialize_model():
    """Initialize the random forest model."""
    global model, MODEL_AVAILABLE
    
    try:
        model_path = get_model_path()
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            MODEL_AVAILABLE = True
            logger.info("Random forest model loaded successfully")
        else:
            logger.warning("Random forest model file not found, using fallback model")
            model = FallbackModel()
            MODEL_AVAILABLE = False
    except Exception as e:
        logger.error(f"Error loading random forest model: {e}")
        model = FallbackModel()
        MODEL_AVAILABLE = False

class FallbackModel:
    """Fallback model that uses a simple rule-based approach."""
    def predict_proba(self, features):
        road_volume, defect_ratio, severity_level = features[0]
        
        # Simple rule-based probability calculation
        prob = 0.0
        
        # Base probability from severity
        prob += severity_level * 0.4
        
        # Add contribution from defect ratio
        prob += min(defect_ratio * 2.0, 0.3)
        
        # Add contribution from traffic volume (normalized to 0-1)
        traffic_factor = min(road_volume / 10000.0, 1.0)
        prob += traffic_factor * 0.3
        
        # Ensure probability is between 0 and 1
        prob = min(max(prob, 0.0), 1.0)
        
        return [[1 - prob, prob]]  # Return [no repair prob, repair prob]

def predict_repair_probability(road_volume, defect_ratio, severity_level, threshold=0.4):
    """
    Predicts the probability of repair for a road defect.
    
    Parameters:
        road_volume (float): The volume of traffic on the road.
        defect_ratio (float): The ratio of defect area to total area.
        severity_level (float): The severity level of the defect (0-1).
        threshold (float): Probability threshold for repair decision.
    
    Returns:
        int: 1 if repair is needed, 0 otherwise.
    """
    global model, MODEL_AVAILABLE
    
    # Initialize model if not already done
    if model is None:
        initialize_model()
    
    try:
        # Ensure inputs are within valid ranges
        road_volume = max(0.0, min(float(road_volume), 10000.0))
        defect_ratio = max(0.0, min(float(defect_ratio), 1.0))
        severity_level = max(0.0, min(float(severity_level), 1.0))
        
        # Prepare features
        features = [[road_volume, defect_ratio, severity_level]]
        
        # Get prediction
        probability = model.predict_proba(features)[0][1]  # Probability of repair
        logger.info(f"Random forest prediction - Repair probability: {probability:.2f}, Threshold: {threshold}")
        
        return int(probability >= threshold)
        
    except Exception as e:
        logger.error(f"Error in predict_repair_probability: {e}")
        # Fallback to simple rule
        return 1 if (severity_level > 0.3 or defect_ratio > 0.1) else 0

# Initialize the model when the module is imported
initialize_model()

