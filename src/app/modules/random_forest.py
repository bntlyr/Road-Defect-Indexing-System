import pickle
import os

# Path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'random_forest_model.pkl')

# Try to load the trained Random Forest model, else use a dummy model
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    MODEL_AVAILABLE = True
except FileNotFoundError:
    MODEL_AVAILABLE = False
    model = None

    # Dummy model for fallback
    class DummyModel:
        def predict_proba(self, features):
            # Always return a fixed probability for demonstration
            return [[0.3, 0.7]]
    model = DummyModel()

def predict_repair_probability(road_volume, defect_ratio, severity_level, threshold=0.5):
    """
    Predicts the probability of repair for a road defect.

    Parameters:
        road_volume (float): The volume of traffic on the road.
        defect_ratio (float): The ratio of defect area to total area.
        severity_level (float): The severity level of the defect.
        threshold (float): Probability threshold for repair decision.

    Returns:
        int: 1 if repair is needed, 0 otherwise.
    """
    features = [[road_volume, defect_ratio, severity_level]]
    probability = model.predict_proba(features)[0][1]  # Assuming class 1 is 'repair'
    return int(probability >= threshold)

