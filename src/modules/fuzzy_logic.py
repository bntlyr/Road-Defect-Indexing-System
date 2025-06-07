import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging

# --- Vehicle Damage Risk Subsystem ---
vdr_length = ctrl.Antecedent(np.arange(0, 101, 1), 'vdr_length')  # length in cm
vdr_width = ctrl.Antecedent(np.arange(0, 101, 1), 'vdr_width')
vehicle_risk = ctrl.Consequent(np.arange(0, 101, 1), 'vehicle_risk')

# Membership functions for length - adjusted for better coverage
vdr_length['short'] = fuzz.trimf(vdr_length.universe, [0, 20, 40])
vdr_length['medium'] = fuzz.trimf(vdr_length.universe, [30, 50, 70])
vdr_length['long'] = fuzz.trimf(vdr_length.universe, [60, 80, 100])

# Membership functions for width - adjusted for better coverage
vdr_width['narrow'] = fuzz.trimf(vdr_width.universe, [0, 20, 40])
vdr_width['medium'] = fuzz.trimf(vdr_width.universe, [30, 50, 70])
vdr_width['wide'] = fuzz.trimf(vdr_width.universe, [60, 80, 100])

# Output membership for vehicle damage risk
vehicle_risk['low'] = fuzz.trimf(vehicle_risk.universe, [0, 20, 40])
vehicle_risk['medium'] = fuzz.trimf(vehicle_risk.universe, [30, 50, 70])
vehicle_risk['high'] = fuzz.trimf(vehicle_risk.universe, [60, 80, 100])

# Rules for vehicle damage risk - simplified and more inclusive
damage_rules = [
    ctrl.Rule(vdr_length['long'] | vdr_width['wide'], vehicle_risk['high']),
    ctrl.Rule(vdr_length['medium'] | vdr_width['medium'], vehicle_risk['medium']),
    ctrl.Rule(vdr_length['short'] | vdr_width['narrow'], vehicle_risk['low']),
    # Add catch-all rules
    ctrl.Rule(vdr_length['short'] & vdr_width['wide'], vehicle_risk['medium']),
    ctrl.Rule(vdr_length['long'] & vdr_width['narrow'], vehicle_risk['medium']),
]

# --- Main Severity Subsystem ---
# Use the same membership functions for det_length and det_width
det_length = vdr_length
det_width = vdr_width

defect_ratio = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'defect_ratio')
traffic = ctrl.Antecedent(np.arange(0, 10001, 1), 'traffic')
vehicle_risk_input = ctrl.Antecedent(np.arange(0, 101, 1), 'vehicle_risk_input')
severity = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'severity')

# Defect ratio memberships - adjusted for better coverage
defect_ratio['low'] = fuzz.trimf(defect_ratio.universe, [0, 0.2, 0.4])
defect_ratio['medium'] = fuzz.trimf(defect_ratio.universe, [0.3, 0.5, 0.7])
defect_ratio['high'] = fuzz.trimf(defect_ratio.universe, [0.6, 0.8, 1.0])

# Traffic memberships - adjusted for better coverage
traffic['low'] = fuzz.trimf(traffic.universe, [0, 2000, 4000])
traffic['medium'] = fuzz.trimf(traffic.universe, [3000, 5000, 7000])
traffic['high'] = fuzz.trimf(traffic.universe, [6000, 8000, 10000])

# Severity memberships - adjusted for better coverage
severity['low'] = fuzz.trimf(severity.universe, [0, 0.2, 0.4])
severity['moderate'] = fuzz.trimf(severity.universe, [0.3, 0.5, 0.7])
severity['high'] = fuzz.trimf(severity.universe, [0.6, 0.8, 0.9])
severity['critical'] = fuzz.trimf(severity.universe, [0.8, 0.9, 1.0])

# Vehicle risk input memberships - adjusted for better coverage
vehicle_risk_input['low'] = fuzz.trimf(vehicle_risk_input.universe, [0, 20, 40])
vehicle_risk_input['medium'] = fuzz.trimf(vehicle_risk_input.universe, [30, 50, 70])
vehicle_risk_input['high'] = fuzz.trimf(vehicle_risk_input.universe, [60, 80, 100])

# Define severity rules - simplified and more inclusive
severity_rules = [
    # Basic rules
    ctrl.Rule(defect_ratio['high'] & (det_length['long'] | det_width['wide']), severity['high']),
    ctrl.Rule(defect_ratio['medium'] & (det_length['medium'] | det_width['medium']), severity['moderate']),
    ctrl.Rule(defect_ratio['low'] & (det_length['short'] | det_width['narrow']), severity['low']),
    
    # Traffic impact rules
    ctrl.Rule(defect_ratio['high'] & traffic['high'], severity['critical']),
    ctrl.Rule(defect_ratio['medium'] & traffic['high'], severity['high']),
    ctrl.Rule(defect_ratio['low'] & traffic['high'], severity['moderate']),
    
    # Vehicle risk impact rules
    ctrl.Rule(vehicle_risk_input['high'] & defect_ratio['medium'], severity['high']),
    ctrl.Rule(vehicle_risk_input['medium'] & defect_ratio['medium'], severity['moderate']),
    ctrl.Rule(vehicle_risk_input['low'] & defect_ratio['medium'], severity['low']),
    
    # Catch-all rules
    ctrl.Rule(defect_ratio['high'], severity['high']),
    ctrl.Rule(defect_ratio['medium'], severity['moderate']),
    ctrl.Rule(defect_ratio['low'], severity['low']),
]

# Create control systems
vdr_ctrl = ctrl.ControlSystem(damage_rules)
sev_ctrl = ctrl.ControlSystem(severity_rules)
vdr_sim = ctrl.ControlSystemSimulation(vdr_ctrl)
sev_sim = ctrl.ControlSystemSimulation(sev_ctrl)

# Example defect type weights (customize as needed)
defect_weights = {
    'Pothole': 1.2,
    'Alligator-Cracks': 1.0,
    'Linear-Cracks': 0.9,
}

def calculate_severity_percentage(length_cm, width_cm, defect_ratio, defect_type: str) -> float:
    """
    Calculate the weighted severity percentage for a defect using fuzzy logic.
    Returns a value between 0 and 100.
    """
    try:
        # Ensure inputs are within valid ranges
        length_cm = max(0, min(100, float(length_cm)))
        width_cm = max(0, min(100, float(width_cm)))
        defect_ratio = max(0, min(1.0, float(defect_ratio)))
        
        # Compute vehicle risk
        vdr_sim.input['vdr_length'] = length_cm
        vdr_sim.input['vdr_width'] = width_cm
        vdr_sim.compute()
        vehicle_risk_val = vdr_sim.output['vehicle_risk']
        
        # Compute severity with all inputs
        sev_sim.input['vdr_length'] = length_cm
        sev_sim.input['vdr_width'] = width_cm
        sev_sim.input['defect_ratio'] = defect_ratio
        sev_sim.input['vehicle_risk_input'] = vehicle_risk_val
        sev_sim.input['traffic'] = 5000.0  # Default to medium traffic
        sev_sim.compute()
        
        # Get base severity and apply defect type weight
        base_severity = sev_sim.output['severity']
        weight = defect_weights.get(defect_type, 1.0)
        weighted_severity = min(base_severity * weight, 1.0)
        
        return weighted_severity * 100  # Convert to percentage
        
    except Exception as e:
        # Fallback calculation if fuzzy logic fails
        logging.warning(f"Fuzzy logic calculation failed, using fallback: {e}")
        return min(100, (defect_ratio * 100) * defect_weights.get(defect_type, 1.0))
