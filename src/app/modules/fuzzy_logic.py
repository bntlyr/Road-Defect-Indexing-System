import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Vehicle Damage Risk Subsystem (unchanged) ---
vdr_length      = ctrl.Antecedent(np.arange(0, 101, 1), 'vdr_length')  # length in cm
vdr_width       = ctrl.Antecedent(np.arange(0, 101, 1), 'vdr_width')
vehicle_risk    = ctrl.Consequent(np.arange(0, 101, 1), 'vehicle_risk')

# Membership functions for length
vdr_length['short']   = fuzz.trimf(vdr_length.universe, [0, 0, 50])
vdr_length['medium']  = fuzz.trimf(vdr_length.universe, [25, 50, 75])
vdr_length['long']    = fuzz.trimf(vdr_length.universe, [50, 100, 100])

# Membership functions for width
vdr_width['narrow'] = fuzz.trimf(vdr_width.universe, [0, 0, 50])
vdr_width['medium'] = fuzz.trimf(vdr_width.universe, [25, 50, 75])
vdr_width['wide']   = fuzz.trimf(vdr_width.universe, [50, 100, 100])

# Output membership for vehicle damage risk
vehicle_risk['low']    = fuzz.trimf(vehicle_risk.universe, [0, 0, 50])
vehicle_risk['medium'] = fuzz.trimf(vehicle_risk.universe, [25, 50, 75])
vehicle_risk['high']   = fuzz.trimf(vehicle_risk.universe, [50, 100, 100])

# Rules for vehicle damage risk
damage_rules = [
    ctrl.Rule(vdr_length['long'] | vdr_width['wide'],        vehicle_risk['high']),
    ctrl.Rule(vdr_length['medium'] & vdr_width['medium'],    vehicle_risk['medium']),
    ctrl.Rule(vdr_length['short'] & vdr_width['narrow'],     vehicle_risk['low']),
]
vdr_ctrl = ctrl.ControlSystem(damage_rules)
vdr_sim  = ctrl.ControlSystemSimulation(vdr_ctrl)

# --- Main Severity Subsystem ---
# Inputs: length, width, defect_ratio (0-1), traffic, vehicle_risk

# Use the same membership functions for det_length and det_width as vdr_length and vdr_width
det_length = vdr_length
det_width = vdr_width

defect_ratio = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'defect_ratio')
traffic = ctrl.Antecedent(np.arange(0, 10001, 1), 'traffic')
vehicle_risk_input = ctrl.Antecedent(np.arange(0, 101, 1), 'vehicle_risk_input')
severity = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'severity')

# Defect ratio memberships
# low ratio: up to 0.1 (10%), medium: around 0.1-0.3, high: >0.3
defect_ratio['low']    = fuzz.trimf(defect_ratio.universe, [0.0, 0.0, 0.1])
defect_ratio['medium'] = fuzz.trimf(defect_ratio.universe, [0.05, 0.2, 0.35])
defect_ratio['high']   = fuzz.trimf(defect_ratio.universe, [0.3, 1.0, 1.0])

# Traffic memberships (unchanged)
traffic['low']    = fuzz.trimf(traffic.universe, [0, 0, 5000])
traffic['medium'] = fuzz.trimf(traffic.universe, [2000, 5000, 8000])
traffic['high']   = fuzz.trimf(traffic.universe, [5000, 10000, 10000])

# Severity memberships
severity['low']      = fuzz.trimf(severity.universe, [0.0, 0.0, 0.25])
severity['moderate'] = fuzz.trimf(severity.universe, [0.2, 0.4, 0.6])
severity['high']     = fuzz.trimf(severity.universe, [0.5, 0.7, 0.9])
severity['critical'] = fuzz.trimf(severity.universe, [0.8, 1.0, 1.0])

vehicle_risk_input['low']    = fuzz.trimf(vehicle_risk_input.universe, [0, 0, 50])
vehicle_risk_input['medium'] = fuzz.trimf(vehicle_risk_input.universe, [25, 50, 75])
vehicle_risk_input['high']   = fuzz.trimf(vehicle_risk_input.universe, [50, 100, 100])

# Define severity rules incorporating defect_ratio
severity_rules = [
    ctrl.Rule(defect_ratio['high'] & (det_length['long'] | det_width['wide']), severity['high']),
    ctrl.Rule(defect_ratio['high'] & traffic['high'] & vehicle_risk_input['high'], severity['critical']),
    ctrl.Rule(defect_ratio['medium'] & det_length['medium'], severity['moderate']),
    ctrl.Rule(defect_ratio['low'] & det_length['short'] & det_width['narrow'], severity['low']),
]
sev_ctrl = ctrl.ControlSystem(severity_rules)
sev_sim  = ctrl.ControlSystemSimulation(sev_ctrl)

# Example evaluation
inputs = {'length': 60, 'width': 40, 'defect_ratio': 0.25, 'traffic': 6000}
det_len, det_wid = inputs['length'], inputs['width']
vdr_sim.input['vdr_length'] = det_len
vdr_sim.input['vdr_width']  = det_wid
vdr_sim.compute()
vr = vdr_sim.output['vehicle_risk']

# The input variable names for sev_sim must match the Antecedent names: 
# 'vdr_length', 'vdr_width', 'defect_ratio', 'traffic', 'vehicle_risk_input'
sev_sim.input['vdr_length'] = det_len
sev_sim.input['vdr_width'] = det_wid
sev_sim.input['defect_ratio'] = inputs['defect_ratio']
sev_sim.input['traffic'] = inputs['traffic']
sev_sim.input['vehicle_risk_input'] = vr
sev_sim.compute()
sev = sev_sim.output['severity']

print(f"Vehicle damage risk: {vr:.1f}%")
print(f"Severity score (0–1): {sev:.3f}")

# Example defect type weights (customize as needed)
defect_weights = {
    'pothole': 1.2,
    'crack': 1.0,
    'rut': 1.1,
    'patch': 0.9,
    # Add more defect types and their weights as needed
}

def calculate_severity_percentage(length_cm, width_cm, defect_ratio, defect_type: str) -> float:
    """
    Calculate the weighted severity percentage for a defect using fuzzy logic.
    Returns a value between 0 and 100.
    """
    # Compute vehicle risk
    vdr_sim.input['vdr_length'] = length_cm
    vdr_sim.input['vdr_width'] = width_cm
    vdr_sim.compute()
    vehicle_risk_val = vdr_sim.output['vehicle_risk']

    # Compute severity
    sev_sim.input['vdr_length'] = length_cm
    sev_sim.input['vdr_width'] = width_cm
    sev_sim.input['defect_ratio'] = defect_ratio
    sev_sim.input['vehicle_risk_input'] = vehicle_risk_val
    # Note: traffic is not used in calculate_severity_percentage, add as needed
    sev_sim.compute()
    base_severity = sev_sim.output['severity']

    # Apply defect-type weight and clamp to [0,1]
    weight = defect_weights.get(defect_type, 1.0)
    weighted_severity = min(base_severity * weight, 1.0)

    # Return as percentage
    return weighted_severity
