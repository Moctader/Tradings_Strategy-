from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig
import numpy as np
import matplotlib.pyplot as plt

# Define the Page-Hinkley configuration
config = PageHinkleyConfig()
config.min_instances = 1
config.delta = 0.0005  # Adjust delta for sensitivity
config.lambda_ = 1     # Adjust lambda for sensitivity
config.alpha = 1 - 0.01

# Initialize Page-Hinkley detector
ph_detector = PageHinkley(config=config)

# Drift detection setup
drift_detected = False
drift_points = []

# Print the types and shapes of y_pred and y_test_actual
print("y_pred type:", type(y_pred), "y_pred shape:", y_pred.shape)
print("y_test_actual type:", type(y_test_actual), "y_test_actual shape:", y_test_actual.shape)

# Ensure y_pred and y_test_actual are numpy arrays for proper indexing
y_pred_values = y_pred.values().flatten()  # Reshape to 1D array
y_test_actual_values = y_test_actual.values().flatten()  # Reshape to 1D array

# Correctly concatenate the arrays
stream = np.concatenate((y_pred_values, y_test_actual_values))  # Use tuple for concatenation

# Monitor predictions for drift using Page-Hinkley
for i, value in enumerate(stream):
    ph_detector.update(value=value)  # Update the detector with the error value
    
    if ph_detector.drift:
        drift_detected = True
        # Only append the original index from the stream
        if i < len(y_pred_values):
            drift_points.append(i)  # drift point from y_pred
        else:
            drift_points.append(i - len(y_pred_values))  # drift point from y_test_actual
        print(f"Drift detected at point: {i}")

# Check for drift detection
if drift_detected:
    print(f"Drift detected at points: {drift_points}")
else:
    print("No drift detected.")

# --- Plotting the results ---
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual_values, label="Actual Values", color="blue")
plt.plot(y_pred_values, label="Predicted Values", color="orange")
# Adjusting to access y_test_actual_values correctly
if drift_points:
    plt.scatter(drift_points, y_test_actual_values[drift_points], color="red", label="Drift Points", zorder=5)
plt.xlabel("Data Points")
plt.ylabel("Values")
plt.title("Drift Detection using Page-Hinkley")
plt.legend()
plt.show()
