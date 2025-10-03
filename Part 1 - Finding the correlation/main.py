import math
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for PyCharm
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
from scipy.stats import linregress
import numpy as np  # Add numpy for array operations
import os  # Add os to handle file paths

# Calculates the average of a list or array
def mean(ar):
    return sum(ar) / len(ar)

# Your data arrays
x = [-5, -4, -3, -1, 1, 3, 5, 7]
y = [2, -4, -1, 1, -2, 1, -3, -2]

# Convert to numpy arrays for later use
x_array = np.array(x)
y_array = np.array(y)

# Calculate mean for both arrays
meanx = mean(x)
meany = mean(y)

print(f"Mean of x: {meanx}")
print(f"Mean of y: {meany}\n")

# Calculates the covariance between two arrays
def covariance(ar1, ar2):
    # Covariance measures how two variables change together
    return sum([(ar1[i] - meanx) * (ar2[i] - meany) for i in range(len(ar1))])

# Calculates the standard deviation (sigma) of an array
def sigma(ar):
    # Standard deviation measures the amount of variation or dispersion
    return math.sqrt(sum((ar[i] - mean(ar)) ** 2 for i in range(len(ar))))

# Calculates the Pearson correlation coefficient
def correlation(ar1, ar2):
    # The formula is: covariance(X, Y) / (std_dev(X) * std_dev(Y))
    return covariance(ar1, ar2) / (sigma(ar1) * sigma(ar2))

# Calculate intermediate values
cov_xy = covariance(x, y)
sigma_x = sigma(x)
sigma_y = sigma(y)

print(f"Covariance(x, y): {cov_xy}")
print(f"Standard deviation of x: {sigma_x}")
print(f"Standard deviation of y: {sigma_y}\n")

# Print the final correlation coefficient
r = correlation(x, y)
print(f"Pearson Correlation Coefficient (r) - Manual Calculation: {r}")
print(f"Rounded to 4 decimal places: {r:.4f}")

# Interpretation
print("\n--- Interpretation ---")
if abs(r) >= 0.7:
    strength = "strong"
elif abs(r) >= 0.4:
    strength = "moderate"
else:
    strength = "weak"

if r > 0:
    direction = "positive"
elif r < 0:
    direction = "negative"
else:
    direction = "no"

print(f"There is a {strength} {direction} linear correlation between x and y.")

# --- Automatic Calculation using scipy ---
print("\n" + "="*50)
print("COMPARISON: Manual vs Automatic Calculation")
print("="*50)

# Perform linear regression to get slope and intercept (this also gives us r_value)
slope, intercept, r_value, p_value, std_err = linregress(x, y)

print(f"\nManual Calculation (From Scratch):")
print(f"  Pearson Correlation (r): {r:.6f}")

print(f"\nAutomatic Calculation (scipy.stats.linregress):")
print(f"  Pearson Correlation (r): {r_value:.6f}")

print(f"\nDifference: {abs(r - r_value):.10f}")

if abs(r - r_value) < 1e-10:
    print("✓ Values match perfectly! Our manual calculation is correct.")
else:
    print("⚠ There is a small difference between the calculations.")

print("\n" + "="*50)
print("REGRESSION ANALYSIS")
print("="*50)
print(f"Slope (β1): {slope:.6f}")
print(f"Intercept (β0): {intercept:.6f}")
print(f"Regression Equation: y = {slope:.4f}x + {intercept:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Standard Error: {std_err:.6f}")

# --- Visualization (Linear Regression and Correlation Analysis) ---
def predict(x_val):
    return slope * x_val + intercept

# Generate y-values for the regression line (use numpy array here)
fit_line = predict(x_array)

# Draw the original scatter plot and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data', s=100, alpha=0.7)
plt.plot(x_array, fit_line, color='red', linewidth=2, label='Line of Best Fit')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title(f'Linear Regression Fit (r = {r:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Save figure as PNG in the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "regression_plot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

# Show on screen
plt.show()

print(f"\n--- Regression Analysis ---")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-value (from linregress): {r_value:.4f}")
print(f"Note: r_value from linregress matches our calculated Pearson correlation!")
print(f"Plot saved as: {save_path}")
