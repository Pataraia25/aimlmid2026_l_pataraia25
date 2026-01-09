import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Input coordinate values
x_values = np.array([-8.70, -6.40, -3.90, -1.60, 0.90, 3.20, 5.50, 8.10, 9.60])
y_values = np.array([-6.20, -4.50, -2.70, -0.40, 1.70, 3.50, 5.30, 7.60, 8.90])

# Total number of data points
num_points = len(x_values)


# Calculate arithmetic means
mean_x = x_values.mean()
mean_y = y_values.mean()

print("Mean of x (x̄):", mean_x)
print("Mean of y (ȳ):", mean_y)


# Compute deviation of each value from its mean
x_diff = x_values - mean_x
y_diff = y_values - mean_y


# Element-wise multiplication of deviations
deviation_product = x_diff * y_diff

# Squared deviations
x_diff_squared = np.power(x_diff, 2)
y_diff_squared = np.power(y_diff, 2)


# Organize intermediate values into a DataFrame
calculation_table = pd.DataFrame({
    "x_i": x_values,
    "y_i": y_values,
    "x_i - x̄": x_diff,
    "y_i - ȳ": y_diff,
    "(x_i - x̄)(y_i - ȳ)": deviation_product,
    "(x_i - x̄)^2": x_diff_squared,
    "(y_i - ȳ)^2": y_diff_squared
})

print("\nManual Calculation Table:\n")
print(calculation_table)

# Compute numerator and denominator separately
sum_product = deviation_product.sum()
sum_x_squared = x_diff_squared.sum()
sum_y_squared = y_diff_squared.sum()

correlation = sum_product / np.sqrt(sum_x_squared * sum_y_squared)

print("\nΣ(x_i - x̄)(y_i - ȳ) =", sum_product)
print("Σ(x_i - x̄)^2 =", sum_x_squared)
print("Σ(y_i - ȳ)^2 =", sum_y_squared)
print("\nPearson Correlation Coefficient (r) =", correlation)

# Initialize plot
plt.figure()

# Scatter plot of original points
plt.scatter(x_values, y_values)

# Calculate regression line parameters
slope, intercept = np.polyfit(x_values, y_values, 1)

# Plot regression line
plt.plot(x_values, slope * x_values + intercept)

# Axis labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot with Line of Best Fit")

# Render plot
plt.show()
