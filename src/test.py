#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Step 1: Define a simple prior for synthetic data generation
def generate_synthetic_data(n_samples=1000, n_features=10, noise=0.1):
    """ Generate synthetic tabular data based on a simple linear regression model. """
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
    return X, y

# Step 2: Create synthetic datasets from the prior
X_synthetic, y_synthetic = generate_synthetic_data()

# Step 3: Split the synthetic data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

# Step 4: Define a simple neural network to act as the PFN
# In a real PFN, this would be a pre-trained network on many prior-sampled tasks
pfn_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
pfn_model.fit(X_train, y_train)

# Step 5: Evaluate the "PFN" on the test set
y_pred = pfn_model.predict(X_test)

# Step 6: Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('PFN Dummy Example: Predicted vs True')
plt.grid()
plt.show()

# Step 7: Show performance
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Dummy Bayesian Inference Step: Predict on a new random sample (in real PFN, this would involve using posterior distributions)
new_sample = np.random.randn(1, 10)  # Random sample from standard normal
new_prediction = pfn_model.predict(new_sample)
print(f"Prediction for new sample: {new_prediction[0]:.4f}")

# %%
