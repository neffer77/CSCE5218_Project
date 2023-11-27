import json
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

# Function to optimize
def objective_function(...):
    # Your objective function
    pass

# Callback function to save progress
def save_progress(optimization, filename="progress.json"):
    with open(filename, "w") as f:
        json.dump(optimization.res, f)

# Setup Bayesian Optimization
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds={"param1": (0, 1), "param2": (0, 10)},
    random_state=1,
)

# Run Optimization with progress saving
optimizer.maximize(init_points=2, n_iter=10, acq="ei")
save_progress(optimizer)

# Plotting the results
target_results = [res["target"] for res in optimizer.res]
iterations = range(1, len(target_results) + 1)

plt.figure(figsize=(10, 5))
plt.plot(iterations, target_results, marker='o')
plt.title('Optimization Progress')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.grid(True)
plt.show()
