import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Load a sample dataset
data = load_iris ()
X = data.data
y = data.target


def objective(trial):
    # Define the search space for hyperparameters
    C = trial.suggest_float ("C", 1e-7, 10.0, log=True)
    solver = trial.suggest_categorical ("solver", ["liblinear", "saga"])

    # Create a pipeline with scaling and logistic regression
    model = make_pipeline (
        StandardScaler (),
        LogisticRegression (C=C, solver=solver, max_iter=10000)  # Increase max_iter
    )

    # Perform cross-validation and return the mean score
    score = cross_val_score (model, X, y, n_jobs=-1, cv=3).mean ()
    return score


# Create an Optuna study
study = optuna.create_study (direction="maximize")

# Run the optimization
study.optimize (objective, n_trials=100)

# Print the best hyperparameters and score
print (f"Best hyperparameters: {study.best_params}")
print (f"Best score: {study.best_value}")


# Define a plot function
def plot_optimization_history(study):
    fig, ax = plt.subplots (1, 1, figsize=(10, 6))
    ax.plot ([trial.number for trial in study.trials], [trial.value for trial in study.trials], marker="o")
    ax.set_xlabel ("Trial number")
    ax.set_ylabel ("Objective value")
    ax.set_title ("Optimization history")
    plt.show ()


# Plot the optimization history
plot_optimization_history (study)
