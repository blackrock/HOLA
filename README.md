# HOLA: Hyperparameter Optimization, Lightweight Asynchronous

## Introduction
HOLA provides a simple interface for single and multi-objective hyperparameter optimization.
The hyperparameter search-space is specified using a simple python dictionary making it easy to integrate with 
existing code.

## Installation
```bash
pip install git+ssh://git@ssh.dev.azure.com/v3/1A4D/AI%20Labs/hola
```

## Example

Below is an example of using HOLA to optimize the hyperparameters of a supervised machine learning model with some 
training and validation data.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from hola.tune import tune

# Define hyperparameter search space
params_config = {
  "n_estimators": {"min": 10, "max": 1000, "param_type": "int", "scale": "log", "grid": 20},
  "max_depth": {"values": [1, 2, 3, 4]},
  "learning_rate": {"min": 1e-4, "max": 1.0, "scale": "log"},
  "subsample": {"min": 0.2, "max": 1.0},
}

# define objectives
objectives_config = {
  "r_squared": {"target": 1.0, "limit": 0.0, "priority": 2.0},
  "abs_error": {"target": 0, "limit": 1000, "priority": 0.5},
}

# Create the simulation function to evaluate
# Note: the arguments of run should be just the hyperparameter names
X = np.random.randn(500, 10)
Y = np.sum(X[:, :5] * X[:, 3:8], axis=1) + np.sum(X ** 2, axis=1)

Xval = np.random.randn(500, 10)
Yval = np.sum(Xval[:, :5] * Xval[:, 3:8], axis=1) + np.sum(Xval ** 2, axis=1)


def run(n_estimators, max_depth, learning_rate, subsample):
  model = GradientBoostingRegressor(
    n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample
  )
  model.fit(X, Y)
  r2 = model.score(Xval, Yval)
  y_pred = model.predict(Xval)
  ae = np.mean(np.abs(y_pred - Yval))
  return {"r_squared": r2, "abs_error": ae}


# Finally run the hyperparameter tuner
tuner = tune(run, params_config, objectives_config, num_runs=200, n_jobs=1)

print(tuner.get_best_params())
print(tuner.get_best_scores())
```

The leader-board of simulation results can be accessed with
```python
tuner.get_leaderboard()
```

The simulation results can be saved by calling
```python
tuner.save('path/to/simulation_results.csv')
```

A hyper-parameter optimization session can be restored from a leader-board csv by calling
```python
tuner.load('path/to/simulation_results.csv')
```

## Hyperparameter search-space configuration
The hyperparameter search-space is specified using a dictionary.
Each key in the dictionary is the name of a hyperparameter to be optimized.
The values of the dictionary are dictionaries that specify the attributes of the hyperparameter, e.g. the minimum and
maximum allowed value or the scale (linear or logarithmic).
All possible attributes are
```python
params_config = {
  "hyperparameter_name": {
    "min": ..., # Minimum allowed value, required
    "max": ..., # Maximum allowed value, required
    "scale": ..., # Scale of the parameter, can be either "linear" or "log", defaults to "linear" if unspecified
    "param_type": ..., # Type of the parameter, "int" or "float", defaults to "float",
    "grid": ..., # Snap the value of the hyperparameter to a grid of evenly spaced values, must be an integer, unused by default
    "values": ... # List of fixed values the hyperparameter can take, can be a list of anything e.g. ['red','blue','green'], ignores all other attributes if used, unused by default
  },
  ...,
}
```
## Objective configuration
One or more objectives are specified with a python dictionary.
The keys of the dictionary are the objective names.
The values are a dictionary specifying the `target`, `limit`, and `priority` of the objective.
The target is the desired value for the objective.
The limit is the worst-possible value, or worst value that could be accepted, it should not be set too close to the target.
The priority indicates the relative importance between the objectives, defaults to `1`.
An example objective configuration could look like
```python
objective_config = {
  "accuracy": {
    "target": 1.0,
    "limit": 0.0,
    "priority": 2.0
  },
  "abs_error": {
    "target": 0,
    "limit": 100
  }
}
```

## Running hyperparameter optimization using hola.Tuner
`hola.Tuner` lets you run multi-processor hyperparameter optimization on your local machine.
First define your hyperparameter search-space and objectives.
Then create a function with keyword arguments that are the same as the hyperparameter names in the search-space configuration and call the `Tuner.tune` function with your function as an input, e.g.

```python
params_config = {
  "hyper_param_1": {...},
  "hyper_param_2": {...},
  "hyper_param_3": {...}
}

objective_config = {
  "objective_1": {...},
  "objective_2": {...}
}


def my_hyper_function(hyper_param_1, hyper_param_2, hyper_param_3):
  # Use supplied hyperparameters to run your code
  # Then return a dictionary of the resulting objective values
  return {"objective_1": ..., "objective_2": ...}


from hola.tune import tune

tuner = tune(my_hyper_function, params_config, objective_config, num_runs=100, n_jobs=4)
print(tuner.get_best_params())
```

# Using HOLA across machines and/or programming languages
HOLA can be run as a hyperparameter server.
This enables workers to communicate with the server using a simple http interface.
The workers can thus be implemented completely independently of HOLA, and multiple workers can execute simultaneously.

## Running the HOLA server
The HOLA server can be run by executing the installed server script.
By default the script is installed in the same directory as the python binary that was used to install it.
```bash
/path/to/python/bin hola_serve
```
By default the HOLA server will use the current directory to look for configuration files and store hyperparameter 
results.
To use a different directory use the `-d` argument
```bash
/path/to/python/bin hola_serve -d /path/to/desired/directory
```
By default the address of the HOLA server is set to `localhost:8675` and workers will need this address to request 
hyperparameters and report results. To use a different port use the `-p` argument
```bash
/path/to/python/bin hola_serve -d /path/to/desired/directory -p 9988
```
By default HOLA will first sample a certain number of points uniformly at random.
```bash
/path/to/python/bin hola_serve -d /path/to/desired/directory -m 50
```

To change the number of random points use the `-m` or `--min_samples` argument.
Upon start the HOLA server will look for `hola_params.json` and `hola_objectives.json` files in the specified directory.
`hola_params.json` should simply be a key value dictionary containing the hyperparameter search-space configuration, e.g.
```javascript
{
  "hyper_param_1": {...},
  "hyper_param_2": {...},
  "hyper_param_3": {...}
}
```
Similarly, `hola_objectives.json` should be a key-value dictionary containing the objective configuration
```javascript
{
  "objective_1": {...},
  "objective_2": {...}
}
```

The HOLA server will save simulation results in `hola_results.csv`.
If a `hola_results.csv` file is already present in the specified directory, then the HOLA server will load these 
results and resume from where it left off.

Once running, the HOLA server exposes the following routes
* `/`
  * `get`: returns an html page of the current leaderboard
* `/report_request`
  * `get` returns a hyperparameter sample key-value dictionary
  * `post`
    * `request`: The request can be empty or optionally a JSON dictionary with keys `params` and `objectives`. `params` should be a dictionary of hyper-parameter names and values. `objectives` should be a dictionary of objective names and values.
    * `response`: If the request is empty, will simply return a JSON hyper-parameter sample dictionary. If the request is non-empty, the supplied hyper-parameter sample and simulation result will be recorded in the leaderboard and a JSON response with a new hyper-parameter sample will be returned.
* `/param`
 * `get`: Will return a key-value dictionary of the best hyperparameters seen so far.
* `/experiment`
  * `get`: Will return a JSON response dictionary with keys `params` and `objectives` containing the hyper-parameter and objective configuration dictionaries respectively.

## Running workers and requesting hyperparameter samples from the HOLA server

Since the HOLA server exposes an HTTP interface, workers can be implemented in any language that supports HTTP requests.

```python
import requests

HOLA_ADDR = "http://localhost:8675"
URL = f"{HOLA_ADDR}/report_request"

param_sample = requests.get(URL).json()  # Will be a key-value dictionary hyper-parameter sample

...
objectives = ... # Run simulation and get key-value dictionary of objectives

sim_result = {
"params": param_sample,
"objectives": objectives
}

new_param_sample = requests.post(URL, json=sim_result).json()

```

Or you can use the `hola.worker.Worker` class to more conveniently get hyper-parameter sample and report simulation results

```python
from hola.worker import Worker

SERVER_ADDR = "http://localhost"
PORT = 8675

worker = Worker(SERVER_ADDR, PORT)

param_sample = worker.get_param_sample()

...
objectives = ...  # Run simulation and get key-value dictionary of objectives

new_param_sample = worker.report_sim_result(objectives=objectives, params=param_sample)
```
