import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class NormalGASModel:
  """
  A class representing a Normal GAS (Generalized Autoregressive Score) model.

  Attributes:
    data (numpy.ndarray): The input data used for fitting the model.
    simple_returns (numpy.ndarray): The computed simple returns from the input data.
    n (int): The length of the simple returns.
    optimal_params (numpy.ndarray): The optimal parameters obtained from model fitting.

  Methods:
    compute_simple_returns: Computes the simple returns from the input data.
    update: Updates the location and scale parameters based on the GAS algorithm.
    log_likelihood: Computes the log-likelihood of the model.
    initialize_arrays: Initializes the arrays for storing the location, scale, and densities.
    objective_function: Computes the objective function to be minimized during model fitting.
    fit: Fits the model to the data.
    run: Runs the fitted model and returns the updated locations and scales.
  
  References:
  Based on the following youtube video:
  https://www.youtube.com/watch?v=nEqkHJE9vOw
  """

  def __init__(self, data):
    self.data = data
    self.simple_returns = self.compute_simple_returns(data)
    self.n = len(self.simple_returns)
    self.optimal_params = None

  @staticmethod
  def compute_simple_returns(data):
    """
    Computes the simple returns from the input data.

    Args:
      data (numpy.ndarray): The input data.

    Returns:
      numpy.ndarray: The computed simple returns.
    """
    return (data[1:] / data[:-1]) - 1

  def update(self, y, location, scale, adjustment_location, adjustment_scale):
    """
    Updates the location and scale parameters based on the GAS algorithm.

    Args:
      y (float): The observed value.
      location (float): The current location parameter.
      scale (float): The current scale parameter.
      adjustment_location (float): The adjustment factor for the location parameter.
      adjustment_scale (float): The adjustment factor for the scale parameter.

    Returns:
      tuple: The updated location, scale, and density values.
    """
    density = norm.pdf(y, location, scale)
    gradient_location = density * (y - location) / scale**2
    gradient_scale = ((density * (y - location)**2) / scale**3) - density / scale
    gradient_location /= 1000000
    gradient_scale /= 1000000
    new_location = location + adjustment_location * gradient_location
    new_scale = scale + adjustment_scale * gradient_scale
    return new_location, new_scale, density

  def log_likelihood(self, densities):
    """
    Computes the log-likelihood of the model.

    Args:
      densities (numpy.ndarray): The densities of the observed values.

    Returns:
      float: The log-likelihood value.
    """
    log_densities = np.where(densities == 0, -1000, np.log(densities))
    return -np.sum(log_densities)

  def initialize_arrays(self):
    """
    Initializes the arrays for storing the location, scale, and densities.

    Returns:
      tuple: The initialized arrays for locations, scales, and densities.
    """
    locations = np.zeros(self.n + 1)
    scales = np.zeros(self.n + 1)
    densities = np.zeros(self.n)
    return locations, scales, densities

  def objective_function(self, params):
    """
    Computes the objective function to be minimized during model fitting.

    Args:
      params (list): The parameters to be optimized.

    Returns:
      float: The value of the objective function.
    """
    location, scale, adjustment_location, adjustment_scale = params
    locations, scales, densities = self.initialize_arrays()
    locations[0], scales[0] = location, scale

    for i in range(1, self.n + 1):
      locations[i], scales[i], densities[i-1] = self.update(self.simple_returns[i-1], locations[i-1], scales[i-1], adjustment_location, adjustment_scale)
    
    return self.log_likelihood(densities)

  def fit(self):
    """
    Fits the model to the data.
    """
    initial_location = 0
    initial_scale = 0.01
    init_adjustment_location = 0
    init_adjustment_scale = 0
    initial_guess = [initial_location, initial_scale, init_adjustment_location, init_adjustment_scale]
    bounds = [(None, None), (1e-6, None), (None, None), (None, None)]

    result = minimize(self.objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

    if result.success:
      self.optimal_params = result.x
    else:
      raise ValueError("Optimization failed")

  def run(self):
    """
    Runs the fitted model and returns the updated locations and scales.

    Returns:
      tuple: The updated locations and scales.
    """
    if self.optimal_params is None:
      raise ValueError("Model is not fitted yet. Please run the 'fit' method first.")

    location, scale, adjustment_location, adjustment_scale = self.optimal_params
    locations, scales, densities = self.initialize_arrays()
    locations[0], scales[0] = location, scale

    for i in range(1, self.n + 1):
      locations[i], scales[i], densities[i-1] = self.update(self.simple_returns[i-1], locations[i-1], scales[i-1], adjustment_location, adjustment_scale)

    return locations, scales

if __name__ == '__main__':
    from ..datasets.gold.data import load

    gold_data = load(currencies=['USD'], start_date='2010-01-01', end_date='2010-12-31')
    usd_prices = gold_data['USD'].values
    log_returns = np.diff(np.log(usd_prices))
    model = NormalGASModel(log_returns)
    model.fit()

    locations, scales = model.run()

    print("Optimal Parameters:", model.optimal_params)
    print("Time-Varying Locations:", locations)
    print("Time-Varying Scales:", scales)
    import matplotlib.pyplot as plt

    plt.plot(log_returns, label='Log Returns')
    plt.plot(locations, label='Time-Varying Locations')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

