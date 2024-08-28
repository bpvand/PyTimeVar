import numpy as np
import matplotlib.pyplot as plt
from timevarmodels import LocalLinearRegression
###########################################################################################################################
# Generate data for testing
###########################################################################################################################

def generate_ar1_model(num_points, phi, sigma):
    # Initialize the time series array
    time_series = [0]

    # Generate the AR1 model time series
    for _ in range(1, num_points):
        epsilon = np.random.normal(0, sigma)  # Generate random noise
        x_t = phi * time_series[-1] + epsilon  # AR1 model equation
        time_series.append(x_t)

    return np.array(time_series)

def create_random_dataset(num_points):
    x = generate_ar1_model(num_points, 0.9, sigma = 2**-2)
    u = generate_ar1_model(num_points, 0.8, sigma  = 4**-2)  # Generate random noise
    t = np.linspace(0,1, num_points)  # Generate time series between 0 and 1

    beta_0 = 0.2 * np.exp(-0.7 + 3.5*t)
    beta_1 = 2*t + np.exp(-16*(t-0.5)**2)-1
  
    y = beta_0 + beta_1* x + u 
    return x , y, beta_0, beta_1
    
#####################
# Test Use Case       
#####################
if __name__ == "__main__":
  np.random.seed(42)
  x, y, b0, b1 = create_random_dataset(200)
  # plt.subplot(2, 2, 1)
  # plt.plot(b0)
  # plt.plot(y)
  # plt.subplot(2, 2, 2)
  # plt.plot(b1)
  mX = np.column_stack((np.ones(len(x)), x))
  model = LocalLinearRegression(y, mX, h=0.275)
  res = model.fit()
  beta0, beta1 = res.betas()
  # plt.subplot(2, 2, 3)
  # plt.plot(b0, label='Real b0')
  # plt.plot(beta0, label='Estimated beta0')
  # plt.legend()
  # plt.subplot(2, 2, 4)
  # plt.plot(b1, label='Real b1')
  # plt.plot(beta1, label='Estimated beta1')
  # plt.legend()
  # plt.title('Local Linear Regression')
  # plt.tight_layout()
  # plt.show()
  res.plot_betas(date_range=("2020-01-01","2020-12-23"))
  res.plot_confidence_bands(Gsubs=[(100, 200), (0, 50)], date_range=("2020-01-01","2020-12-23"))



    
