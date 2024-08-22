from timevarmodels.datasets import temperature
from timevarmodels import BoostedHP, LocalLinearRegression, Kalman, tGAS
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Load data
data = temperature.load(regions=['World'])
vY = data['World'].values
X = np.ones_like(vY)

# HP filter
HPmodel = BoostedHP(vY, iMaxIter=1)
HPtrend, HPresiduals = HPmodel.fit()

# Boosted HP filter
bHPmodel = BoostedHP(vY)
bHPtrend, bHPresiduals = bHPmodel.fit()

# Local Linear Regression filter
LLRModel = LocalLinearRegression(vY, X, h=0.5795442)
res = LLRModel.fit()
LLRtrend = res.betas()[0]  # Extract the trend component

# Kalman Filter
Q = np.array([[0.03]])
kalmanmodel = Kalman(Q=Q)
filter_trend = kalmanmodel.filter(vY)
smooth_trend = kalmanmodel.smoother(vY)[:,0]

# tGAS
model = tGAS(vY, model_type='local_linear_trend')
result = model.fit()
tGAStrend = model.mu


# Plotting
plt.plot(data['Date'], data['World'], label='Original Series')
plt.plot(data['Date'], HPtrend, label='HP', linestyle='--')
plt.plot(data['Date'], bHPtrend, label='bHP', linestyle='--')
plt.plot(data['Date'], LLRtrend, label='LLR', linestyle='--')
plt.plot(data['Date'], tGAStrend, label='tGAS', linestyle='--')
plt.plot(data['Date'], smooth_trend, label='Kalman Smoother', linestyle='--')
# plt.plot(data['Date'], smoothed, label='Spline', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Temp change (°C)')
plt.title('Temperature change in World')
plt.legend()
plt.show()

