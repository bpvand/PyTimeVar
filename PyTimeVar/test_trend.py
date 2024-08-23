from timevarmodels.datasets import gold, usd, temperature, co2_panel
from timevarmodels import BoostedHP, LocalLinearRegression, Kalman, tGAS
import numpy as np
import matplotlib.pyplot as plt

# download data
gold_data = gold.load(currencies=['USD'], start_date='2000-01-01', end_date='2023-12-31')
vY_gold = gold_data.values
X_gold = np.ones_like(vY_gold)

usd_data = usd.load(type='Close', start_date='2010-01-01', end_date='2023-12-31')
vY_usd = usd_data.values
X_usd = np.ones_like(vY_usd)

temp_data = temperature.load(regions=['World'],start_date='1961-01-01', end_date='2023-12-31')
vY_temp = temp_data.values
X_temp = np.ones_like(vY_temp)

co2_data = co2_panel.load(start_date='1900-01-01', end_date='2017-01-01', regions=['AUSTRIA'])
vY_co2 = co2_data.values
X_co2 = np.ones_like(vY_co2)

plt.plot(vY_usd)
plt.show()


# Figure 1
# local linear estimation for each dataset

# LLRModel_gold = LocalLinearRegression(vY_gold, X_gold)
# res_gold = LLRModel_gold.fit()
# cb_gold = res_gold.plot_confidence_bands(bootstrap_type="LBWB")

# LLRModel_usd = LocalLinearRegression(vY_usd, X_usd)
# res_usd = LLRModel_usd.fit()
# cb_usd = res_usd.plot_confidence_bands(bootstrap_type="LBWB")

LLRModel_temp = LocalLinearRegression(vY_temp, X_temp)
res_temp = LLRModel_temp.fit()
cb_temp = res_temp.plot_confidence_bands(bootstrap_type="LBWB")

LLRModel_co2 = LocalLinearRegression(vY_co2, X_co2)
res_co2 = LLRModel_co2.fit()
cb_co2 = res_co2.plot_confidence_bands(bootstrap_type="LBWB")


# Figure 2
# local linear estimation using different kernels. CO2 dataset is used here
kernels = ['epanechnikov', 'gaussian', 'uniform', 'triangular', 'quartic', 'tricube']
x_vals = np.linspace(0, 1, len(vY_co2))
for kernel in kernels:
    llr_model = LocalLinearRegression(vY_co2, X_co2, kernel = kernel)
    res = llr_model.fit()
    plt.plot(x_vals, res.betas()[0], label=kernel)

plt.legend()
plt.title('Temperature: LLR estimates for different kernel functions')
plt.show()


# Figure 3
# LLR and alternative trend estimates. Temperature and CO2 datasets are used here
# Temperature
HPmodel = BoostedHP(vY_temp)
HPtrend, HPresiduals = HPmodel.fit(boost=False)
bHPmodel = BoostedHP(vY_temp.flatten())
bHPtrend, bHPresiduals = bHPmodel.fit()
Q = np.array([[0.03]])
kalmanmodel = Kalman(Q=Q)
smooth_trend = kalmanmodel.smoother(vY_temp)
model = tGAS(vY_temp.flatten(), model_type='local_linear_trend')
result = model.fit()
tGAStrend = model.mu

data = temp_data
plt.plot(data.index, data['World'], label='Original Series')
plt.plot(data.index, HPtrend, label='HP', linestyle='--')
plt.plot(data.index, bHPtrend, label='bHP', linestyle='--')
plt.plot(data.index, res_temp.betas()[0], label='LLR', linestyle='--')
plt.plot(data.index, tGAStrend, label='tGAS', linestyle='--')
plt.plot(data.index, smooth_trend, label='Kalman Smoother', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Temp change (°C)')
plt.title('Temperature change in World')
plt.legend()
plt.show()

# CO2
HPmodel = BoostedHP(vY_co2)
HPtrend, HPresiduals = HPmodel.fit(boost=False)
bHPmodel = BoostedHP(vY_co2.flatten())
bHPtrend, bHPresiduals = bHPmodel.fit()
Q = np.array([[0.03]])
kalmanmodel = Kalman(Q=Q)
smooth_trend = kalmanmodel.smoother(vY_co2)
model = tGAS(vY_co2.flatten(), model_type='local_linear_trend')
result = model.fit()
tGAStrend = model.mu

# Plotting
data = co2_data
plt.plot(data.index, data['AUSTRIA'], label='Original Series')
plt.plot(data.index, HPtrend, label='HP', linestyle='--')
plt.plot(data.index, bHPtrend, label='bHP', linestyle='--')
plt.plot(data.index, res_co2.betas()[0], label='LLR', linestyle='--')
plt.plot(data.index, tGAStrend, label='tGAS', linestyle='--')
plt.plot(data.index, smooth_trend, label='Kalman Smoother', linestyle='--')
plt.xlabel('Date')
plt.ylabel('CO2')
plt.title('CO2 Emissions in Austria')
plt.legend()
plt.show()

# Load data
# data = temperature.load(regions=['World'])
# vY = data['World'].values
# X = np.ones_like(vY)

# # HP filter
# HPmodel = BoostedHP(vY, boost=False)
# HPtrend, HPresiduals = HPmodel.fit()

# # Boosted HP filter
# bHPmodel = BoostedHP(vY)
# bHPtrend, bHPresiduals = bHPmodel.fit()

# # Local Linear Regression filter
# LLRModel = LocalLinearRegression(vY, X, h=0.5795442)
# res = LLRModel.fit()
# LLRtrend = res.betas()[0]  # Extract the trend component

# # Kalman Filter
# Q = np.array([[0.03]])
# kalmanmodel = Kalman(Q=Q)
# filter_trend = kalmanmodel.filter(vY)
# smooth_trend = kalmanmodel.smoother(vY)[:,0]

# # tGAS
# model = tGAS(vY, model_type='local_linear_trend')
# result = model.fit()
# tGAStrend = model.mu


# # Plotting
# plt.plot(data['Date'], data['World'], label='Original Series')
# plt.plot(data['Date'], HPtrend, label='HP', linestyle='--')
# plt.plot(data['Date'], bHPtrend, label='bHP', linestyle='--')
# plt.plot(data['Date'], LLRtrend, label='LLR', linestyle='--')
# plt.plot(data['Date'], tGAStrend, label='tGAS', linestyle='--')
# plt.plot(data['Date'], smooth_trend, label='Kalman Smoother', linestyle='--')
# # plt.plot(data['Date'], smoothed, label='Spline', linestyle='--')
# plt.xlabel('Date')
# plt.ylabel('Temp change (°C)')
# plt.title('Temperature change in World')
# plt.legend()
# plt.show()

