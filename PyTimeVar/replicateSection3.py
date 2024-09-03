'''

Section 3: illustration of code on Temperature dataset

'''
# Download data
import matplotlib.pyplot as plt
from PyTimeVar import PowerLaw
from PyTimeVar.datasets import herding
from PyTimeVar import GAS
from PyTimeVar import Kalman
from PyTimeVar import BoostedHP
from PyTimeVar import LocalLinear
from PyTimeVar.datasets import temperature
import numpy as np

data = temperature.load(
    regions=['World'], start_date='1961-01-01', end_date='2023-01-01')
vY = data.values
X = np.ones_like(vY)

# illustrate LLR
model = LocalLinear(vY, X)
res = model.fit()

# print summary
res.summary()

# get betas
res.betas()

# plot trend and data
res.plot_actual_vs_predicted(date_range=["1980-01-01", "2000-01-01"])

# plot confidence bands using LBWB
cb = res.plot_confidence_bands(bootstrap_type='LBWB', date_range=[
                               '1980-01-01', '2000-01-01'], Gsubs=None)

# illustrate boosted HP filter
bHPmodel = BoostedHP(vY, dLambda=1600, iMaxIter=100)
bHPtrend, bHPresiduals = bHPmodel.fit(
    boost=True, stop="adf", dAlpha=0.05, verbose=False)
bHPmodel.summary()
bHPmodel.plot()

kalmanmodel = Kalman(vY=vY)
smooth_trend = kalmanmodel.fit('smoother')
kalmanmodel.summary()
kalmanmodel.plot()

gasmodel = GAS(vY, X, 'student')
tGAStrend, tGASparams = gasmodel.fit()
gasmodel.plot(date_range=['1980-01-01', '2000-01-01'])

PwrLaw = PowerLaw(vY, n_powers=3)
pwrTrend = PwrLaw.fit()

plt.plot(pwrTrend, c='r')
plt.plot(vY)
plt.show()


# herd_data = herding.load(start_date='2015-01-05', end_date='2022-01-05')
# vY = herd_data[['CSAD_AVG']].values
# mX = herd_data[['AVG_RTN', 'RTN_ABS', 'RTN_2', 'Intercept']].values

# gasmodel = GAS(vY, mX, 'student')
# tGAStrend, tGASparams = gasmodel.fit()
# gasmodel.plot(date_range=['2015-01-05', '2022-01-05'])

# kalmanmodel = Kalman(vY=vY, regressors=mX)
# smooth_trend = kalmanmodel.fit('smoother')
# kalmanmodel.summary()
# kalmanmodel.plot(date_range=['2015-01-05', '2022-01-05'])
