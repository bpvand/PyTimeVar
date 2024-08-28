'''

Section 3: illustration of code on Temperature dataset

'''
# Download data
from PyTimeVar.datasets import temperature
import numpy as np
data = temperature.load(regions=['World'],start_date='1961-01-01', end_date='2023-01-01')
vY = data.values
X = np.ones_like(vY)

# illustrate LLR
from PyTimeVar import LocalLinear
model = LocalLinear(vY, X)
res = model.fit()

# print summary
res.summary()

# get betas
res.betas()

# plot trend and data
res.plot_actual_vs_predicted(date_range=["1980-01-01","2000-01-01"])

# plot confidence bands using LBWB
cb = res.plot_confidence_bands(bootstrap_type='LBWB', date_range=['1980-01-01', '2000-01-01'], Gsubs=None)

# illustrate boosted HP filter
from PyTimeVar import BoostedHP
bHPmodel = BoostedHP(vY, dLambda=1600, iMaxIter=100)
bHPtrend, bHPresiduals = bHPmodel.fit(boost=True, stop="adf", dAlpha=0.05, verbose=False)
bHPmodel.summary()
bHPmodel.plot()

from PyTimeVar import Kalman
kalmanmodel = Kalman(vY=vY)
smooth_trend = kalmanmodel.fit('smoother')
kalmanmodel.summary()
kalmanmodel.plot()

from PyTimeVar import GAS
gasmodel = GAS(vY, X, 'student')
tGAStrend, tGASparams = gasmodel.fit()
gasmodel.plot(date_range=['1980-01-01', '2000-01-01'])
