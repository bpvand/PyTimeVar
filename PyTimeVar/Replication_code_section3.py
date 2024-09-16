'''

Section 3: illustration of code on Temperature dataset

'''
# Load data
from PyTimeVar.datasets import temperature
import numpy as np
data = temperature.load(
    regions=['World'], start_date='1961-01-01', end_date='2023-01-01')
vY = data.values
X = np.ones_like(vY)

from PyTimeVar.datasets import gold
import numpy as np
data = gold.load(
    currencies=['USD'], start_date='2015-01-05', end_date='2016-01-05')
vY = data.values
print(vY.shape)
X = np.ones_like(vY)

# set seed
np.random.seed(123)

# illustrate LLR
# from PyTimeVar import LocalLinear
# model = LocalLinear(vY, X)
# betaHatLLR = model.fit()

# # print summary
# model.summary()

# # plot trend and data
# model.plot_predicted()

# # plot confidence bands using LBWB
# S_LB, S_UB, P_LB, P_UB = model.confidence_bands(bootstrap_type='LBWB', Gsubs=None, plots=True)

# # auxiliary LLR model to illustrate kernel, bandwidth selection, and tau
# tau = np.linspace(0, 0.5, len(vY))
# model2LLR = LocalLinear(vY, X, kernel='Gaussian', bw_selection='lmcv_8', tau=tau)
# beta_hat_model2 = model2LLR.fit()

# illustrate boosted HP filter
from PyTimeVar import BoostedHP
bHPmodel = BoostedHP(vY, dLambda=1600, iMaxIter=100)
bHPtrend, bHPresiduals = bHPmodel.fit(
    boost=True, stop="adf", dAlpha=0.05, verbose=False)
bHPmodel.summary()
bHPmodel.plot()

# illustrate power-law trend
from PyTimeVar import PowerLaw
PwrLaw = PowerLaw(vY, n_powers=1)
pwrTrend, pwrGamma = PwrLaw.fit()
PwrLaw.summary()
PwrLaw.plot()

# # auxiliary power-law model to illustrate options
# vgamma0 = np.arange(0, 0.1, 0.1)
# options = {'maxiter': 5E5, 'disp': False}
# auxPwr = PowerLaw(vY, n_powers=1, vgamma0=vgamma0, options=options)
# auxPwrTrend, auxPwrGamma = auxPwr.fit()
# auxPwr.summary()

# # illustrate Kalman smoother
# from PyTimeVar import Kalman
# kalmanmodel = Kalman(vY=vY)
# smooth_trend = kalmanmodel.fit('smoother')
# kalmanmodel.plot()

# # illustrate GAS model
# from PyTimeVar import GAS
# N_gasmodel = GAS(vY=vY, mX=X, method='gaussian')
# N_GAStrend, N_GASparams = N_gasmodel.fit()
# N_gasmodel.plot()
