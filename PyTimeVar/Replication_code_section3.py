'''
Section 3: illustration of code on Temperature dataset

'''
# Load data
from PyTimeVar.datasets import temperature
import numpy as np
data = temperature.load(
    regions=['World'], start_date='1961', end_date='2023')
vY = data.values
mX = np.ones_like(vY)
# mX=np.ones(shape=(63,3))
# mX[:,1] = temperature.load(
#     regions=['Africa'], start_date='1961', end_date='2023').values.flatten()
# mX[:,2] = temperature.load(
#     regions=['Europe'], start_date='1961', end_date='2023').values.flatten()

# set seed
np.random.seed(123)

# illustrate LLR
from PyTimeVar import LocalLinear
model = LocalLinear(vY=vY, mX=mX)
betaHatLLR = model.fit()

# auxiliary LLR model to illustrate kernel, bandwidth selection, and tau
model2LLR = LocalLinear(vY=vY, mX=mX, kernel='Gaussian', bw_selection='lmcv_8')
beta_hat_model2 = model2LLR.fit()

# print summary
model.summary()

# plot trend and data
model.plot_predicted(tau=[0.4,0.8])

# Examples of invalid tau values
model.plot_predicted(tau=[-0.2, 1.5])
model.plot_predicted(tau=[0.6, 0.4])
# model.plot_predicted(tau=[2, 3])

# plot confidence bands using LBWB
S_LB, S_UB, P_LB, P_UB = model.confidence_bands(bootstrap_type='LBWB', Gsubs=None, plots=True)

# illustrate boosted HP filter
from PyTimeVar import BoostedHP
bHPmodel = BoostedHP(vY=vY, dLambda=1600, iMaxIter=100)
bHPtrend, bHPresiduals = bHPmodel.fit(
    boost=True, stop="adf", dAlpha=0.05, verbose=False)
bHPmodel.summary()
bHPmodel.plot()

# illustrate power-law trend
from PyTimeVar import PowerLaw
PwrLaw = PowerLaw(vY=vY, n_powers=1)
pwrTrend, pwrGamma = PwrLaw.fit()
PwrLaw.summary()
PwrLaw.plot()
C_LB_coeff, C_UB_coeff, C_LB_gamma, C_UB_gamma, C_LB_trend, C_UB_trend = PwrLaw.confidence_intervals(bootstraptype='SWB', B=1299, alpha=0.05,
        block_constant=2, verbose=True)

# # auxiliary power-law model to illustrate options
vgamma0 = np.arange(0, 0.1, 0.05)
options = {'maxiter': 1E3, 'disp': False}
bounds = ((0,0),(0.1, 5), )
auxPwr = PowerLaw(vY, n_powers=2, vgamma0=vgamma0, bounds=bounds, options=options)
auxPwrTrend, auxPwrGamma = auxPwr.fit()
# auxPwr.summary()
C_LB_coeff, C_UB_coeff, C_LB_gamma, C_UB_gamma, C_LB_trend, C_UB_trend = auxPwr.confidence_intervals(bootstraptype='SWB',
                                                                                                     B=1299, alpha=0.05, block_constant=2,verbose=True)
auxPwr.plot(confidence_intervals=True)

# illustrate Kalman smoother
from PyTimeVar import Kalman
kalmanmodel = Kalman(vY=vY,mX=mX)
[kl_filter, kl_predictor, kl_smoother] = kalmanmodel.fit('all')
kalmanmodel.plot(individual=False)

sigma_u = 1/(1+np.exp(np.random.normal(0,1,len(vY))))
kalmanmodel = Kalman(vY=vY,mX=mX, sigma_u = sigma_u)
[kl_filter, kl_predictor, kl_smoother] = kalmanmodel.fit('all')
kalmanmodel.plot(individual=True, confidence_intervals=True)

# illustrate GAS model
from PyTimeVar import GAS
N_gasmodel = GAS(vY=vY, mX=mX, method='gaussian', niter=10)
N_GAStrend, N_GASparams= N_gasmodel.fit()

N_gasmodel.plot(confidence_intervals=True, alpha=0.05)

