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

# set seed
np.random.seed(123)

# # illustrate LLR
# from PyTimeVar import LocalLinear
# model = LocalLinear(vY=vY, mX=mX)
# betaHatLLR = model.fit()

# # print summary
# model.summary()

# # plot trend and data
# model.plot_predicted(tau=[0.4,0.8])

# # plot confidence bands using LBWB
# S_LB, S_UB, P_LB, P_UB = model.confidence_bands(bootstrap_type='LBWB', Gsubs=None, plots=True)


# auxiliary LLR model to illustrate kernel, bandwidth selection, and tau
# model2LLR = LocalLinear(vY=vY, mX=mX, kernel='Gaussian', bw_selection='lmcv_8')
# beta_hat_model2 = model2LLR.fit()

# # illustrate boosted HP filter
# from PyTimeVar import BoostedHP
# bHPmodel = BoostedHP(vY=vY, dLambda=1600, iMaxIter=100)
# bHPtrend, bHPresiduals = bHPmodel.fit(
#     boost=True, stop="adf", dAlpha=0.05, verbose=False)
# bHPmodel.summary()
# bHPmodel.plot()

# # illustrate power-law trend
# from PyTimeVar import PowerLaw
# PwrLaw = PowerLaw(vY=vY, n_powers=1)
# pwrTrend, pwrGamma = PwrLaw.fit()
# PwrLaw.summary()
# PwrLaw.plot()
# C_LB_coeff, C_UB_coeff, C_LB_gamma, C_UB_gamma = PwrLaw.confidence_intervals(bootstraptype='SWB', B=1299, C=2)


# # # auxiliary power-law model to illustrate options
# vgamma0 = np.arange(0, 0.1, 0.05)
# options = {'maxiter': 1E3, 'disp': False}
# bounds = ((0,0),(0.1, 5), )
# auxPwr = PowerLaw(vY, n_powers=2, vgamma0=vgamma0, bounds=bounds, options=options)
# auxPwrTrend, auxPwrGamma = auxPwr.fit()
# auxPwr.summary()

# illustrate Kalman smoother
from PyTimeVar import Kalman
# kalmanmodel = Kalman(vY=vY)
# [kl_filter, kl_predictor, kl_smoother] = kalmanmodel.fit('all')
# kalmanmodel.plot(individual=False)

sigma_u = 1/(1+np.exp(np.random.normal(0,1,len(vY))))
kalmanmodel = Kalman(vY=vY, sigma_u = sigma_u)
[kl_filter, kl_predictor, kl_smoother] = kalmanmodel.fit('all')
kalmanmodel.plot(individual=True, confidence_intervals=True)

# # illustrate GAS model
# from PyTimeVar import GAS
# N_gasmodel = GAS(vY=vY, mX=mX, method='student', niter=10)
# N_GAStrend, N_GASparams = N_gasmodel.fit()
# N_gasmodel.plot(confidence_intervals=True)

# # illustrate srtuctural breaks class
# from PyTimeVar import Breaks
# breaksmodel = Breaks(vY, mX = np.ones((len(vY), 1)), iM=4)
# mBetaHat, glb, datevec = breaksmodel.fit()
# breaksmodel.plot()

# # illustrate Markov switching class
# from PyTimeVar import MarkovSwitching
# msmodel = MarkovSwitching(vY[1:] - vY[:-1], mX = np.ones((len(vY)-1, 1)), iS=2)
# best_beta, best_sigma2, best_P, best_smoothed_probs = msmodel.fit()
# msmodel.plot_coefficients()

# ------ Simulation study for N-GAS model ---
# initer = 10
# iM = 1000
# iT = 200
# vN_cov, vN_length = np.zeros(initer), np.zeros(initer)
# dbeta0 = 0
# vparams = [0.03, 1, 0.2]
# def sim_N_GAS(iT, dbeta0, vparams):
#     vBeta_true_N = np.zeros(iT)
#     vBeta_true_N[0] = dbeta0
#     vY = np.zeros(iT)
#     vEps = np.random.normal(0, 2, iT)
#     for t in range(1, iT):
#         yt = vY[t-1]
#         epst = yt - vBeta_true_N[t-1]
#         vxt = np.ones((1, 1))
#         mNablat = vxt * epst
#         vbetaNow = vparams[0] + vparams[1] * vBeta_true_N[t-1] + vparams[2] * mNablat.squeeze()
#         vBeta_true_N[t] = vbetaNow
#         vY[t] = vBeta_true_N[t] + vEps[t]
#     return vY, vBeta_true_N

# import matplotlib.pyplot as plt
# for i in range(initer):
#     vY, vBeta_true_N = sim_N_GAS(iT, dbeta0, vparams)
#     mX = np.ones((iT, 1))
#     N_gasmodel = GAS(vY=vY, mX=mX, method='gaussian', niter=10)
#     N_GAStrend, N_GASparams = N_gasmodel.fit()
#     mCI_l, mCI_u = N_gasmodel._confidence_bands(alpha=0.05, iM=iM)
#     vN_cov[i] = np.mean((N_GAStrend <= mCI_u[:,0]) & (N_GAStrend >= mCI_l[:,0]))
#     vN_length[i] = np.mean(mCI_u[:,0] - mCI_l[:,0])
#     plt.plot(vBeta_true_N, c='b')
#     plt.plot(N_GAStrend, c='r')
#     plt.plot(mCI_l[:,0], c='g')
#     plt.plot(mCI_u[:,0], c='g')
#     plt.show()
#     print(vN_cov[i], vN_length[i])
# print(np.mean(vN_cov), np.mean(vN_length))


# # ------ Simulation study for t-GAS model ---
# initer = 10
# iM = 1000
# iT = 200
# vt_cov, vt_length = np.zeros(initer), np.zeros(initer)
# dbeta0 = 0
# vparams = [0.03, 1, 0.2, 4, 0.2]
# def sim_t_GAS(iT, dbeta0, vparams):
#     vBeta_true_N = np.zeros(iT)
#     vBeta_true_N[0] = dbeta0
#     vY = np.zeros(iT)
#     vEps = np.random.normal(0, 2, iT)
#     for t in range(1, iT):
#         yt = vY[t-1]
#         epst = yt - vBeta_true_N[t-1]
#         vxt = np.ones((1, 1))
#         temp1 = (1 + vparams[3]**(-1)) * (1 + vparams[3]**(-1)
#                                                 * (epst / vparams[4])**2)**(-1)
#         mNablat = (1 + vparams[3])**(-1) * (3 + vparams[3]) * \
#                 temp1 * vxt * epst
#         vbetaNow = vparams[0] + vparams[1] * vBeta_true_N[t-1] + vparams[2] * mNablat.squeeze()
#         vBeta_true_N[t] = vbetaNow
#         vY[t] = vBeta_true_N[t] + vEps[t]
#     return vY, vBeta_true_N

# for i in range(initer):
#     vY, vBeta_true_t = sim_t_GAS(iT, dbeta0, vparams)
#     mX = np.ones((iT, 1))
#     t_gasmodel = GAS(vY=vY, mX=mX, method='student', niter=10)
#     t_GAStrend, t_GASparams = t_gasmodel.fit()
#     mCI_l, mCI_u = t_gasmodel._confidence_bands(alpha=0.05, iM=iM)
#     vt_cov[i] = np.mean((t_GAStrend <= mCI_u[:,0]) & (t_GAStrend >= mCI_l[:,0]))
#     vt_length[i] = np.mean(mCI_u[:,0] - mCI_l[:,0])
#     plt.plot(vBeta_true_t, c='b')
#     plt.plot(mCI_l[:,0])
#     plt.plot(mCI_u[:,0])
#     plt.show()
#     print(vt_cov[i], vt_length[i])
# print(np.mean(vt_cov), np.mean(vt_length))
