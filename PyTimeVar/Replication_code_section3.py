'''
Section 3: illustration of code on Temperature dataset

'''
import os
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

# illustrate power-law trend
from PyTimeVar import PowerLaw
# PwrLaw = PowerLaw(vY=vY, n_powers=2)
# pwrTrend, pwrGamma = PwrLaw.fit()
# PwrLaw.summary()
# PwrLaw.plot()
# C_LB_coeff, C_UB_coeff, C_LB_gamma, C_UB_gamma, C_LB_trend, C_UB_trend = PwrLaw.confidence_intervals(bootstraptype='LBWB', B=1299, alpha=0.05, block_constant=2)


# # # auxiliary power-law model to illustrate options
# vgamma0 = np.arange(0, 0.1, 0.05)
# options = {'maxiter': 1E3, 'disp': False}
# bounds = ((0,0),(0.1, 5), )
# auxPwr = PowerLaw(vY, n_powers=2, vgamma0=vgamma0, bounds=bounds, options=options)
# auxPwrTrend, auxPwrGamma = auxPwr.fit()
# auxPwr.summary()

# illustrate Kalman smoother
# from PyTimeVar import Kalman
# kalmanmodel = Kalman(vY=vY)
# [kl_filter, kl_predictor, kl_smoother] = kalmanmodel.fit('all')
# kalmanmodel.plot(individual=False)
# print(kalmanmodel.H, kalmanmodel.Q)

# sigma_u = 1/(1+np.exp(np.random.normal(0,1,len(vY))))
# kalmanmodel = Kalman(vY=vY, sigma_u = sigma_u)
# [kl_filter, kl_predictor, kl_smoother] = kalmanmodel.fit('all')
# kalmanmodel.plot(individual=True, confidence_intervals=True)

# # illustrate GAS model
# from PyTimeVar import GAS
# N_gasmodel = GAS(vY=vY, mX=mX, method='gaussian', niter=10)
# N_GAStrend, N_GASparams = N_gasmodel.fit()
# print(N_GASparams)
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



# In[]
#### SIMULATION STUDIES ##################
# ------------------- Simulation study for Power Law confidence intervals------
def sim_PwrLaw(iT, vGamma, vCoeff):
    vE = np.random.normal(0,1,iT)
    trend = np.arange(1, iT+1, 1).reshape(iT, 1)
    mP = trend ** vGamma
    vPwrTrend = mP @ vCoeff
    vY = vPwrTrend + vE
    return vY, vPwrTrend

initer = 1000
vT = [100, 250]

n_powers = 1
n_params = 3
lGamma = [np.array([1.0]), np.array([1.5]), np.array([1.8])]
vCoeff = np.array([0.001])

gamma_results_to_save = {}
output_directory = r'C:\Users\bpvan\OneDrive - Erasmus University Rotterdam\Documents\PhD projects\JSS_filters'
os.makedirs(output_directory, exist_ok=True)
for gamma_idx, vGamma_val in enumerate(lGamma):
    print(f'Starting simulations for True Gamma = {vGamma_val.item()}')
    current_gamma_aggregated_data = [] 
    for t_idx in range(len(vT)):
        current_T_val = vT[t_idx]
        print(f'\n--- Processing T = {current_T_val} ---')
        mCov = np.zeros((initer, n_params))
        mLen = np.zeros((initer, n_params))
        for i in range(initer):
            if (i + 1) % (initer // 100) == 0 or i == 0 or i == initer - 1:
                print(f'  Gamma={vGamma_val.item()}, T={current_T_val}, Iteration {i+1}/{initer}')
            
            vY, vPwrTrend_true = sim_PwrLaw(current_T_val, vGamma_val, vCoeff)
            PwrLaw = PowerLaw(vY=vY, n_powers=1)
            pwrTrend, pwrGamma = PwrLaw.fit()

            C_LB_coeff, C_UB_coeff, C_LB_gamma, C_UB_gamma, C_LB_trend, C_UB_trend = \
                PwrLaw.confidence_intervals(
                    bootstraptype='WB', B=1299, alpha=0.05,
                    block_constant=2, verbose=False
                )

            mCov[i, 0] = np.mean((vPwrTrend_true <= C_UB_trend) & (vPwrTrend_true >= C_LB_trend))
            mLen[i, 0] = np.mean(C_UB_trend - C_LB_trend)
            mCov[i, 1] = (C_LB_coeff <= vCoeff) & (C_UB_coeff >= vCoeff)
            mLen[i, 1] = C_UB_coeff - C_LB_coeff
            mCov[i, 2] = (C_LB_gamma <= vGamma_val) & (C_UB_gamma >= vGamma_val)
            mLen[i, 2] = C_UB_gamma - C_LB_gamma
        
        # Calculate mean coverage and mean length for each parameter for this T and Gamma
        mean_covs = np.mean(mCov, axis=0)
        mean_lens = np.mean(mLen, axis=0)

        # Append the results for Trend, Coeff, and Gamma for current T
        current_gamma_aggregated_data.append([mean_covs[0], mean_lens[0]]) # Trend
        current_gamma_aggregated_data.append([mean_covs[1], mean_lens[1]]) # Coefficient
        current_gamma_aggregated_data.append([mean_covs[2], mean_lens[2]]) # Gamma
    
    # Store the aggregated data for the current gamma
    gamma_results_to_save[vGamma_val.item()] = current_gamma_aggregated_data

# --- Writing Results to Files ---

print("\n--- Writing results to text files ---")
parameter_labels = ["Trend", "Coefficient", "Gamma"] # For internal reference/comments

for gamma_val, data_to_write in gamma_results_to_save.items():
    filename = f'results_npowers_{n_powers}_gamma_{gamma_val}.txt'
    output_filepath = os.path.join(output_directory, filename)
    with open(output_filepath, 'w') as f:
        f.write(f"# Simulation Results for True Gamma = {gamma_val:.3f}\n")
        f.write(f"# Columns: Coverage, Length\n")
        f.write(f"# Rows (T=100): Trend, Coeff, Gamma\n")
        f.write(f"# Rows (T=250): Trend, Coeff, Gamma\n")
        f.write(f"# Rows (T=500): Trend, Coeff, Gamma\n") # Assuming vT has 3 values
        f.write("#" * 30 + "\n")
        
        # Write the data in the specified format
        row_counter = 0
        for T_idx, T_val in enumerate(vT):
            f.write(f"# --- T = {T_val} ---\n")
            for param_idx in range(n_params):
                cov_val = data_to_write[row_counter][0]
                len_val = data_to_write[row_counter][1]
                f.write(f"{cov_val:.6f}\t{len_val:.6f}\n")
                row_counter += 1
                
    print(f"Results for Gamma = {gamma_val:.3f} saved to '{output_filepath}'")

print("\nAll simulation results saved.")
        
n_powers = 2
n_params = 5
lGamma = [np.array([2, 1]), np.array([2, 2]), np.array([2, 3])]
vCoeff = np.array([0.001, -0.003])

gamma_results_to_save = {}
output_directory = r'C:\Users\bpvan\OneDrive - Erasmus University Rotterdam\Documents\PhD projects\JSS_filters'
os.makedirs(output_directory, exist_ok=True)

for gamma_idx, vGamma_val in enumerate(lGamma):
    # Joining array elements for print statement
    print(f'Starting simulations for True Gamma = {", ".join(map(str, vGamma_val))}')
    current_gamma_aggregated_data = [] 
    for t_idx in range(len(vT)):
        current_T_val = vT[t_idx]
        print(f'\n--- Processing T = {current_T_val} ---')
        mCov = np.zeros((initer, n_params))
        mLen = np.zeros((initer, n_params))
        for i in range(initer):
            if (i + 1) % (initer // 100 if initer >= 10 else 1) == 0 or i == 0 or i == initer - 1: # Corrected print frequency
                print(f'  Gamma={", ".join(map(str, vGamma_val))}, T={current_T_val}, Iteration {i+1}/{initer}')
            
            vY, vPwrTrend_true = sim_PwrLaw(current_T_val, vGamma_val, vCoeff)
            PwrLaw = PowerLaw(vY=vY, n_powers=n_powers)
            pwrTrend, pwrGamma = PwrLaw.fit()

            C_LB_coeff, C_UB_coeff, C_LB_gamma, C_UB_gamma, C_LB_trend, C_UB_trend = \
                PwrLaw.confidence_intervals(
                    bootstraptype='WB', B=1299, alpha=0.05,
                    block_constant=2, verbose=False
                )

            mCov[i, 0] = np.mean((vPwrTrend_true <= C_UB_trend) & (vPwrTrend_true >= C_LB_trend))
            mLen[i, 0] = np.mean(C_UB_trend - C_LB_trend)
            
            # Loop for multiple coefficients and gammas
            for p in range(n_powers):
                mCov[i, 1 + p] = (C_LB_coeff[p] <= vCoeff[p]) & (C_UB_coeff[p] >= vCoeff[p])
                mLen[i, 1 + p] = C_UB_coeff[p] - C_LB_coeff[p]
                
                mCov[i, 1 + n_powers + p] = (C_LB_gamma[p] <= vGamma_val[p]) & (C_UB_gamma[p] >= vGamma_val[p])
                mLen[i, 1 + n_powers + p] = C_UB_gamma[p] - C_LB_gamma[p]
            
        mean_covs = np.mean(mCov, axis=0)
        mean_lens = np.mean(mLen, axis=0)

        # Append results for Trend
        current_gamma_aggregated_data.append([mean_covs[0], mean_lens[0]]) 
        
        # Append results for multiple Coefficients
        for p in range(n_powers):
            current_gamma_aggregated_data.append([mean_covs[1 + p], mean_lens[1 + p]]) 
        
        # Append results for multiple Gammas
        for p in range(n_powers):
            current_gamma_aggregated_data.append([mean_covs[1 + n_powers + p], mean_lens[1 + n_powers + p]]) 
    
    # Use a string representation of the array as the key for dictionary
    gamma_results_to_save[str(vGamma_val)] = current_gamma_aggregated_data

print("\n--- Writing results to text files ---")

# Adjusted parameter labels for multiple powers
parameter_labels_summary = ["Trend"]
for p in range(n_powers):
    parameter_labels_summary.append(f"Coeff_{p+1}")
for p in range(n_powers):
    parameter_labels_summary.append(f"Gamma_{p+1}")


for gamma_array_str, data_to_write in gamma_results_to_save.items():
    # Parse the string key back to an array for filename formatting if needed, or keep as string
    true_gamma_vals = np.array(list(map(float, gamma_array_str.strip('[]').split()))) # Converts "[[1. 1.5]]" to array
    
    filename = f'results_npowers_{n_powers}_gamma_{"_".join(map(lambda x: f"{x:.3f}", true_gamma_vals))}.txt'
    output_filepath = os.path.join(output_directory, filename)
    
    with open(output_filepath, 'w') as f:
        f.write(f"# Simulation Results for True Gamma = {gamma_array_str}\n")
        f.write(f"# Columns: Coverage, Length\n")
        f.write(f"# Order for each T: {', '.join(parameter_labels_summary)}\n")
        f.write(f"# T values: {vT}\n")
        f.write("#" * 30 + "\n")
        
        row_counter = 0
        for T_idx, T_val in enumerate(vT):
            f.write(f"# --- T = {T_val} ---\n")
            for param_idx in range(n_params): # Loop through all n_params rows for each T
                cov_val = data_to_write[row_counter][0]
                len_val = data_to_write[row_counter][1]
                f.write(f"{cov_val:.6f}\t{len_val:.6f}\n")
                row_counter += 1
                
    print(f"Results for Gamma = {gamma_array_str} saved to '{output_filepath}'")

print("\nAll simulation results saved.")





# import matplotlib.pyplot as plt
# # ------ Simulation study for N-GAS model ---
# initer = 10
# iM = 10
# iT = 500
# vN_cov, vN_length = np.zeros(initer), np.zeros(initer)
# dbeta0 = 0
# vparams = [0.03, 0.9, 0.2]
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

# for i in range(initer):
#     vY, vBeta_true_N = sim_N_GAS(iT, dbeta0, vparams)
#     mX = np.ones((iT, 1))
#     N_gasmodel = GAS(vY=vY, mX=mX, method='gaussian', niter=20)
#     N_GAStrend, N_GASparams = N_gasmodel.fit()
#     print('Parameters', N_GASparams)
#     mCI_l, mCI_u = N_gasmodel._confidence_bands(alpha=0.05, iM=iM)
#     vN_cov[i] = np.mean((N_GAStrend <= mCI_u[:,0]) & (N_GAStrend >= mCI_l[:,0]))
#     vN_length[i] = np.mean(mCI_u[:,0] - mCI_l[:,0])
#     plt.figure(figsize=(12,6))
#     plt.scatter(np.arange(0,iT,1),vBeta_true_N, c='b')
#     plt.scatter(np.arange(0,iT,1),N_GAStrend, c='r')
#     plt.scatter(np.arange(0,iT,1),mCI_l[:,0], c='g', alpha=0.2)
#     plt.scatter(np.arange(0,iT,1),mCI_u[:,0], c='g', alpha=0.2)
#     plt.show()
#     print(vN_cov[i], vN_length[i])
# print(np.mean(vN_cov), np.mean(vN_length))


# ------ Simulation study for t-GAS model ---
# initer = 10
# iM = 10
# iT = 100
# vt_cov, vt_length = np.zeros(initer), np.zeros(initer)
# dbeta0 = 0
# vparams = [3, 1, 0.2, 0.9, 0.2]
# def sim_t_GAS(iT, dbeta0, vparams):
#     vBeta_true_t = np.zeros(iT)
#     vBeta_true_t[0] = dbeta0
#     vY = np.zeros(iT)
#     vEps = np.random.standard_t(vparams[0], iT)
#     for t in range(1, iT):
#         yt = vY[t-1]
#         epst = yt - vBeta_true_t[t-1]
#         vxt = np.ones((1, 1))
#         temp1 = (1 + vparams[0]**(-1)) * (1 + vparams[0]**(-1)
#                                            * (epst / vparams[1])**2)**(-1)
#         mNablat = (1 + vparams[0])**(-1) * (3 + vparams[0]) * \
#             temp1 * vxt * epst
#         vbetaNow = vparams[2] + vparams[3]*vBeta_true_t[t-1] + vparams[4] * mNablat.squeeze()
#         vBeta_true_t[t] = vbetaNow
#         vY[t] = vBeta_true_t[t] + vEps[t]
#     return vY, vBeta_true_t

# for i in range(initer):
#     vY, vBeta_true_t = sim_t_GAS(iT, dbeta0, vparams)
#     mX = np.ones((iT, 1))
#     t_gasmodel = GAS(vY=vY, mX=mX, method='student', niter=10)
#     t_GAStrend, t_GASparams = t_gasmodel.fit()
#     np.set_printoptions(suppress=True, formatter={'float_kind': '{:.10f}'.format})
#     print('Parameters',t_GASparams)
#     mCI_l, mCI_u = t_gasmodel._confidence_bands(alpha=0.05, iM=iM)
#     vt_cov[i] = np.mean((t_GAStrend <= mCI_u[:,0]) & (t_GAStrend >= mCI_l[:,0]))
#     vt_length[i] = np.mean(mCI_u[:,0] - mCI_l[:,0])
#     plt.figure(figsize=(12,6))
#     plt.scatter(np.arange(0,iT,1),vBeta_true_t, c='b')
#     plt.scatter(np.arange(0,iT,1),t_GAStrend, c='r')
#     plt.scatter(np.arange(0,iT,1),mCI_l[:,0], c='g', alpha=0.2)
#     plt.scatter(np.arange(0,iT,1),mCI_u[:,0], c='g', alpha=0.2)
#     plt.show()
#     print(vt_cov[i], vt_length[i])
# print(np.mean(vt_cov), np.mean(vt_length))
