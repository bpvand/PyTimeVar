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
PwrLaw = PowerLaw(vY=vY, n_powers=2)
pwrTrend, pwrGamma = PwrLaw.fit()
PwrLaw.summary()
PwrLaw.plot()
C_LB_coeff, C_UB_coeff, C_LB_gamma, C_UB_gamma, C_LB_trend, C_UB_trend = PwrLaw.confidence_intervals(bootstraptype='LBWB', B=1299, alpha=0.05, block_constant=2)
PwrLaw.plot(confidence_intervals=True)

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

# sigma_u = 1/(1+np.exp(np.random.normal(0,1,len(vY))))
# kalmanmodel = Kalman(vY=vY, sigma_u = sigma_u)
# [kl_filter, kl_predictor, kl_smoother] = kalmanmodel.fit('all')
# kalmanmodel.plot(individual=True, confidence_intervals=True)

# # illustrate GAS model
from PyTimeVar import GAS
# N_gasmodel = GAS(vY=vY, mX=mX, method='gaussian', niter=10)
# N_GAStrend, N_GASparams = N_gasmodel.fit()
# print(N_GASparams)
# N_gasmodel.plot(confidence_intervals=True, alpha=0.05)


# In[]
#### SIMULATION STUDIES ##################
# ------------------- Simulation study for Power Law confidence intervals------
# def sim_PwrLaw(iT, vGamma, vCoeff):
#     vE = np.random.normal(0,0.25,iT)
#     trend = np.arange(1, iT+1, 1).reshape(iT, 1)
#     mP = trend ** vGamma
#     vPwrTrend = mP @ vCoeff
#     vY = vPwrTrend + vE
#     return vY, vPwrTrend

# initer = 1000
# vT = [100, 250]

# n_powers = 1
# n_params = 3
# lGamma = [np.array([1.0]), np.array([1.5]), np.array([1.8])]
# vCoeff = np.array([0.001])

# gamma_results_to_save = {}
# output_directory = r'C:\Users\bpvan\OneDrive - Erasmus University Rotterdam\Documents\PhD projects\JSS_filters'
# os.makedirs(output_directory, exist_ok=True)
# for gamma_idx, vGamma_val in enumerate(lGamma):
#     print(f'Starting simulations for True Gamma = {vGamma_val.item()}')
#     current_gamma_aggregated_data = [] 
#     for t_idx in range(len(vT)):
#         current_T_val = vT[t_idx]
#         print(f'\n--- Processing T = {current_T_val} ---')
#         mCov = np.zeros((initer, n_params))
#         mLen = np.zeros((initer, n_params))
#         for i in range(initer):
#             if (i + 1) % (initer // 100) == 0 or i == 0 or i == initer - 1:
#                 print(f'  Gamma={vGamma_val.item()}, T={current_T_val}, Iteration {i+1}/{initer}')
            
#             vY, vPwrTrend_true = sim_PwrLaw(current_T_val, vGamma_val, vCoeff)
#             PwrLaw = PowerLaw(vY=vY, n_powers=1)
#             pwrTrend, pwrGamma = PwrLaw.fit()

#             C_LB_coeff, C_UB_coeff, C_LB_gamma, C_UB_gamma, C_LB_trend, C_UB_trend = \
#                 PwrLaw.confidence_intervals(
#                     bootstraptype='WB', B=1299, alpha=0.05,
#                     block_constant=2, verbose=False
#                 )

#             mCov[i, 0] = np.mean((vPwrTrend_true <= C_UB_trend) & (vPwrTrend_true >= C_LB_trend))
#             mLen[i, 0] = np.mean(C_UB_trend - C_LB_trend)
#             mCov[i, 1] = (C_LB_coeff <= vCoeff) & (C_UB_coeff >= vCoeff)
#             mLen[i, 1] = C_UB_coeff - C_LB_coeff
#             mCov[i, 2] = (C_LB_gamma <= vGamma_val) & (C_UB_gamma >= vGamma_val)
#             mLen[i, 2] = C_UB_gamma - C_LB_gamma
        
#         # Calculate mean coverage and mean length for each parameter for this T and Gamma
#         mean_covs = np.mean(mCov, axis=0)
#         mean_lens = np.mean(mLen, axis=0)

#         # Append the results for Trend, Coeff, and Gamma for current T
#         current_gamma_aggregated_data.append([mean_covs[0], mean_lens[0]]) # Trend
#         current_gamma_aggregated_data.append([mean_covs[1], mean_lens[1]]) # Coefficient
#         current_gamma_aggregated_data.append([mean_covs[2], mean_lens[2]]) # Gamma
    
#     # Store the aggregated data for the current gamma
#     gamma_results_to_save[vGamma_val.item()] = current_gamma_aggregated_data

# # --- Writing Results to Files ---

# print("\n--- Writing results to text files ---")
# parameter_labels = ["Trend", "Coefficient", "Gamma"] # For internal reference/comments

# for gamma_val, data_to_write in gamma_results_to_save.items():
#     filename = f'results_npowers_{n_powers}_gamma_{gamma_val}.txt'
#     output_filepath = os.path.join(output_directory, filename)
#     with open(output_filepath, 'w') as f:
#         f.write(f"# Simulation Results for True Gamma = {gamma_val:.3f}\n")
#         f.write(f"# Columns: Coverage, Length\n")
#         f.write(f"# Rows (T=100): Trend, Coeff, Gamma\n")
#         f.write(f"# Rows (T=250): Trend, Coeff, Gamma\n")
#         f.write(f"# Rows (T=500): Trend, Coeff, Gamma\n") # Assuming vT has 3 values
#         f.write("#" * 30 + "\n")
        
#         # Write the data in the specified format
#         row_counter = 0
#         for T_idx, T_val in enumerate(vT):
#             f.write(f"# --- T = {T_val} ---\n")
#             for param_idx in range(n_params):
#                 cov_val = data_to_write[row_counter][0]
#                 len_val = data_to_write[row_counter][1]
#                 f.write(f"{cov_val:.6f}\t{len_val:.6f}\n")
#                 row_counter += 1
                
#     print(f"Results for Gamma = {gamma_val:.3f} saved to '{output_filepath}'")

# print("\nAll simulation results saved.")
       
# import pandas as pd
# n_powers = 2
# n_params = 5
# lGamma = [np.array([0.7, 2])]#[np.array([0.7, 1.5]), np.array([0.7, 1.8]), np.array([0.7, 2])]
# vCoeff = np.array([0.15, 0.005])

# import matplotlib.pyplot as plt
# gamma_results_to_save = {}
# output_directory = r'C:\Users\bpvan\OneDrive - Erasmus University Rotterdam\Documents\PhD projects\JSS_filters'
# os.makedirs(output_directory, exist_ok=True)

# for gamma_idx, vGamma_val in enumerate(lGamma):
#     # Joining array elements for print statement
#     print(f'Starting simulations for True Gamma = {", ".join(map(str, vGamma_val))}')
#     current_gamma_aggregated_data = [] 
#     for t_idx in range(len(vT)):
#         current_T_val = vT[t_idx]
#         print(f'\n--- Processing T = {current_T_val} ---')
#         mCov = np.zeros((initer, n_params))
#         mLen = np.zeros((initer, n_params))
#         for i in range(initer):
#             if (i + 1) % (initer // 100 if initer >= 10 else 1) == 0 or i == 0 or i == initer - 1: # Corrected print frequency
#                 print(f'  Gamma={", ".join(map(str, vGamma_val))}, T={current_T_val}, Iteration {i+1}/{initer}')
            
#             vY, vPwrTrend_true = sim_PwrLaw(current_T_val, vGamma_val, vCoeff)
            
#             # plt.plot(vY)
#             # plt.plot(vPwrTrend_true)
#             # plt.show()
            
#             PwrLaw = PowerLaw(vY=vY, n_powers=n_powers)
#             pwrTrend, pwrGamma = PwrLaw.fit()
            
#             # print('Gamma',pwrGamma)
#             # print('Coeff',PwrLaw.coeffHat)

#             C_LB_coeff, C_UB_coeff, C_LB_gamma, C_UB_gamma, C_LB_trend, C_UB_trend = \
#                 PwrLaw.confidence_intervals(
#                     bootstraptype='WB', B=1299, alpha=0.05,
#                     block_constant=2, verbose=False
#                 )
            
#             # print('Coeff:', C_LB_coeff, C_UB_coeff)
#             # print('Gamma:', C_LB_gamma, C_UB_gamma)
            
#             plt.plot(vY, label='data', c='k')
#             plt.plot(vPwrTrend_true, label='True', c='r')
#             plt.fill_between(np.arange(0,len(vY),1), C_LB_trend, C_UB_trend, label='CI', color='grey', alpha=0.2)
#             plt.plot(pwrTrend, label='est')
#             plt.legend()
#             plt.show()
            
#             mCov[i, 0] = np.mean((vPwrTrend_true <= C_UB_trend) & (vPwrTrend_true >= C_LB_trend))
#             mLen[i, 0] = np.mean(C_UB_trend - C_LB_trend)
            
#             # Loop for multiple coefficients and gammas
#             for p in range(n_powers):
#                 mCov[i, 1 + p] = (C_LB_coeff[p] <= vCoeff[p]) & (C_UB_coeff[p] >= vCoeff[p])
#                 mLen[i, 1 + p] = C_UB_coeff[p] - C_LB_coeff[p]
                
#                 mCov[i, 1 + n_powers + p] = (C_LB_gamma[p] <= vGamma_val[p]) & (C_UB_gamma[p] >= vGamma_val[p])
#                 mLen[i, 1 + n_powers + p] = C_UB_gamma[p] - C_LB_gamma[p]
            
#         pd.DataFrame(mCov).to_csv(r'C:\Users\bpvan\OneDrive - Erasmus University Rotterdam\Documents\PhD projects\JSS_filters\Cov_Pwrlaw_sims_T_{t_idx}_gamma_{vGamma_val}.csv')
#         pd.DataFrame(mLen).to_csv(r'C:\Users\bpvan\OneDrive - Erasmus University Rotterdam\Documents\PhD projects\JSS_filters\Len_Pwrlaw_sims_T_{t_idx}_gamma_{vGamma_val}.csv')
        
#         mean_covs = np.mean(mCov, axis=0)
#         mean_lens = np.mean(mLen, axis=0)

#         # Append results for Trend
#         current_gamma_aggregated_data.append([mean_covs[0], mean_lens[0]]) 
        
#         # Append results for multiple Coefficients
#         for p in range(n_powers):
#             current_gamma_aggregated_data.append([mean_covs[1 + p], mean_lens[1 + p]]) 
        
#         # Append results for multiple Gammas
#         for p in range(n_powers):
#             current_gamma_aggregated_data.append([mean_covs[1 + n_powers + p], mean_lens[1 + n_powers + p]]) 
    
#     # Use a string representation of the array as the key for dictionary
#     gamma_results_to_save[str(vGamma_val)] = current_gamma_aggregated_data

# print("\n--- Writing results to text files ---")

# # Adjusted parameter labels for multiple powers
# parameter_labels_summary = ["Trend"]
# for p in range(n_powers):
#     parameter_labels_summary.append(f"Coeff_{p+1}")
# for p in range(n_powers):
#     parameter_labels_summary.append(f"Gamma_{p+1}")


# for gamma_array_str, data_to_write in gamma_results_to_save.items():
#     # Parse the string key back to an array for filename formatting if needed, or keep as string
#     true_gamma_vals = np.array(list(map(float, gamma_array_str.strip('[]').split()))) # Converts "[[1. 1.5]]" to array
    
#     filename = f'results_npowers_{n_powers}_gamma_{"_".join(map(lambda x: f"{x:.3f}", true_gamma_vals))}.txt'
#     output_filepath = os.path.join(output_directory, filename)
    
#     with open(output_filepath, 'w') as f:
#         f.write(f"# Simulation Results for True Gamma = {gamma_array_str}\n")
#         f.write(f"# Columns: Coverage, Length\n")
#         f.write(f"# Order for each T: {', '.join(parameter_labels_summary)}\n")
#         f.write(f"# T values: {vT}\n")
#         f.write("#" * 30 + "\n")
        
#         row_counter = 0
#         for T_idx, T_val in enumerate(vT):
#             f.write(f"# --- T = {T_val} ---\n")
#             for param_idx in range(n_params): # Loop through all n_params rows for each T
#                 cov_val = data_to_write[row_counter][0]
#                 len_val = data_to_write[row_counter][1]
#                 f.write(f"{cov_val:.6f}\t{len_val:.6f}\n")
#                 row_counter += 1
                
#     print(f"Results for Gamma = {gamma_array_str} saved to '{output_filepath}'")

# print("\nAll simulation results saved.")


# # In[]
# import matplotlib.pyplot as plt
# #### SIMULATION STUDIES GAS ##################
# iM = 1000
# ### Simulation study for N-GAS model
# def sim_N_GAS(iT, dbeta0, vparams):
#     """
#     Simulates data from a Gaussian GAS (N-GAS) model.

#     Args:
#         iT (int): Number of time points.
#         dbeta0 (float): Initial value for beta.
#         vparams (list): List of true parameters [omega, A, B].

#     Returns:
#         tuple: (vY, vBeta_true_N)
#             vY (np.array): Simulated observations.
#             vBeta_true_N (np.array): True underlying beta trend.
#     """
#     vBeta_true_N = np.zeros(iT)
#     vBeta_true_N[0] = dbeta0
#     vY = np.zeros(iT)
#     vEps = np.random.normal(0, 1, iT) # Error term with std dev 2
    
#     omega, A, B = vparams[0], vparams[1], vparams[2]

#     for t in range(1, iT):
#         yt_minus_1 = vY[t-1]
#         beta_true_t_minus_1 = vBeta_true_N[t-1]
#         epst = yt_minus_1 - beta_true_t_minus_1 

#         vxt = np.ones((1, 1)) # Exogenous variable (constant 1 for a simple model)
        
#         # Score function (nabla_t) for Gaussian distribution is simply the error
#         mNablat = vxt * epst

#         # Update equation for beta
#         vbetaNow = omega + A * beta_true_t_minus_1 + B * mNablat.squeeze()
#         vBeta_true_N[t] = vbetaNow
#         vY[t] = vBeta_true_N[t] + vEps[t]
#     return vY, vBeta_true_N

# # --- N-GAS Simulation Parameters ---
# initer = 20 # Number of independent simulations
# vT = [100, 250] # Time series lengths to simulate
# dbeta0_N_GAS = 0
# vParams_N_GAS_true = np.array([0.03, 0.9, 0.2]) # True parameters [omega, A, B]

# # Only 1 parameter for tracking: the Trend (index 0)
# n_tracked_params_N_GAS = 1 
# output_directory = r'C:\Users\bpvan\OneDrive - Erasmus University Rotterdam\Documents\PhD projects\JSS_filters\GAS_results' # Dedicated folder for GAS
# os.makedirs(output_directory, exist_ok=True)

# # List to store aggregated results (each element will be [coverage, length] for the trend at a given T)
# gas_results_for_trend_only = []

# print(f'Starting N-GAS simulations for True Parameters = {", ".join(map(str, vParams_N_GAS_true))}')

# for t_idx in range(len(vT)):
#     current_T_val = vT[t_idx]
#     print(f'\n--- Processing T = {current_T_val} for N-GAS ---')
#     # mCov and mLen now only need to store results for the trend
#     mCov = np.zeros((initer, n_tracked_params_N_GAS))
#     mLen = np.zeros((initer, n_tracked_params_N_GAS))

#     for i in range(initer):
#         if (i + 1) % (initer // 10 if initer >= 10 else 1) == 0 or i == 0 or i == initer - 1:
#             print(f'  N-GAS Parameters={", ".join(map(str, vParams_N_GAS_true))}, T={current_T_val}, Iteration {i+1}/{initer}')

#         vY, vBeta_true = sim_N_GAS(current_T_val, dbeta0_N_GAS, vParams_N_GAS_true)
#         mX = np.ones((current_T_val, 1))

#         gas_model = GAS(vY=vY, mX=mX, method='gaussian', niter=50000)
#         gas_trend_est, _ = gas_model.fit() # We only care about the trend_est here
        
#         print(gas_model.params)

#         # Get confidence intervals for the trend only
#         C_LB_trend, C_UB_trend = gas_model._confidence_bands(alpha=0.05, iM=iM)
#         C_LB_trend = C_LB_trend[:,0]
#         C_UB_trend = C_UB_trend[:,0]
        
#         # plt.plot(vY, label='data', c='k')
#         plt.plot(vBeta_true, label='True', c='r')
#         plt.fill_between(np.arange(0,len(vY),1), C_LB_trend, C_UB_trend, label = 'CI', color='grey', alpha=0.2)
#         plt.plot(gas_trend_est, label='est')
#         plt.legend()
#         plt.show()
        
#         # Coverage and Length for the Trend (index 0)
#         mCov[i, 0] = np.mean((vBeta_true <= C_UB_trend) & (vBeta_true >= C_LB_trend))
#         mLen[i, 0] = np.mean(C_UB_trend - C_LB_trend)
        
#         # print('Cov:', mCov[i,0])
    
#     # Calculate mean coverage and mean length for the trend for this T
#     mean_cov_trend = np.mean(mCov[:, 0])
#     mean_len_trend = np.mean(mLen[:, 0])

#     # Append results for Trend
#     gas_results_for_trend_only.append([mean_cov_trend, mean_len_trend])
        
#     print(mean_cov_trend, mean_len_trend)
# # --- Writing N-GAS Results to Files ---
# print("\n--- Writing N-GAS results to text files ---")

# param_string_for_filename = "_".join(map(lambda x: f"{x:.3f}".replace('.', ''), vParams_N_GAS_true))
# filename = f'050725_results_NGAS_trend_only_params_{param_string_for_filename}.txt'
# output_filepath = os.path.join(output_directory, filename)

# with open(output_filepath, 'w') as f:
#     f.write(f"# Simulation Results for N-GAS Model (Trend Only)\n")
#     f.write(f"# True Parameters (Omega, A, B): {str(vParams_N_GAS_true)}\n")
#     f.write(f"# Columns: Coverage, Length\n")
#     f.write(f"# Measured for: Trend\n")
#     f.write(f"# T values: {vT}\n")
#     f.write("#" * 30 + "\n")
    
#     for T_idx, T_val in enumerate(vT):
#         f.write(f"# --- T = {T_val} ---\n")
#         cov_val = gas_results_for_trend_only[T_idx][0]
#         len_val = gas_results_for_trend_only[T_idx][1]
#         f.write(f"{cov_val:.6f}\t{len_val:.6f}\n")
            
# print(f"Results for N-GAS (trend only) saved to '{output_filepath}'")

# print("\nAll N-GAS simulation results saved.")

# ### Simulation study for t-GAS model
# def sim_t_GAS(iT, dbeta0, vparams):
#     """
#     Simulates data from a Student's t-GAS model.

#     Args:
#         iT (int): Number of time points.
#         dbeta0 (float): Initial value for beta.
#         vparams (list): List of true parameters [nu, omega, A, B, C].

#     Returns:
#         tuple: (vY, vBeta_true_t)
#             vY (np.array): Simulated observations.
#             vBeta_true_t (np.array): True underlying beta trend.
#     """
#     vBeta_true_t = np.zeros(iT)
#     vBeta_true_t[0] = dbeta0
#     vY = np.zeros(iT)
    
#     nu, omega, A, B, C = vparams[0], vparams[1], vparams[2], vparams[3], vparams[4]
#     vEps = np.random.standard_t(nu, iT) # Error term from Student's t-distribution

#     for t in range(1, iT):
#         yt_minus_1 = vY[t-1]
#         beta_true_t_minus_1 = vBeta_true_t[t-1]
#         epst = yt_minus_1 - beta_true_t_minus_1

#         vxt = np.ones((1, 1))

#         # Score function for Student's t-distribution (simplified to match your original)
#         # Note: In actual t-GAS, the scaling factor involves the conditional variance.
#         # This is based on your provided original sim_t_GAS.
#         temp1 = (1 + nu**(-1)) * (1 + nu**(-1) * (epst / omega)**2)**(-1)
#         mNablat = (1 + nu)**(-1) * (3 + nu) * temp1 * vxt * epst 

#         # Update equation for beta
#         vbetaNow = A + B * beta_true_t_minus_1 + C * mNablat.squeeze()
#         vBeta_true_t[t] = vbetaNow
#         vY[t] = vBeta_true_t[t] + vEps[t]
#     return vY, vBeta_true_t

# # --- t-GAS Simulation Parameters ---
# initer = 1000
# vT = [100, 250] # Consistent with N-GAS simulation
# dbeta0_t_GAS = 0
# vParams_t_GAS_true = np.array([3, 1, 0.03, 0.9, 0.2]) # True parameters [nu, omega, A, B, C]

# # Only 1 parameter for tracking: the Trend (index 0)
# n_tracked_params_t_GAS = 1 
# # Output directory is already defined and created above.

# # List to store aggregated results (each element will be [coverage, length] for the trend at a given T)
# t_gas_results_for_trend_only = []

# print(f'Starting t-GAS simulations for True Parameters = {", ".join(map(str, vParams_t_GAS_true))}')

# for t_idx in range(len(vT)):
#     current_T_val = vT[t_idx]
#     print(f'\n--- Processing T = {current_T_val} for t-GAS ---')
#     # mCov and mLen now only need to store results for the trend
#     mCov = np.zeros((initer, n_tracked_params_t_GAS))
#     mLen = np.zeros((initer, n_tracked_params_t_GAS))

#     for i in range(initer):
#         if (i + 1) % (initer // 10 if initer >= 10 else 1) == 0 or i == 0 or i == initer - 1:
#             print(f'  t-GAS Parameters={", ".join(map(str, vParams_t_GAS_true))}, T={current_T_val}, Iteration {i+1}/{initer}')

#         vY, vBeta_true = sim_t_GAS(current_T_val, dbeta0_t_GAS, vParams_t_GAS_true)
#         mX = np.ones((current_T_val, 1))

#         gas_model = GAS(vY=vY, mX=mX, method='student', niter=10) # niter from your original t-GAS setup
#         gas_trend_est, _ = gas_model.fit() # We only care about the trend_est here

#         # Get confidence intervals for the trend only
#         C_LB_trend, C_UB_trend = gas_model._confidence_bands(alpha=0.05, iM=iM)
#         C_LB_trend = C_LB_trend[:,0]
#         C_UB_trend = C_UB_trend[:,0]
        
#         # Coverage and Length for the Trend (index 0)
#         mCov[i, 0] = np.mean((vBeta_true <= C_UB_trend) & (vBeta_true >= C_LB_trend))
#         mLen[i, 0] = np.mean(C_UB_trend - C_LB_trend)
    
#     # Calculate mean coverage and mean length for the trend for this T
#     mean_cov_trend = np.mean(mCov[:, 0])
#     mean_len_trend = np.mean(mLen[:, 0])

#     # Append results for Trend
#     t_gas_results_for_trend_only.append([mean_cov_trend, mean_len_trend])
        
# # --- Writing t-GAS Results to Files ---
# print("\n--- Writing t-GAS results to text files ---")

# param_string_for_filename = "_".join(map(lambda x: f"{x:.3f}".replace('.', ''), vParams_t_GAS_true))
# filename = f'050725_ results_tGAS_trend_only_params_{param_string_for_filename}.txt'
# output_filepath = os.path.join(output_directory, filename)

# with open(output_filepath, 'w') as f:
#     f.write(f"# Simulation Results for t-GAS Model (Trend Only)\n")
#     f.write(f"# True Parameters (Nu, Omega, A, B, C): {str(vParams_t_GAS_true)}\n")
#     f.write(f"# Columns: Coverage, Length\n")
#     f.write(f"# Measured for: Trend\n")
#     f.write(f"# T values: {vT}\n")
#     f.write("#" * 30 + "\n")
    
#     for T_idx, T_val in enumerate(vT):
#         f.write(f"# --- T = {T_val} ---\n")
#         cov_val = t_gas_results_for_trend_only[T_idx][0]
#         len_val = t_gas_results_for_trend_only[T_idx][1]
#         f.write(f"{cov_val:.6f}\t{len_val:.6f}\n")
            
# print(f"Results for t-GAS (trend only) saved to '{output_filepath}'")

# print("\nAll t-GAS simulation results saved.")


# In[]
# import matplotlib.pyplot as plt
# import scipy.stats as st
# alpha= 0.05
# #### SIMULATION STUDIES KALMAN ##################
# def sim_Kalman(iT):
#     vTrend = np.sin(2*np.pi*np.linspace(0,1,iT))
#     vY = vTrend + np.random.normal(0,1,iT)
#     return vY, vTrend

# # --- N-GAS Simulation Parameters ---
# initer = 25 # Number of independent simulations
# vT = [250, 500, 1000] # Time series lengths to simulate

# # Only 1 parameter for tracking: the Trend (index 0)
# n_tracked_params = 1 
# output_directory = r'C:\Users\bpvan\OneDrive - Erasmus University Rotterdam\Documents\PhD projects\JSS_filters\Kalman_results' # Dedicated folder for Kalman
# os.makedirs(output_directory, exist_ok=True)

# # List to store aggregated results (each element will be [coverage, length] for the trend at a given T)
# kalman_results_for_trend_only = []

# print('Starting Kalman simulations')

# for t_idx in range(len(vT)):
#     current_T_val = vT[t_idx]
#     print(f'\n--- Processing T = {current_T_val} for Kalman ---')
#     # mCov and mLen now only need to store results for the trend
#     mCov = np.zeros((initer, n_tracked_params))
#     mLen = np.zeros((initer, n_tracked_params))

#     for i in range(initer):
#         if (i + 1) % (initer // 10 if initer >= 10 else 1) == 0 or i == 0 or i == initer - 1:
#             print(f'T={current_T_val}, Iteration {i+1}/{initer}')

#         vY, vBeta_true = sim_Kalman(current_T_val)

#         kl_model = Kalman(vY=vY)
#         kl_trend_est = kl_model.fit('smoother') # We only care about the trend_est here

#         # Get confidence intervals for the trend only
#         C_UB_trend = kl_model.smooth + st.norm.ppf(1-alpha)*np.sqrt(kl_model.V[:,0,0])
#         C_LB_trend = kl_model.smooth + st.norm.ppf(alpha)*np.sqrt(kl_model.V[:,0,0])
        
#         # plt.plot(vY, label='data', c='k')
#         plt.plot(vBeta_true, label='True', c='r')
#         plt.fill_between(np.arange(0,len(vY),1), C_LB_trend, C_UB_trend, color='grey', alpha=0.2)
#         plt.plot(kl_trend_est, label='est')
#         plt.legend()
#         plt.show()
        
#         # Coverage and Length for the Trend (index 0)
#         mCov[i, 0] = np.mean((vBeta_true <= C_UB_trend) & (vBeta_true >= C_LB_trend))
#         mLen[i, 0] = np.mean(C_UB_trend - C_LB_trend)
        
#         # print('Cov:', mCov[i,0])
    
#     # Calculate mean coverage and mean length for the trend for this T
#     mean_cov_trend = np.mean(mCov[:, 0])
#     mean_len_trend = np.mean(mLen[:, 0])

#     # Append results for Trend
#     kalman_results_for_trend_only.append([mean_cov_trend, mean_len_trend])
        
# # --- Writing N-GAS Results to Files ---
# print("\n--- Writing N-GAS results to text files ---")

# filename = 'results_Kalman_smoother.txt'
# output_filepath = os.path.join(output_directory, filename)

# with open(output_filepath, 'w') as f:
#     f.write("# Simulation Results for Kalman Model (Trend Only)\n")
#     f.write("# Columns: Coverage, Length\n")
#     f.write("# Measured for: Trend\n")
#     f.write(f"# T values: {vT}\n")
#     f.write("#" * 30 + "\n")
    
#     for T_idx, T_val in enumerate(vT):
#         f.write(f"# --- T = {T_val} ---\n")
#         cov_val = kalman_results_for_trend_only[T_idx][0]
#         len_val = kalman_results_for_trend_only[T_idx][1]
#         f.write(f"{cov_val:.6f}\t{len_val:.6f}\n")
            
# print(f"Results for Kalman smoother (trend only) saved to '{output_filepath}'")

# print("\nAll Kalman simulation results saved.")



