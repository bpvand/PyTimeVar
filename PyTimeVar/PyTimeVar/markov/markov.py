import numpy as np
import matplotlib.pyplot as plt


class MarkovSwitching:
    """
    Class for Markov-switching coefficient estimation.
    
    Parameters
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    mX : np.ndarray
        The independent variable (predictor) matrix.
    iS : int
        The number of regimes.
    niter : int
        The number of EM repetitions.
    conv_iter : int
        The number of EM iterations until convergence.
        
    Attributes
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    mX : np.ndarray
        The independent variable (predictor) matrix.
    iS : int
        The number of regimes.
    niter : int
        The number of EM repetitions.
    conv_iter : int
        The number of EM iterations until convergence.
    n_est : int
        The number of regressors.
    n : int
        The sample size.
    mBetaHat : np.ndarray
        The matrix of estimated coefficients.
    dS2_hat : float
        The estimated noise variance.
    mP_hat : np.ndarray
        The estimated transition matrix.
    vProb_pred : np.ndarray
        The array of predicted switching probabilities.
    vProb_filt : np.ndarray
        The array of filtered switching probabilities.
    vProb_smooth : np.ndarray
        The array of smoothed switching probabilities.
    

    """

    def __init__(self, vY: np.ndarray, mX: np.ndarray, iS: int = 2, niter: int = 10, conv_iter: int = 20):
        self.vY = vY.flatten()
        self.mX = mX
        self.iS = iS
        self.niter = niter
        self.conv_iter = conv_iter
        
        
        self.n = len(vY)
        self.n_est = np.shape(mX)[1]
        
        self.mBetaHat = None
        self.dS2_hat = None
        self.mP_hat = None
        self.vProb_pred = None
        self.vProb_filt = None
        self.vProb_smooth = None
        
    def _conditional_pdf_state(self, vBeta, dS2, iState, t):
        """
        Calculates the conditional probability density function for a given state.
        """
        mean = self.mX[t, :] @ vBeta[:, iState]
        return (1 / np.sqrt(2 * np.pi * dS2)) * np.exp(-(self.vY[t] - mean)**2 / (2 * dS2))

    def _inference_est(self, vLikelihood, vXi_pred_t):
        """
        Calculates the filtered probabilities.
        """
        vNumerator = vLikelihood * vXi_pred_t
        return vNumerator / np.sum(vNumerator)
        
    def _Estep(self, vBeta, dS2, mP):
        # Initialize
        mProb_filt = np.zeros((self.iS, self.n))
        mProb_pred = np.zeros((self.iS, self.n + 1))
        mProb_smooth = np.zeros((self.iS, self.n))

        # Initialization of predicted probabilities
        mProb_pred[:, 0] = np.ones(self.iS) / self.iS  # Uniform initial probabilities

        # Filtering
        for t in range(self.n):
            # 1. Prediction step
            mProb_pred[:, t + 1] = mP.T @ mProb_filt[:, t] if t > 0 else mP.T @ mProb_pred[:, 0]

            # 2. Calculate likelihood of observation y_t given each state
            vLikelihood = np.array([self._conditional_pdf_state(vBeta, dS2, s, t) for s in range(self.iS)])

            # 3. Filtering step
            mProb_filt[:, t] = self._inference_est(vLikelihood, mProb_pred[:, t + 1])

        # Smoothing (Kim's algorithm)
        mProb_smooth[:, -1] = mProb_filt[:, -1]
        for t in range(self.n - 2, -1, -1):
            vNumerator = mP @ mProb_smooth[:, t + 1]
            vDenominator = mProb_pred[:, t + 1]
            vRatio = vNumerator / vDenominator
            mProb_smooth[:, t] = mProb_filt[:, t] * vRatio

        return mProb_filt, mProb_pred, mProb_smooth
    
    def _Mstep(self, mProb_filt, mProb_pred, mProb_smooth, mP_prev):
        """
        M-step of the EM algorithm.
        """
        # Update coefficients (closed form solution for linear regression)
        vBeta_new = np.zeros((self.n_est, self.iS))
        dS2_new = 0

        for s in range(self.iS):
            weights = mProb_smooth[s, :]
            weighted_X = self.mX * weights[:, np.newaxis]
            vBeta_new[:, s] = np.linalg.solve(self.mX.T @ weighted_X, self.mX.T @ (weights * self.vY))

            residuals = self.vY - self.mX @ vBeta_new[:, s]
            dS2_new += np.sum(weights * residuals**2)

        dS2_new /= self.n

        # Update transition probabilities (closed form solution)
        mP_new = np.zeros((self.iS, self.iS))
        for i in range(self.iS):
            for j in range(self.iS):
                numerator = np.sum(mProb_smooth[i, :-1] * mP_prev[i, j] * (mProb_smooth[j, 1:] / mProb_pred[j, 1:]))
                denominator = np.sum(mProb_smooth[i, :-1])
                if denominator > 0:
                    mP_new[i, j] = numerator / denominator
                else:
                    mP_new[i, j] = 1 / self.iS # If a state is never visited, use uniform probability

        # Ensure rows of P sum to 1
        for i in range(self.iS):
            mP_new[i, :] /= np.sum(mP_new[i, :])

        return vBeta_new, dS2_new, mP_new
    
    def _EMLoss(self, mProb_smooth, vBeta, dS2, mP):
        """
        Calculates the approximate log-likelihood (can be used for monitoring convergence).
        """
        loglik = 0
        for t in range(self.n):
            likelihood_t = 0
            for s in range(self.iS):
                likelihood_t += mProb_smooth[s, t] * self._conditional_pdf_state(vBeta, dS2, s, t)
            if likelihood_t > 0:
                loglik += np.log(likelihood_t)
        return -loglik
        
    def fit(self):
        """
        Fits the Markov switching model using the EM algorithm with multiple repetitions.
        """
        best_beta = None
        best_sigma2 = None
        best_P = None
        best_smoothed_probs = None
        best_loglik = np.inf

        for _ in range(self.niter):
            # Initialize parameters randomly for each repetition
            vBeta = np.random.normal(0, 1, size=(self.n_est, self.iS))
            dS2 = np.var(self.vY)
            mP = np.random.uniform(0, 1, size=(self.iS, self.iS))
            mP = mP / np.sum(mP, axis=1, keepdims=True)  # Normalize rows to sum to 1

            prev_loglik = -np.inf
            loglik_diff = np.inf

            for it in range(self.conv_iter):
                # E-step
                mProb_filt, mProb_pred, mProb_smooth = self._Estep(vBeta, dS2, mP)

                # M-step
                vBeta_new, dS2_new, mP_new = self._Mstep(mProb_filt, mProb_pred, mProb_smooth, mP)

                # Check for convergence (using log-likelihood)
                current_loglik = self._EMLoss(mProb_smooth, vBeta_new, dS2_new, mP_new)
                loglik_diff = current_loglik - prev_loglik
                prev_loglik = current_loglik

                vBeta = vBeta_new
                dS2 = dS2_new
                mP = mP_new

                if np.abs(loglik_diff) < 1e-6:
                    print(f"Convergence reached in repetition {_ + 1} at iteration {it + 1}")
                    break
                elif it == self.conv_iter - 1:
                    print(f"Convergence not reached in repetition {_ + 1} after {self.conv_iter} iterations.")

            # Store the best results across repetitions based on log-likelihood
            if current_loglik < best_loglik:
                best_loglik = current_loglik
                best_beta = vBeta
                best_sigma2 = dS2
                best_P = mP
                best_smoothed_probs = mProb_smooth

        self.mBetaHat = best_beta
        self.dS2_hat = best_sigma2
        self.mP_hat = best_P
        self.vProb_pred = None  # These are within each EM run, not the final best
        self.vProb_filt = None
        self.vProb_smooth = best_smoothed_probs

        return best_beta, best_sigma2, best_P, best_smoothed_probs
    
    def plot_state_probabilities(self, smoothed=True):
        """
        Plots the filtered or smoothed state probabilities over time.

        Parameters
        ----------
        smoothed : bool, optional
            If True, plots the smoothed probabilities. Otherwise, plots the
            filtered probabilities. Defaults to True.
        """
        if self.vProb_smooth is None or self.vProb_filt is None:
            raise ValueError("Model has not been fitted yet. Run the 'fit' method first.")

        plt.figure(figsize=(12, 6))
        if smoothed:
            probabilities = self.vProb_smooth
            title = "Smoothed State Probabilities"
        else:
            probabilities = self.vProb_filt
            title = "Filtered State Probabilities"

        t = np.arange(1, self.n + 1)/self.n
        for s in range(self.iS):
            plt.plot(t, probabilities[s, :], label=f'Regime {s+1}')

        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_coefficients(self, tau: list = None):
        '''
        Plot the beta coefficients over a normalized x-axis from 0 to 1,
        showing the coefficient corresponding to the estimated regime at each time point,
        and add vertical lines for inferred regime changes.
    
        Parameters
        ----------
        tau : list, optional
            The list looks the  following: tau = [start,end].
            The function will plot all data and estimates between start and end.
    
        Raises
        ------
        ValueError
            No valid tau is provided.
    
        '''
        tau_index = None
        x_vals = np.arange(1 / self.n, (self.n + 1) / self.n, 1 / self.n)
    
        if tau is None:
            tau_index = np.array([0, self.n])
        elif isinstance(tau, list):
            if min(tau) <= 0:
                tau_index = np.array([int(0), int(max(tau) * self.n)])
            else:
                tau_index = np.array([int(min(tau) * self.n - 1), int(max(tau) * self.n)])
        else:
            raise ValueError('The optional parameter tau is required to be a list.')
    
        estimated_regimes = np.argmax(self.vProb_smooth[:, :], axis=0)
        coefficient_paths = np.zeros((self.n, self.n_est))
        for t in range(self.n):
            coefficient_paths[t, :] = self.mBetaHat[:, estimated_regimes[t]]
    
        # Infer regime changes
        regime_change_indices = np.where(np.diff(estimated_regimes))[0] + 1
        regime_change_normalized = regime_change_indices / self.n
    
        # Handle single vs multiple regressors
        if self.n_est == 1:
    
            plt.figure(figsize=(12, 6))
            plt.plot(x_vals[tau_index[0]:tau_index[1]], self.vY[tau_index[0]:tau_index[1]],
                     label="True data", linewidth=1, color='black')
            plt.plot(x_vals[tau_index[0]:tau_index[1]], self.mBetaHat[tau_index[0]:tau_index[1]],
                     label="Estimated $\\beta_{0}$", linestyle="--", linewidth=2)
    
            # Add vertical lines for break dates
            for break_point in regime_change_normalized:
                if tau is None or (min(tau) <= break_point <= max(tau)):
                    plt.axvline(x=break_point, color='r', linestyle='--', linewidth=0.8, label='Break Date')
    
            plt.grid(linestyle='dashed')
            plt.xlabel('$t/n$', fontsize="xx-large")
            plt.tick_params(axis='both', labelsize=16)
            plt.legend(fontsize="x-large")
            plt.show()
    
        else:
            plt.figure(figsize=(10, 6 * self.n_est))
            for i in range(self.n_est):
                plt.subplot(self.n_est, 1, i + 1)
                plt.plot(x_vals[tau_index[0]:tau_index[1]],
                         (self.mBetaHat[:, i])[tau_index[0]:tau_index[1]],
                         label=f'Estimated $\\beta_{i}$', color='black', linewidth=2)
    
                # Add vertical lines for break dates
                for break_point in regime_change_normalized:
                    if tau is None or (min(tau) <= break_point <= max(tau)):
                        plt.axvline(x=break_point, color='r', linestyle='--', linewidth=0.8, label='Break Date')
    
                plt.grid(linestyle='dashed')
                plt.xlabel('$t/n$', fontsize="xx-large")
                plt.tick_params(axis='both', labelsize=16)
                plt.legend(fontsize="x-large")
            plt.show()