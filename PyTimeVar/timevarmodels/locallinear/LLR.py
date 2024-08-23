import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import time
import pandas as pd
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.ar_model import AutoReg
from datetime import datetime, timedelta


class LocalLinearRegression:
    """
    Class for performing local linear regression.

    Local linear regression is a non-parametric regression method that
    fits linear models to localized subsets of the data to form a
    regression function. This class uses the Epanechnikov kernel for
    weighting observations.
    Based on the code provided by Yicong Lin [1] and the estimation is based on the methods discussed by Chai [2].

    Parameters
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    mX : np.ndarray
        The independent variable (predictor) matrix.
    h : float
        The bandwidth parameter controlling the size of the local neighborhood.
    tau : np.ndarray, optional
        The array of points at which predictions are made. If not provided,
        it is set to a linear space between 0 and 1 with the same length as `vY`.

    Attributes
    ----------
    h : float
        The bandwidth used for the kernel function.
    times : np.ndarray
        Linearly spaced time points for the local regression.
    vY : np.ndarray
        The response variable array.
    mX : np.ndarray
        The predictor matrix.
    tau : np.ndarray
        Points at which the regression is evaluated.
    n_est : int
        The number of estimated coefficients.

    Methods
    -------
    fit()
        Estimates the coefficients for the local linear regression model.

    Notes
    -----
    This class uses the Epanechnikov kernel function for local smoothing.
    The local linear regression is computed at each point specified in `tau`.
    The bandwidth `h` controls the degree of smoothing.

    References
    ----------
    [1] Yicong Lin, "Local Linear Regression",
    [2] Zongwu Cai,
        Trending time-varying coefficient time series models with serially correlated errors,
        Journal of Econometrics,
        Volume 136, Issue 1,
        2007,
        Pages 163-188,
        ISSN 0304-4076,
        https://doi.org/10.1016/j.jeconom.2005.08.004.

    Examples
    --------
    """

    def __init__(
        self,
        vY: np.ndarray,
        mX: np.ndarray,
        h: float = 0,
        tau: np.ndarray = None,
        kernel: str = "epanechnikov",
        verbose: bool = False,
    ):
        if h == 0:
            self.h = 1.06 * np.std(vY) * len(vY) ** (-1 / 5)
        else:
            self.h = h
        self.times = np.linspace(0, 1, len(vY))
        self.vY = vY
        self.mX = mX
        self.n = len(vY)
        if tau is None:
            self.tau = np.linspace(0, 1, len(vY))
        elif isinstance(tau, np.ndarray):
            self.tau = tau
        elif isinstance(tau, float):
            self.tau = np.array([tau])
        if len(mX.shape) == 1:
            self.n_est = 1
        else:
            self.n_est = np.shape(mX)[1]
        self.kernel = kernel
        self.verbose = verbose

    def _kernel(self, u: np.ndarray) -> np.ndarray:
        """
        Epanechnikov Kernel function for local linear regression.

        Parameters
        ----------
        u : np.ndarray
            An array of scaled distances.

        Returns
        -------
        np.ndarray
            Weighted values computed using the Epanechnikov kernel.
        """
        if self.kernel == "epanechnikov":
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
        elif self.kernel == "gaussian":
            return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        elif self.kernel == "uniform":
            return np.where(np.abs(u) <= 1, 0.5, 0)
        elif self.kernel == "triangular":
            return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)
        elif self.kernel == "quartic":
            return np.where(np.abs(u) <= 1, (15/16) * (1 - u**2)**2, 0)
        elif self.kernel == "tricube":
            return np.where(np.abs(u) <= 1, (70/81) * (1 - np.abs(u)**3)**3, 0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        

    def _compute_Sn(
        self, k: int, h: float, times: np.ndarray, tau: float, mX: np.ndarray
    ) -> np.ndarray:
        """
        Compute the S_n matrix for local linear regression.

        Parameters
        ----------
        k : int
            The power of (t-tau) term.
        h : float
            The bandwidth parameter.
        times : np.ndarray
            Linearly spaced time points for the local regression.
        tau : float
            The point at which the regression is evaluated.
        mX : np.ndarray
            The matrix of regressors.

        Returns
        -------
        np.ndarray
            The calculated S_n matrix.
        """
        u = (times[None, :] - tau) / h
        K_u = self._kernel(u)
        if mX.ndim == 2:
            return np.sum(
                (mX[:, None, :] * mX[:, :, None])
                * np.reshape(
                    ((times[None, :] - tau) ** k) * K_u, newshape=(len(times), 1, 1)
                ),
                axis=0,
            ) / (h)
        elif mX.ndim == 1:
            return np.sum(
                (mX[:, None] * mX[:, None])
                * np.reshape(
                    ((times[None, :] - tau) ** k) * K_u, newshape=(len(times), 1)
                ),
                axis=0,
            ) / (h)

    def _compute_Tn(
        self,
        k: int,
        h: float,
        times: np.ndarray,
        tau: float,
        mX: np.ndarray,
        vY: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the T_n matrix for local linear regression.

        Parameters
        ----------
        k : int
            The power of (t-tau) term.
        h : float
            The bandwidth parameter.
        times : np.ndarray
            Linearly spaced time points for the local regression.
        tau : float
            The point at which the regression is evaluated.
        mX : np.ndarray
            The matrix of regressors.
        vY : np.ndarray
            The response variable.

        Returns
        -------
        np.ndarray
            The calculated T_n matrix.
        """
        u = (times[None, :] - tau) / h
        K_u = self._kernel(u)

        if mX.ndim == 2:
            return np.sum(
                mX[:, :, None]
                * np.reshape(
                    (times[None, :] - tau) ** k * K_u, newshape=(len(times), 1, 1)
                )
                * vY.reshape(len(times), 1, 1),
                axis=0,
            ) / (h)
        elif mX.ndim == 1:
            return np.sum(
                mX[:, None]
                * np.reshape(
                    (times[None, :] - tau) ** k * K_u, newshape=(len(times), 1)
                )
                * vY.reshape(len(times), 1),
                axis=0,
            ) / (h)

    def _construct_S_matrix(
        self, h: float, times: np.ndarray, tau: float, mX: np.ndarray, n_dim: int
    ) -> np.ndarray:
        """
        Constructs the matrix S for local linear regression.

        Parameters
        ----------
        h : float
            The bandwidth parameter.
        times : np.ndarray
            Linearly spaced time points for the local regression.
        tau : float
            The point at which the regression is evaluated.
        mX : np.ndarray
            The matrix of regressors.
        n_dim : int
            The dimension of the S matrix.

        Returns
        -------
        np.ndarray
            The constructed S matrix.
        """
        mS = np.zeros(shape=(n_dim, n_dim))
        Sn0 = self._compute_Sn(0, h, times, tau, mX)
        Sn1 = self._compute_Sn(1, h, times, tau, mX)
        Sn2 = self._compute_Sn(2, h, times, tau, mX)
        size = Sn0.shape[0]

        mS[:size, :size] = Sn0
        mS[:size, size:] = Sn1
        mS[size:, :size] = Sn1.T
        mS[size:, size:] = Sn2
        return mS

    def _construct_T_matrix(
        self,
        h: float,
        times: np.ndarray,
        tau: float,
        mX: np.ndarray,
        vY: np.ndarray,
        n_dim: int,
    ) -> np.ndarray:
        """
        Constructs the matrix T for local linear regression.

        Parameters
        ----------
        h : float
            The bandwidth parameter.
        times : np.ndarray
            Linearly spaced time points for the local regression.
        tau : float
            The point at which the regression is evaluated.
        mX : np.ndarray
            The matrix of regressors.
        vY : np.ndarray
            The response variable.
        n_dim : int
            The dimension of the T matrix.

        Returns
        -------
        np.ndarray
            The constructed T matrix.
        """
        mT = np.zeros(shape=(n_dim, 1))
        Tn0 = self._compute_Tn(0, h, times, tau, mX, vY)
        Tn1 = self._compute_Tn(1, h, times, tau, mX, vY)
        size = Tn0.shape[0]

        if n_dim == 2:
            mT[:size, 0] = Tn0
            mT[size:, 0] = Tn1
        else:
            mT[:size, 0] = Tn0[:, 0]
            mT[size:, 0] = Tn1[:, 0]

        return mT

    def _est_betas(
        self,
        vY: np.ndarray,
        mX: np.ndarray,
        h: float,
        tau: np.ndarray,
        times: np.ndarray,
        n_est: int,
    ) -> np.ndarray:
        """
        Estimate the beta coefficients for local linear regression.

        Parameters
        ----------
        vY : np.ndarray
            The dependent variable vector.
        mX : np.ndarray
            The independent variable matrix.
        h : float
            The bandwidth parameter.
        tau : np.ndarray
            Where to evaluate the regression.
        times : np.ndarray
            The time points.
        n_est : int
            The number of beta coefficients to estimate.

        Returns
        -------
        np.ndarray
            The estimated beta coefficients.
        """
        betahat = np.zeros(shape=(n_est, len(tau)))
        for i in range(len(tau)):
            mS, mT = self._construct_S_matrix(
                h, times, tau[i], mX, 2 * n_est
            ), self._construct_T_matrix(h, times, tau[i], mX, vY, 2 * n_est)
            mul = np.linalg.pinv(mS) @ mT

            for j in range(n_est):
                betahat[j, i] = mul[j] if j < n_est else 0
        return betahat

    def fit(self):
        """
        Fits the local linear regression model to the data.

        Returns
        -------
        Results
            An instance of the Results class containing the fitted model results.
        """
        betahat = self._est_betas(
            self.vY, self.mX, self.h, self.tau, self.times, self.n_est
        )

        if betahat.shape[0] == 1:
            # Only one beta
            predicted_y = betahat[0]
        else:
            # Multiple betas
            predicted_y = np.sum(betahat * self.mX.T, axis=0)

        return self.Results(betahat, self.vY, predicted_y, self)

    ############################################################################################################
    #### Bandwidth Selection
    ############################################################################################################

    def _omega(self, x, tau):
        """
        Calculate the weight for a given data point based on the local linear regression model.

        Parameters
        ----------
        x : float
            The data point for which the weight is calculated.
        tau : float
            The center of the local linear regression model.

        Returns
        -------
        float
            The weight for the given data point.
        """
        return scipy.stats.norm.pdf(x, loc=tau, scale=np.sqrt(0.025))

    def _beta_estimation_lmcv(
        self,
        vY: np.ndarray,
        mX: np.ndarray,
        h: float,
        tau: np.ndarray,
        times: np.ndarray,
        n_est: int,
        lmcv_type: int,
    ) -> np.ndarray:
        """
        Estimate the beta coefficients using the Local Linear Regression method with Leave-Multiple-Covariates-Out (LMCV).

        Parameters
        ----------
        vY : np.ndarray
            The dependent variable vector.
        mX : np.ndarray
            The independent variable matrix.
        h : float
            The bandwidth parameter.
        tau : np.ndarray
            The quantile levels.
        times : np.ndarray
            The time points.
        n_est : int
            The number of beta coefficients to estimate.
        lmcv_type : int
            The type of LMCV method to use (0 for LMCV-0, 1 for LMCV-1, etc.).

        Returns
        -------
        np.ndarray
            The estimated beta coefficients.
        """
        T = len(vY)
        betahat = np.zeros(shape=(n_est, len(tau)))

        if lmcv_type == 0:
            for i in range(len(tau)):
                new_times = np.delete(times, i)
                new_mX = np.delete(mX, i, axis=0)
                new_vY = np.delete(vY, i)

                mS, mT = self._construct_S_matrix(
                    h, new_times, tau[i], new_mX, 2 * n_est
                ), self._construct_T_matrix(
                    h, new_times, tau[i], new_mX, new_vY, 2 * n_est
                )
                mul = np.linalg.pinv(mS) @ mT
                for j in range(n_est):
                    betahat[j, i] = mul[j] if j < n_est else 0
        else:
            for i in range(len(tau)):
                deleted_indices = []
                for j in range(lmcv_type + 1):
                    deleted_indices.append(i - j)
                    deleted_indices.append((i + j) % T)
                new_times = np.delete(times, deleted_indices)
                new_mX = np.delete(mX, deleted_indices, axis=0)
                new_vY = np.delete(vY, deleted_indices)

                mS, mT = self._construct_S_matrix(
                    h, new_times, tau[i], new_mX, 2 * n_est
                ), self._construct_T_matrix(
                    h, new_times, tau[i], new_mX, new_vY, 2 * n_est
                )
                mul = np.linalg.pinv(mS) @ mT
                for j in range(n_est):
                    betahat[j, i] = mul[j] if j < n_est else 0
        return betahat

    def _compute_LMCV_score(
        self, betahat_lmcv: np.ndarray, mX: np.ndarray, vY: np.ndarray, one_tau: float
    ):
        """
        Calculate the Leave-One-Out Modified Cross-Validation (LMCV) score.

        Parameters
        ----------
        betahat_lmcv : np.ndarray
            The estimated coefficients.
        mX : np.ndarray
            The design matrix.
        vY : np.ndarray
            The response variable.
        one_tau : float
            The bandwidth parameter.

        Returns
        -------
        float
            The LMCV score.
        """
        T = len(vY)
        taut = np.arange(1 / T, (T + 1) / T, 1 / T)
        if mX.ndim == 1:
            aa = (vY - (mX * betahat_lmcv)) ** 2
            b = self._omega(taut, one_tau)

            return np.sum(aa * b) / T
        elif mX.ndim == 2:
            aa = (vY - (mX @ betahat_lmcv).diagonal()) ** 2
            b = self._omega(taut, one_tau)

            return np.sum(aa * b) / T

    def _get_optimalh_lmcv(
        self, vY: np.ndarray, mX: np.ndarray, lmcv_type: int, n_est: int
    ):
        """
        Calculates the optimal value of h for local linear regression using Leave-One-Out Cross Validation (LMCV).

        Parameters
        ----------
        vY : np.ndarray
            The dependent variable vector.
        mX : np.ndarray
            The independent variable matrix.
        lmcv_type : int
            The type of LMCV to use.
        n_est : int
            The number of estimations to perform.

        Returns
        -------
        float
            The optimal value of h.
        """
        T = len(vY)

        times = np.arange(1 / T, (T + 1) / T, 1 / T)
        optimal_h_tau = []
        vh = np.arange(0.06, 0.2, 0.005)

        betasss = np.zeros(shape=(len(vh), n_est, T))
        for index, h in enumerate(vh):
            print(f"\r estimating for h = {h} ", end="")
            betasss[index] = self._beta_estimation_lmcv(
                vY, mX, h, times, times, n_est, lmcv_type
            )

        for one_tau in times:
            contain = []
            print(f"\r calculating best h for one_tau = {one_tau} ", end="")
            for index, h in enumerate(vh):
                contain.append(
                    self._compute_LMCV_score(betasss[index], mX, vY, one_tau)
                )
            optimal_h_tau.append(np.argmin(np.array(contain)))

        return vh[min(optimal_h_tau)]

    def _get_lmcv_bandwiths(self, vY: np.ndarray, mX: np.ndarray, n_est: int):
        """
        Calculates the local linear regression bandwidths using Leave-One-Out Cross Validation (LMCV).

        Parameters
        ----------
        vY : np.ndarray
            The dependent variable vector.
        mX : np.ndarray
            The independent variable matrix.
        n_est : int
            The number of estimations to perform.

        Returns
        -------
        list
            A list of bandwidths calculated using LMCV, including the average bandwidth.
        """
        h = []
        for lmcv_type in tqdm([0, 2, 4, 6]):
            h.append(self._get_optimalh_lmcv(vY, mX, lmcv_type, n_est))
        AVG = np.mean(h)
        h.append(AVG)
        return h

    def bLMCV(self):
        """
        Calculate the optimal bandwidth for the local linear regression model using Leave-Multiple-Covariates-Out (LMCV).

        Returns
        -------
        list
            The optimal bandwidths for LMCV 0, 2, 4, 6, AVG.
        """
        return self._get_lmcv_bandwiths(self.vY, self.mX, self.n_est)[-1]

    ############################################################################################################
    #### Bootstrap
    ############################################################################################################

    def AR(self, zhat, T, ic='aic'):
        """
        Estimate the AR model and compute residuals.

        Parameters
        ----------
        zhat : np.ndarray
            Array of predicted values.
        T : int
            Number of observations.

        Returns
        -------
        tuple
            epsilontilde : np.ndarray
                Array of residuals.
            max_lag : int
                Maximum lag order.
            armodel : AutoReg
                Fitted autoregressive model.
            ic : string
                Information criterion to select number of lagsy. Default criterion is AIC
            
        """
        maxp = 10 * np.log10(T)
        arm_selection = ar_select_order(zhat, ic, trend="n", maxlag=int(maxp))

        if arm_selection.ar_lags is None:
            armodel = AutoReg(zhat, trend="n", lags=0).fit()
            max_lag = 0
            epsilonhat = zhat
            epsilontilde = epsilonhat - np.mean(epsilonhat)
        else:
            armodel = arm_selection.model.fit()
            max_lag = max(arm_selection.ar_lags)
            epsilonhat = armodel.resid
            epsilontilde = epsilonhat - np.mean(epsilonhat)

        return epsilontilde, max_lag, armodel

    def _get_Zstar_AR(self, max_lags, armodel, T, epsilonstar):
        """
        Get the Z* values for the AR model.

        Parameters
        ----------
        max_lags : np.ndarray
            Maximum lags.
        armodel : AutoReg
            Fitted autoregressive model.
        T : int
            Number of observations.
        epsilonstar : np.ndarray
            Array of residuals.

        Returns
        -------
        np.ndarray
            Array of Z* values.
        """
        zstar = np.zeros(len(max_lags))

        for i in range(len(max_lags), T):
            ar_component = 0
            for j, lag in enumerate(max_lags):
                lagged_data = zstar[i - lag]
                ar_component += armodel.params[j] * lagged_data

            ar_component += epsilonstar[i + 20 - len(max_lags)]

            zstar = np.append(zstar, ar_component)
        return zstar

    def S_BT(
        self,
        epsilontilde: np.ndarray,
        max_lag: int,
        armodel,
        mX: np.ndarray,
        betatilde: np.ndarray,
        T: int,
    ):
        """
        Sieve Bootstrap
        Calculate the transformed response variable using the local linear regression model.

        Parameters
        ----------
        epsilontilde : np.ndarray
            Array of residuals.
        max_lag : int
            Maximum lag order for the autoregressive model.
        armodel : AutoReg
            Fitted autoregressive model.
        mX : np.ndarray
            Array of exogenous variables.
        betatilde : np.ndarray
            Array of estimated coefficients.
        T : int
            Total number of observations.

        Returns
        -------
        np.ndarray
            Transformed response variable.
        """
        epsilonstar = np.random.choice(epsilontilde, T - max_lag + 50)
        if max_lag == 0:
            zstar = epsilonstar[50:]
            zstar_array = zstar
        else:
            max_lag = np.arange(1, max_lag + 1)
            zstar_array = self._get_Zstar_AR(max_lag, armodel, T, epsilonstar)

        if mX.ndim == 1:
            vYstar = mX * betatilde[0] + zstar_array
            return vYstar
        elif mX.ndim == 2:
            vYstar = (mX @ betatilde + zstar_array).diagonal()
            return vYstar
    
    ######### Autoregressive Bootstrap #########
    def AW_BT(
    self,
    zhat: np.ndarray, 
    mX: np.ndarray, 
    betatilde: np.ndarray,
    T: int,
    h: float,
    gamma: float
    ):
        """
        Autoregressive Wild Bootstrap
        Compute a bootstrap sample using the autoregressive wild bootstrap.

        Parameters
        ----------
        zhat : np.ndarray
            Array of residuals.
        mX : np.ndarray
            Array of exogenous variables.
        betatilde : np.ndarray
            Array of estimated coefficients.
        T : int
            Total number of observations.

        Returns
        -------
        np.ndarray
            Transformed response variable.
        """
        # v_nv_star=np.random.normal(0,np.sqrt(1-gamma**2),200)
        xi_star0 = np.random.normal(0, 1, 1)

        v_xi_star = np.zeros(T)
        v_xi_star[0] = xi_star0
        for i in range(1, T):
            v_xi_star[i] = gamma * v_xi_star[i - 1] + np.random.normal(0, np.sqrt(1 - gamma ** 2))

        zstar = v_xi_star * np.array(zhat)

        vYstar = (mX @ betatilde + zstar).diagonal()

        return vYstar
    
    def W_BT(
    self,
    zhat: np.ndarray,
    mX: np.ndarray,
    betatilde: np.ndarray,
    T: int,
    h: float, #not used
    gamma: float
    ):
      """
      Wild Bootstrap
      Calculate the transformed response variable using the local linear regression model.

      Parameters
      ----------
      zhat : np.ndarray
          Array of residuals.
      mX : np.ndarray
          Array of exogenous variables.
      betatilde : np.ndarray
          Array of estimated coefficients.
      T : int
          Total number of observations.

      Returns
      -------
      np.ndarray
          Transformed response variable.
      """
      #generate zstar by 
      zstar =  zhat * np.random.normal(0, 1, T)

      if mX.ndim == 1:
          vYstar = mX * betatilde[0] + zstar
          return vYstar
      elif mX.ndim == 2:
          vYstar = (mX @ betatilde + zstar).diagonal()
          return vYstar

    def SW_BT(
        self,
        epsilontilde: np.ndarray,
        max_lag: int,
        armodel,
        mX: np.ndarray,
        betatilde: np.ndarray,
        T: int,
    ):
        """
        Sieve Wild Bootstrap
        Calculate the transformed response variable using the local linear regression model.

        Parameters
        ----------
        epsilontilde : np.ndarray
            Array of residuals.
        max_lag : int
            Maximum lag order.
        armodel : AutoReg
            Fitted autoregressive model.
        mX : np.ndarray
            Array of exogenous variables.
        betatilde : np.ndarray
            Array of estimated coefficients.
        T : int
            Total number of observations.

        Returns
        -------
        np.ndarray
            Transformed response variable.
        """
        epsilonstar = epsilontilde * np.random.normal(0, 1, T - max_lag)
        epsilonstar = np.random.choice(epsilonstar, T - max_lag + 50)
        if max_lag == 0:
            zstar = epsilonstar[50:]
            zstar_array = zstar
        else:
            max_lag = np.arange(1, max_lag + 1)
            zstar_array = self._get_Zstar_AR(max_lag, armodel, T, epsilonstar)

        if mX.ndim == 1:
            vYstar = mX * betatilde[0] + zstar_array
            return vYstar
        elif mX.ndim == 2:
            vYstar = (mX @ betatilde + zstar_array).diagonal()
            return vYstar
    
    def LBW_BT(
        self, zhat: np.ndarray, mX: np.ndarray, betatilde: np.ndarray, T: int, h: float, gamma:float
    ):
        """
        Performs the local blockwise wild bootstrap algorithm to generate a bootstrap sample.

        Parameters
        ----------
        zhat : np.ndarray
            The input array of shape (T, 1) containing the original data.
        mX : np.ndarray
            The input array of shape (T, k) containing the covariates.
        betatilde : np.ndarray
            The input array of shape (k,) or (k, 1) containing the estimated coefficients.
        T : int
            The number of observations.
        h : float
            The bandwidth parameter.

        Returns
        -------
        np.ndarray
            The bootstrap sample generated by the LBW_BT algorithm.
        """
        l = int(4.5 * ((T * h) ** (0.25)))
        number_blocks = T - l + 1

        overlapping_blocks = np.zeros(shape=(number_blocks, l, 1))
        for i in range(number_blocks):
            overlapping_blocks[i] = np.array(zhat[i : i + l]).reshape(l, 1)

        zstar = np.zeros(shape=(l * int(np.ceil(T / l)), 1))
        for tau in range(0, T, l):
            local_number_blocks = np.shape(
                overlapping_blocks[max(tau - l, 0) : tau + l]
            )[0]

            random_choice = np.random.choice(np.arange(0, local_number_blocks), 1)

            vWild = np.repeat(np.random.normal(0, 1, 1), l)

            overlapping_blocks_star = (overlapping_blocks[max(tau - l, 0) : tau + l])[
                random_choice
            ]

            zstar[tau : tau + l] = overlapping_blocks_star.reshape(
                l, 1
            ) * vWild.reshape(l, 1)

        zstar = zstar[:T]

        if mX.ndim == 1:
            vYstar = (mX * betatilde[0] + zstar).diagonal()
            return vYstar
        elif mX.ndim == 2:
            vYstar = (mX @ betatilde + zstar).diagonal()
            return vYstar
        
        ############################################################################################################
        #### Confidence Bands
        ############################################################################################################

    def _get_qtau(self, alphap, diff, tau):
        """
        Calculate the quantiles for confidence bands.

        Parameters
        ----------
        alphap : float
            Alpha level for quantiles.
        diff : np.ndarray
            Difference array.
        tau : np.ndarray
            Array of tau values.

        Returns
        -------
        np.ndarray
            Array of quantiles.
        """
        qtau = np.zeros(shape=(2, len(tau)))
        for i in range(len(tau)):
            qtau[0, i] = np.quantile(diff[:, i], alphap / 2)
            qtau[1, i] = np.quantile(diff[:, i], (1 - alphap / 2))
        return qtau

    def ABS_value(self, qtau, diff, tau):
        """
        Calculate the absolute value for quantiles.

        Parameters
        ----------
        qtau : np.ndarray
            Array of quantiles.
        diff : np.ndarray
            Difference array.
        tau : np.ndarray
            Array of tau values.

        Returns
        -------
        float
            Absolute value.
        """
        B = 1299
        check = np.sum(
            (qtau[0][:, None] < diff[:, :, None])
            & (diff[:, :, None] < qtau[1][:, None]),
            axis=1,
        )

        return np.abs((np.sum(np.where(check == len(tau), 1, 0)) / B) - 0.95)

    def min_alphap(self, diff, tau):
        """
        Calculate the minimum alpha value for confidence bands.

        Parameters
        ----------
        diff : np.ndarray
            Difference array.
        tau : np.ndarray
            Array of tau values.

        Returns
        -------
        float
            Minimum alpha value.
        """
        B = 1299
        last = self.ABS_value(self._get_qtau(1 / B, diff, tau), diff, tau)
        for index, alphap in enumerate(np.arange(2, 1299) / 1299):
            qtau = self._get_qtau(alphap, diff, tau)
            value = self.ABS_value(qtau, diff, tau)
            if value <= last:
                last = value
                if index == 63:
                    return 0.05
            else:
                if index == 0:
                    return (index + 1) / B
                if index == 1:
                    return (index) / B
                else:
                    return (index + 1) / B
    def K_ZW_Q(self, vTi_t, h):
        """
        Kernel function for Multiplier Bootstrap quantile.

        Parameters
        ----------
        vTi_t : np.ndarray
            Array of time indices.
        h : float
            Bandwidth parameter.

        Returns
        -------
        np.ndarray
            Array of kernel values.
        """
        return 2 * np.sqrt(2) * self._kernel(np.sqrt(2) * vTi_t / h) - self._kernel(vTi_t / h)
    
    def getQ(self, B, T, taut, h, alpha):
        """
        Calculate quantile for Multiplier Bootstrap samples.

        Parameters
        ----------
        B : int
            Number of bootstrap samples.
        T : int
            Number of observations.
        taut : np.ndarray
            Array of tau values.
        h : float
            Bandwidth parameter.
        alpha : float
            Alpha level for quantiles.

        Returns
        -------
        float
            Quantile value.
        """
        h = h * 2 
        mV = np.random.normal(0, 1, (T, B))
        vQ = np.zeros(B)
        mMu = np.zeros((T, B))
        
        mTi_t = np.repeat(taut, T).reshape((-1, T)) - taut
        dDenum = T * h
        
        for i in range(B):
            for t in range(1, T + 1):
                vMu = mMu[:, i]
                vTi_t = mTi_t[:, t - 1]
                vV = mV[:, i]
                vKernel = self.K_ZW_Q(vTi_t, h)
                vMu[t - 1] = vV @ vKernel
            vQ[i] = max(np.absolute(vMu)) / dDenum
        dQ = np.quantile(vQ, 1 - alpha)
        return dQ   
    
    def getLambda(self, mmDelta, iN, iM, taut, dT, dTau, dGamma):
        """
        Calculate lambda values for Multiplier Bootstrap samples.

        Parameters
        ----------
        mmDelta : np.ndarray
            Array of delta values.
        iN : int
            Number of observations.
        iM : int
            Number of blocks.
        taut : np.ndarray
            Array of tau values.
        dT : float
            Time parameter.
        dTau : float
            Tau parameter.
        dGamma : float
            Gamma parameter.

        Returns
        -------
        np.ndarray
            Array of lambda values.
        """
        if dT <= dGamma:
            dT = dGamma
        if dT >= 1 - dGamma:
            dT = 1 - dGamma
        vTi_t = taut - dT
        vKernel = self._kernel(vTi_t / dTau)
        
        vWeights = vKernel / np.sum(vKernel)
        mLam = np.zeros((mmDelta.shape[0], mmDelta.shape[1]))
        for i in range(len(taut)):
            mLam = mLam + vWeights[i] * mmDelta[:, :, i]
        return mLam   
    
    def getDelta(self, mX, vEps, iM):
        """
        Calculate delta values for Multiplier Bootstrap samples.

        Parameters
        ----------
        mX : np.ndarray
            Array of covariates.
        vEps : np.ndarray
            Array of residuals.
        iM : int
            Number of blocks.

        Returns
        -------
        np.ndarray
            Array of delta values.
        """
        if mX.ndim == 1:
            mX = mX.reshape(-1, 1)
        mX = mX.T
        mL = vEps * mX
        mL = mL.T
        (iP, T) = mX.shape
        dDenum = 2 * iM + 1
        mmDelta = np.zeros((iP, iP, T))
        for i in range(0, T):
            mQi = (np.sum(mL[max(0, i - iM):min(i + iM, T), :], axis=0)).reshape(-1, 1)
            mDeltai = (mQi @ mQi.T) / dDenum
            mmDelta[:, :, i] = mDeltai 
        return mmDelta
    
    def getSigma(self, mM, mLam):
        """
        Calculate sigma values for Multiplier Bootstrap samples.

        Parameters
        ----------
        mM : np.ndarray
            Array of M values.
        mLam : np.ndarray
            Array of lambda values.

        Returns
        -------
        np.ndarray
            Array of sigma values.
        """
        mM_inv = np.linalg.inv(mM)
        vSigma = np.array([mM_inv[d, d] * np.sqrt(mLam[d, d]) for d in range(mLam.shape[0])])
        return vSigma
    
    def getM(self, mX, T, dT, h):
        """
        Calculate M values for Multiplier Bootstrap samples.

        Parameters
        ----------
        mX : np.ndarray
            Array of covariates.
        T : int
            Number of observations.
        dT : float
            Time parameter.
        h : float
            Bandwidth parameter.

        Returns
        -------
        np.ndarray
            Array of M values.
        """
        if mX.ndim == 1:
            mX = mX.reshape(-1, 1)
        vTi_t = np.arange(1, T + 1) / T - dT
        vKernel = self._kernel(vTi_t / h)
        mX_tilde = mX * (np.sqrt(vKernel).reshape(-1, 1))
        mM = mX_tilde.T @ mX_tilde / (T * h)
        return mM
    
    def MC_ZW(self, alpha, h, vY, mX, T):
        """
        Perform the Multiplier Bootstrap bootstrap algorithm.
        Zhou, Z., & Wu, W. B. (2010). Simultaneous inference of linear models with time varying coefficients.
        Journal of the Royal Statistical Society. Series B (Statistical Methodology), 72(4), 513–531. http://www.jstor.org/stable/40802223

        Parameters
        ----------
        h : float
            Bandwidth parameter.
        vY : np.ndarray
            Array of dependent variable values.
        mX : np.ndarray
            Array of covariates.
        T : int
            Number of observations.

        Returns
        -------
        tuple
            Lower and upper confidence bands and beta coefficients.
        """
        if mX.ndim == 1:
            mX = mX.reshape(-1, 1)
        # Initialisation
        B = 3000
        # Parameter specification
        iM = int((T) ** (2 / 7))
        dTau = (T) ** (-1 / 7)
        dGamma = dTau + (iM + 1) / T
        taut = np.arange(1 / T, (T + 1) / T, 1 / T)
        # Estimation
        betahat = self._est_betas(vY, mX, h, taut, taut, self.n_est)
        
        # Jackknife bias-corrected estimator
        h_jk = 2 * h
        betahat_j = self._est_betas(vY, mX, h, taut, taut, self.n_est)
        betahat_k = self._est_betas(vY, mX, h_jk, taut, taut, self.n_est)
        betahat_jk = 2 * betahat_k - betahat_j
        
        # Simulation quantile
        dQ = self.getQ(B, T, taut, h, alpha)
        
        vEps = vY - (mX @ betahat).diagonal()
        mmDelta = self.getDelta(mX, vEps, iM)        
        
        vT_star = np.maximum(np.array([h] * T), np.minimum(taut, 1 - h))
        
        mmSCT = np.zeros((T, 2, mX.shape[1]))
        
        for l in range(T):
            # Construct mMHat(tau)
            mM = self.getM(mX, T, vT_star[l], h)
            
            # Construct mLambdaTilde(tau)
            mLam = self.getLambda(mmDelta, T, iM, taut, taut[l], dTau, dGamma)
            
            # Construct LRV
            vSigma = self.getSigma(mM, mLam)
            for d in range(mX.shape[1]):
                mmSCT[l, :, d] = betahat_jk[d, l] + vSigma[d] * dQ * np.array([-1, 1])
        
        S_LB_beta = [
          mmSCT[:, 0, i] for i in range(self.n_est)
        ]
        S_UB_beta = [
          mmSCT[:, 1, i] for i in range(self.n_est)
        ]
        P_LB_beta = S_LB_beta
        P_UB_beta = S_UB_beta
        return S_LB_beta, S_UB_beta, P_LB_beta, P_UB_beta, betahat
          
            
          
            
    def construct_confidence_bands(self, bootstraptype: str, alpha: float=None, gamma:float=None, ic:str=None, Gsubs: list = None, Chtilde:float=None):
      """
      Construct confidence bands using bootstrap methods.

      Parameters
      ----------
      bootstraptype : str
          Type of bootstrap to use ('SB', 'WB', 'SWB', 'MB', 'LBWB, 'AWB').
      alpha : float
          Significance level for quantiles.                              
      gamma : float
          Parameter value for Autoregressive Wild Bootstrap.
      ic : str
          Type of information criterion to use for Sieve and Sieve Wild Bootstrap.
          Possible values are: 'aic', 'hqic', 'bic'
      Gsubs : list of tuples, optional
          List of sub-ranges for G. Each sub-range is a tuple (start_index, end_index).
          Default is None, which uses the full range (0, T).
      Chtilde : float, optional
          Multiplication constant to determine size of oversmoothing bandwidth htilde.
          Default is 2, if none or negative is specified.

      Returns
      -------
      list of tuples
          Each tuple contains simultaneous and pointwise lower and upper bands for each sub-range,
          and beta coefficients for each sub-range.

      Examples
      --------
      Construct confidence bands using the Sieve Bootstrap method for the full range:
      >>> confidence_bands = model.construct_confidence_bands(bootstraptype='SB')
      >>> S_LB_beta, S_UB_beta, P_LB_beta, P_UB_beta = confidence_bands[0]
      >>> betahat = confidence_bands[1]

      Construct confidence bands using the Local Blockwise Wild Bootstrap method for these ranges [(0, 50), (150, 200)]:
      >>> confidence_bands = model.construct_confidence_bands(bootstraptype='LBWB', Gsubs=[(0, 50), (150, 200)])
      >>> g1_S_LB_beta, g1_S_UB_beta, g1_P_LB_beta, g1_P_UB_beta = confidence_bands[0]
      >>> g2_S_LB_beta, g2_S_UB_beta, g2_P_LB_beta, g2_P_UB_beta = confidence_bands[1]
      >>> betahat = confidence_bands[2]
      """
      if bootstraptype == "MB":
          print("Calculating Multiplier Bootstrap Samples")
          return self.MC_ZW(self.h, self.vY, self.mX, len(self.vY))
      
      if Chtilde is None or Chtilde<=0:
          Chtilde=2

      htilde = Chtilde * (self.h ** (5 / 9))
      T = len(self.vY)
      taut = np.arange(1 / T, (T + 1) / T, 1 / T)
      B = 1299

      if Gsubs is None:
          Gsubs = [(0, T)]

      results = []
      
      if alpha is None or alpha <= 0 or alpha >= 1:
          alpha=0.05

      # Determine the appropriate bootstrap function
      bootstrap_functions = {
          "SB": self.S_BT,
          "WB": self.W_BT,
          "SWB": self.SW_BT,
          "LBWB": self.LBW_BT,
          "AWB": self.AW_BT
      }

      if bootstraptype not in bootstrap_functions:
          raise ValueError("Invalid bootstrap type. Choose one of 'SB','WB', 'SWB','MB' ,'LBWB', 'AWB'")

      bootstrap_function = bootstrap_functions[bootstraptype]
      
      if gamma is None or gamma <= 0 or gamma >= 1:
          l = int(4.5 * ((T * self.h) ** (0.25)))
          gamma=(0.01)**(1/l)

      # Calculate betatilde and betahat once
      betatilde = self._est_betas(self.vY, self.mX, htilde, taut, taut, self.n_est)
      betahat = self._est_betas(self.vY, self.mX, self.h, taut, taut, self.n_est)

      if self.n_est == 1:
          zhat = (self.vY - (self.mX * betatilde[0])).diagonal()
      else:
          zhat = self.vY - (self.mX @ betatilde).diagonal()

      # Initialize storage for bootstrap samples
      betahat_star_G_all = {i: np.zeros((B, self.n_est, end - start)) for i, (start, end) in enumerate(Gsubs)}

      print(f"Calculating {bootstraptype} Bootstrap Samples")
      for i in tqdm(range(B)):
          if bootstraptype in ["SB", "SWB"]:
              epsilonhat, max_lag, armodel = self.AR(zhat, T, ic)
              epsilontilde = epsilonhat - np.mean(epsilonhat)
              vYstar = bootstrap_function(epsilontilde, max_lag, armodel, self.mX, betatilde, T)              
          else:
              vYstar = bootstrap_function(zhat, self.mX, betatilde, T, self.h, gamma)

          for j, (start_index, end_index) in enumerate(Gsubs):
              G = taut[start_index:end_index]
              betahat_star_G_all[j][i] = self._est_betas(vYstar, self.mX, self.h, G, taut, self.n_est)

      for j, (start_index, end_index) in enumerate(Gsubs):
          G = taut[start_index:end_index]
          
          betatilde_G = self._est_betas(self.vY, self.mX, htilde, G, taut, self.n_est)
          betahat_G = self._est_betas(self.vY, self.mX, self.h, G, taut, self.n_est)
          
          diff_beta_G = np.zeros(shape=(self.n_est, B, len(G)))

          for i in range(B):
              diff_G = betahat_star_G_all[j][i] - betatilde_G
              for k in range(self.n_est):
                  diff_beta_G[k][i] = diff_G[k]

          optimal_alphap_G = [self.min_alphap(diff_beta_G[i], G) for i in range(self.n_est)]

          S_LB_beta = [betahat_G[i] - self._get_qtau(optimal_alphap_G[i], diff_beta_G[i], G)[1] for i in range(self.n_est)]
          S_UB_beta = [betahat_G[i] - self._get_qtau(optimal_alphap_G[i], diff_beta_G[i], G)[0] for i in range(self.n_est)]

          P_LB_beta = [betahat_G[i] - self._get_qtau(alpha, diff_beta_G[i], G)[1] for i in range(self.n_est)]
          P_UB_beta = [betahat_G[i] - self._get_qtau(alpha, diff_beta_G[i], G)[0] for i in range(self.n_est)]

          results.append((S_LB_beta, S_UB_beta, P_LB_beta, P_UB_beta))
      results.append(betahat)
      return results
    class Results:
      """
      Class representing the results of a local linear regression model.

      Attributes
      ----------
      model : LocalLinearRegression
          The fitted local linear regression model.
      betahat : np.ndarray
          The estimated beta coefficients.
      vY : np.ndarray
          The actual values of the response variable.
      predicted_y : np.ndarray
          The predicted values of the response variable.
      residuals : np.ndarray
          The residuals of the model.

      Methods
      -------
      summary()
          Print a summary of the regression results.
      plot_betas(date_range=None)
          Plot the beta coefficients over tau.
      plot_actual_vs_predicted(date_range=None)
          Plot the actual values of Y against the predicted values of Y.
      plot_residuals(date_range=None)
          Plot the residuals.
      plot()
          Plot all the available plots (betas, actual vs predicted, residuals).
      betas()
          Get the estimated beta coefficients.
      predicted()
          Get the predicted values of the response variable.
      residuals()
          Get the residuals of the model.
      """

      def __init__(self, betahat, vY, predicted_y, model):
          self.model = model
          self.betahat = betahat
          self.vY = vY
          self.predicted_y = predicted_y
          self.residuals = vY - predicted_y

      def summary(self):
          """
          Print a summary of the regression results.
          """
          print("Local Linear Regression Results")
          print("=" * 30)
          print(f"Bandwidth: {self.model.h}")
          print(f"Number of observations: {len(self.vY)}")
          print(f"Number of predictors: {self.betahat.shape[0]}")
          print("=" * 30)
          print(f"Beta coefficients (shape: {self.betahat.shape}):")
          print("Use the 'betas()' method to get the beta coefficients.")
          print("Use the 'plot_betas()' method to plot the beta coefficients.")
          print("=" * 30)
          print("Use the 'get_confidence_bands()' method to get the confidence bands.")
          print("Use the 'plot_confidence_bands()' method to plot the confidence bands.")
          print("You can choose out of 6 types of Bootstrap to construct confidence bands:")
          print("SB (Sieve Bootstrap), WB (Wild Bootstrap), SWB (Sieve Wild Bootstrap), MB (Multiplier Bootstrap), LBWB (Local Blockwise Wild Bootstrap), AWB (Autoregressive Wild Bootstrap)")
          print("=" * 30)
          print("Use the 'residuals()' method to get the residuals.")
          print("Use the 'plot_residuals()' method to plot the residuals.")

      def _generate_dates(self, length, start_date, end_date):
          # Calculate the total duration in days
          total_days = (end_date - start_date).days

          # Generate a date range with the appropriate number of periods
          dates = pd.date_range(start=start_date, end=end_date, periods=length)

          return dates

      def _format_x_axis(self, ax, date_list):
          def format_func(x, pos):
              if len(date_list) > 0:
                  index = int(x)
                  if 0 <= index < len(date_list):
                      date = date_list[index]
                      if date_list[-1] - date_list[0] > pd.Timedelta(days=730):
                          return date.strftime('%Y')
                      else:
                          return date.strftime('%b %Y')
              return ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

      def plot_betas(self, date_range=None):
          """
          Plot the beta coefficients over a normalized x-axis from 0 to 1.
          """
          num_betas = self.betahat.shape[0]
          fig, axs = plt.subplots(num_betas, 1, figsize=(10, 6))

          # Ensure axs is always an array even if there's only one subplot
          if num_betas == 1:
              axs = [axs]

          if date_range:
              start_date, end_date = [datetime.strptime(date, "%Y-%m-%d") for date in date_range]
              x_vals = self._generate_dates(self.betahat.shape[1], start_date, end_date)
          else:
              x_vals = np.linspace(0, 1, self.betahat.shape[1])

          for i in range(num_betas):
              axs[i].plot(x_vals, self.betahat[i], label=f"Beta {i + 1}")
              axs[i].set_title(f"Beta {i + 1}")
              axs[i].set_xlabel("Date" if date_range else "t/n")
              axs[i].set_ylabel("Beta Value")
              axs[i].legend()
              if date_range:
                  self._format_x_axis(axs[i], x_vals)

          plt.tight_layout()
          plt.show()

      def plot_actual_vs_predicted(self, date_range=None):
          """
          Plot the actual values of Y against the predicted values of Y over a normalized x-axis from 0 to 1.
          """
          plt.figure(figsize=(10, 6))

          if date_range:
              start_date, end_date = [datetime.strptime(date, "%Y-%m-%d") for date in date_range]
              x_vals = self._generate_dates(len(self.vY), start_date, end_date)
          else:
              x_vals = np.linspace(0, 1, len(self.vY))

          plt.plot(x_vals, self.vY, label="Actual Y")
          plt.plot(x_vals, self.predicted_y, label="Predicted Y")
          plt.title("Actual vs Predicted Y")
          plt.xlabel("Date" if date_range else "t/n")
          plt.ylabel("Y Value")
          plt.legend()

          if date_range:
              ax = plt.gca()
              self._format_x_axis(ax, x_vals)

          plt.show()

      def plot_residuals(self, date_range=None):
          """
          Plot the residuals over a normalized x-axis from 0 to 1.
          """
          plt.figure(figsize=(10, 6))

          if date_range:
              start_date, end_date = [datetime.strptime(date, "%Y-%m-%d") for date in date_range]
              x_vals = self._generate_dates(len(self.residuals), start_date, end_date)
          else:
              x_vals = np.linspace(0, 1, len(self.residuals))

          plt.plot(x_vals, self.residuals, label="Residuals")
          plt.title("Residuals")
          plt.xlabel("Date" if date_range else "t/n")
          plt.ylabel("Residual Value")
          plt.legend()

          if date_range:
              ax = plt.gca()
              self._format_x_axis(ax, x_vals)

          plt.show()

      def plot_confidence_bands(self, bootstrap_type: str = 'LBWB', Gsubs=None, date_range=None):
          """
          Plot the beta coefficients with confidence bands over a normalized x-axis from 0 to 1.

          Parameters
          ----------
          bootstrap_type : str, optional
              The type of bootstrap to use for constructing confidence bands (default is 'LBWB').
          Gsubs : list of tuples, optional
              List of tuples specifying the subsample ranges for G. If None, plot for the whole sample.
          date_range : tuple of str, optional
              Tuple containing start and end dates in 'YYYY-MM-DD' format.
          """
          # Construct confidence bands
          if Gsubs is None:
              confidence_bands_list = self.model.construct_confidence_bands(bootstrap_type)
              confidence_bands = confidence_bands_list[:-1][0]
              betahat = confidence_bands_list[-1]
              if date_range:
                  start_date, end_date = [datetime.strptime(date, "%Y-%m-%d") for date in date_range]
                  G_full = self._generate_dates(len(self.vY), start_date, end_date)
              else:
                  G_full = np.linspace(0, 1, len(self.vY))
          else:
              confidence_bands_dict = self.model.construct_confidence_bands(bootstrap_type, Gsubs=Gsubs)
              confidence_bands_list = confidence_bands_dict[:-1]
              betahat = confidence_bands_dict[-1]
              if date_range:
                  start_date, end_date = [datetime.strptime(date, "%Y-%m-%d") for date in date_range]
                  G_full = self._generate_dates(len(self.vY), start_date, end_date)
              else:
                  G_full = np.linspace(0, 1, len(self.vY))
          
          # Number of beta coefficients
          n_betas = self.betahat.shape[0]

          # Plotting
          if Gsubs is None:
              plt.figure(figsize=(6.5, 5 * n_betas))
              for j in range(n_betas):
                  S_LB_beta = confidence_bands[0][j]
                  S_UB_beta = confidence_bands[1][j]
                  P_LB_beta = confidence_bands[2][j]
                  P_UB_beta = confidence_bands[3][j]

                  plt.subplot(n_betas, 1, j + 1)
                  plt.plot(G_full, betahat[j], label=f'Estimated beta {j}', color='black')
                  plt.plot(G_full, S_LB_beta, 'r--', label='Simultaneous LB')
                  plt.plot(G_full, S_UB_beta, 'r--', label='Simultaneous UB')
                  plt.fill_between(G_full, P_LB_beta, P_UB_beta, color='grey', alpha=0.3, label='Pointwise CB')
                  plt.title(f'{bootstrap_type} - beta {j}')
                  plt.xlabel('Date' if date_range else 't/n')
                  plt.ylabel('Beta')
                  plt.legend()
                  if date_range:
                      ax = plt.gca()
                      self._format_x_axis(ax, G_full)

              plt.tight_layout()
              plt.show()
          else:
              fig, axes = plt.subplots(n_betas, 1, figsize=(6.5, 5 * n_betas))

              if n_betas == 1:
                  axes = [axes]

              # Plot betahat for the full range and confidence bands for each beta coefficient
              for j in range(n_betas):
                  ax = axes[j]
                  ax.plot(G_full, betahat[j], label=f'beta {j} estimate', color='black')

                  for i, (start_index, end_index) in enumerate(Gsubs):
                      G = G_full[start_index:end_index]
                      S_LB_beta, S_UB_beta, P_LB_beta, P_UB_beta = confidence_bands_list[i]

                      ax.plot(G, S_LB_beta[j], 'r--', label='Simultaneous LB' if i == 0 else "")
                      ax.plot(G, S_UB_beta[j], 'r--', label='Simultaneous UB' if i == 0 else "")
                      ax.fill_between(G, P_LB_beta[j], P_UB_beta[j], color='grey', alpha=0.3, label='Pointwise CB' if i == 0 else "")

                  ax.set_title(f'Beta {j} Estimates and Confidence Bands for {bootstrap_type}')
                  ax.set_xlabel('Date' if date_range else 't/n')
                  ax.set_ylabel('Beta')
                  ax.legend()
                  if date_range:
                      self._format_x_axis(ax, G_full)

              plt.tight_layout()
              plt.show()
          return confidence_bands_list

      def betas(self):
          """
          Get the estimated beta coefficients.

          Returns
          -------
          np.ndarray
              The estimated beta coefficients.
          """
          return self.betahat

      def predicted(self):
          """
          Get the predicted values of the response variable.

          Returns
          -------
          np.ndarray
              The predicted values of the response variable.
          """
          return self.predicted_y

      def residuals(self):
          """
          Get the residuals of the model.

          Returns
          -------
          np.ndarray
              The residuals of the model.
          """
          return self.residuals
