import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.ar_model import AutoReg


class LocalLinear:
    """
    Class for performing local linear regression (LLR).

    Local linear regression is a non-parametric regression method that fits linear models to localized subsets of the data to form a
    regression function. The code is based on the code provided by Lin et al. [1].

    Parameters
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    mX : np.ndarray
        The independent variable (predictor) matrix.
    h : float
        The bandwidth parameter controlling the size of the local neighborhood.
        If not provided, it is estimated as the average of all methods in the package.
    bw_selection : string
        The name of the bandwidth selection method to be used.
        Choice between 'aic', 'gcv', 'lmcv-l' with l=0,2,4,6,etc., or 'all'.
        If not provided, it is set to 'all'.
    tau : np.ndarray
        The array of points at which predictions are made. 
        If not provided, it is set to a linear space between 0 and 1 with the same length as `vY`.
    kernel : string
        The name of the kernel function used for estimation.
        If not provided, it is set to 'epanechnikov' for the Epanechnikov kernel.
    LB_bw : float
        The lower bound for the bandwidth selection.
        If not provided, it is set to 0.06.
    UB_bw : float
        The upper bound for the bandwidth selection.
        If not provided, it is set to 0.2 for LMCV-l and to 0.7 for AIC and GCV.

    Attributes
    ----------
    vY : np.ndarray
        The response variable array.
    mX : np.ndarray
        The predictor matrix.
    n : int
        The length of vY.
    times : np.ndarray
        Linearly spaced time points for the local regression.
    tau : np.ndarray
        Points at which the regression is evaluated.
    n_est : int
        The number of coefficients.
    kernel : string
        The name of the kernel function.
    bw_selection : string
        The name of the bandwidth selection.
    lmcv_type : float
        If LMCV is used for bandwidth selection, this attribute denotes the l in leave-2*l+1-out.
    dict_bw : dict
        The dictionary that contains the optimnal bandwidth values for each individual method.
    h : float
        The bandwidth used for local linear regression.
    betahat : np.ndarray
        The estimated coefficients.
    predicted_y : np.ndarray
        The fitted values for the response variable.
    residuals : np.ndarray
        The residuals resulting from the local linear regression.
        
    Raises
    ------
    ValueError
        No valid bandwidth selection procedure is provided.

    Notes
    -----
    The local linear regression is computed at each point specified in `tau`.
    The bandwidth `h` controls the degree of smoothing.

    References
    ----------
    [1] Lin Y, Song M, van der Sluis B (2024), 
        Bootstrap inference for linear time-varying coefficient models in locally stationary time series,
        Journal of Computational and Graphical Statistics,
        Forthcoming.

    """

    def __init__(
        self,
        vY: np.ndarray,
        mX: np.ndarray,
        h: float = 0,
        bw_selection: str = None,
        kernel: str = "epanechnikov",
        LB_bw: float = None,
        UB_bw: float = None
    ):

        self.vY = vY.reshape(-1,1)
        self.mX = mX
        self.n = len(vY)
        self.times = np.arange(1 / self.n, ( self.n + 1) /  self.n, 1 /  self.n)

        self.tau = np.arange(1 / self.n, (self.n + 1) / self.n, 1 / self.n)
        # self.tau_index = np.array([0, self.n])

        # elif isinstance(tau, float):
        #     self.tau = np.array([tau])
        self.tau_bw_selection = np.arange(1/self.n,(self.n+1)/self.n,1/self.n)

        if mX.ndim == 1:
            self.mX = mX.reshape(-1, 1)
            self.n_est = 1
        elif mX.ndim == 2:
            if np.shape(mX)[1] == 1:
                self.n_est = 1
            else:
                self.n_est = np.shape(mX)[1]

        self.kernel = kernel.lower()
        self.h_gcv = None
        self.bw_selection = bw_selection
        self.lmcv_type = None
        if h == 0:
            if self.bw_selection is None:
                print(
                    'No bandwidth or selection method is specified.')
                self.bw_selection = 'all'
            else:
                self.bw_selection= self.bw_selection.lower()

            if self.bw_selection not in ['all', 'aic', 'gcv']:
                if self.bw_selection[:4] != 'lmcv':
                    raise ValueError(
                        'Bandwidth selection method is invalid. \nPlease provide an expression from the following options: [\'all\', \'aic\', \'gcv\'] or use \'lmcv_l\'')
                else:
                    self.lmcv_type=self.bw_selection[5]

                    # if self.lmcv_type not in ['0','2','4','6']:
                    #     self.lmcv_type = self.bw_selection


            if bw_selection is None:
                self.dict_bw = self._bandwidth_selection(LB_bw, UB_bw)
                print('\n')
                print("=" * 60)
                print('Optimal bandwidth selected by individual method:')
                print('- AIC method:', format(self.dict_bw['aic'], '.4f'))
                print('- GCV method: ', format(self.dict_bw['gcv'], '.4f'))
                print('- LMCV-0 method: ', format(self.dict_bw['0'],'.4f'))
                print('- LMCV-2 method: ', format(self.dict_bw['2'],'.4f'))
                print('- LMCV-4 method: ', format(self.dict_bw['4'],'.4f'))
                print('- LMCV-6 method: ', format(self.dict_bw['6'],'.4f'))
                print("=" * 60)
                self.dict_bw['all'] = np.array(list(self.dict_bw.values())).mean()
                self.h = self.dict_bw['all']
                self.h_gcv = self.dict_bw['gcv']
                print(f'Optimal bandwidth used is the avg. of all methods: {self.h: .4f}')
                print("=" * 60)
                print(
                    f'Note: (1) For constructing confidence intervals/bands using the residual-based bootstrap method, the avg. bandwidth of all methods {self.h: .4f} is used; (2) For constructing confidence intervals/bands using the MB method, the GCV bandwidth {self.h_gcv: .4f} is used.')
                # print(f'If the MB is implemented, the GCV bandwidth {self.the_gcv_h: .4f} is used.')


            elif self.bw_selection[:4] == 'lmcv':
                self.h = self._bandwidth_selection(LB_bw, UB_bw)
                print('\n')
                print(f'- LMCV-{self.lmcv_type} method: ', format(self.h,'.4f'))
                print(f'Optimal bandwidth used is {self.bw_selection}: {self.h: .4f}')
                print("=" * 60)
                print('Note: For constructing confidence intervals/bands using the MB method, a GCV bandwidth is recommended.\n')
                # print(
                #     f'Note: If a residual-based bootstrap method (LBWB, WB, SB, SWB, AWB) is adopted, the {self.bw_selection} bandwidth {self.h: .4f} is used.')
                # print(f'If the MB is implemented, the GCV bandwidth {self.the_gcv_h: .4f} is used.')

            # self.h = self.dict_bw[self.bw_selection]
            elif self.bw_selection == 'all':
                self.dict_bw = self._bandwidth_selection(LB_bw, UB_bw)
                print('\n')
                print("=" * 60)
                print('Optimal bandwidth selected by individual method:')
                print('- AIC method:', format(self.dict_bw['aic'], '.4f'))
                print('- GCV method: ', format(self.dict_bw['gcv'], '.4f'))
                print('- LMCV-0 method: ', format(self.dict_bw['0'], '.4f'))
                print('- LMCV-2 method: ', format(self.dict_bw['2'], '.4f'))
                print('- LMCV-4 method: ', format(self.dict_bw['4'], '.4f'))
                print('- LMCV-6 method: ', format(self.dict_bw['6'], '.4f'))
                print("=" * 60)
                self.dict_bw['all'] = np.array(list(self.dict_bw.values())).mean()
                self.h = self.dict_bw['all']
                print(f'Optimal bandwidth used is the avg. of all methods: {self.h: .4f}')
                print("=" * 60)
                print('Note: For constructing confidence intervals/bands using the MB method, a GCV bandwidth is recommended.\n')
                # print(f'If the MB is implemented, the GCV bandwidth {self.the_gcv_h: .4f} is used.')

            elif self.bw_selection == 'gcv' or 'aic':
                self.h  = self._bandwidth_selection(LB_bw, UB_bw)
                print(f'Optimal bandwidth used is {self.bw_selection}: {self.h: .4f}')
                print("=" * 60)
                if self.bw_selection == 'gcv':
                    print(
                        'Note: For constructing confidence intervals/bands using the residual-based bootstrap method, a LMCV bandwidth is recommended.\n')

                else:
                    print(
                        'Note: For constructing confidence intervals/bands using the MB method, a GCV bandwidth is recommended.\n')

        else:
            self.h = h
            print(f'Bandwidth specified by user: {self.h: .4f}\n')
            print('Note: this bandwidth will be used for any bootstrap implementation.\n')
            
        self.betahat = None
        self.predicted_y = None
        self.residuals = None

    def _kernel(self, u: np.ndarray) -> np.ndarray:
        """
        Kernel function for local linear regression.

        Parameters
        ----------
        u : np.ndarray
            An array of scaled distances.
            
        Raises
        ------
        ValueError
            No valid kernel name is provided.

        Returns
        -------
        np.ndarray
            Weighted values computed using the self.kernel function.
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
            Array of points where to evaluate the regression.
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
        self.betahat : np.ndarray
            The estimated coefficients.
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
            
        # store results in corresponding attributes
        self.betahat = betahat
        self.predicted_y = (predicted_y.reshape(-1,1))
        self.residuals = (self.vY) - self.predicted_y

        return self.betahat

    ############################################################################################################
    # Bandwidth Selection
    ############################################################################################################

    def _omega(self, x, tau):
        """
        Calculate the LMCV weight for a given data point based on the local linear regression model.

        Parameters
        ----------
        x : float
            The data point for which the weight is calculated.
        tau : float
            The center of the local linear regression model.

        Returns
        -------
        float
            The LMCV weight for the given data point.
        """
        return scipy.stats.norm.pdf(x, loc=tau, scale=np.sqrt(0.025))

    def _beta_estimation_lmcv(self, h, tau, lmcv_type) -> np.ndarray:
        """
        Estimate the beta coefficients using the Local Linear Regression method with Local-Modified-Cross-Validation (LMCV).

        Parameters
        ----------
        h : float
            The bandwidth parameter.
        tau : np.ndarray
            The quantile levels that are evaluated.
        lmcv_type : int
            The type of LMCV method to use (0 for LMCV-0, 2 for LMCV-2, etc.).

        Returns
        -------
        np.ndarray
            The estimated beta coefficients.
        """
        betahat = np.zeros(shape=(self.n_est, len(tau)))

        if lmcv_type == 0:
            for i in range(self.n):
                new_times = np.delete(self.times, i)
                new_mX = np.delete(self.mX, i, axis=0)
                new_vY = np.delete(self.vY, i)

                mS, mT = self._construct_S_matrix(
                    h, new_times, self.tau_bw_selection[i], new_mX, 2 * self.n_est
                ), self._construct_T_matrix(
                    h, new_times, self.tau_bw_selection[i], new_mX, new_vY, 2 * self.n_est
                )
                mul = np.linalg.pinv(mS) @ mT
                for j in range(self.n_est):
                    betahat[j, i] = mul[j] if j < self.n_est else 0
        else:
            for i in range(self.n):
                deleted_indices = []
                for j in range(int(lmcv_type) + 1):
                    deleted_indices.append(i - j)
                    deleted_indices.append((i + j) % self.n)
                new_times = np.delete(self.times, deleted_indices)
                new_mX = np.delete(self.mX, deleted_indices, axis=0)
                new_vY = np.delete(self.vY, deleted_indices)

                mS, mT = self._construct_S_matrix(
                    h, new_times, self.tau_bw_selection[i], new_mX, 2 * self.n_est
                ), self._construct_T_matrix(
                    h, new_times, self.tau_bw_selection[i], new_mX, new_vY, 2 * self.n_est
                )
                mul = np.linalg.pinv(mS) @ mT
                for j in range(self.n_est):
                    betahat[j, i] = mul[j] if j < self.n_est else 0
        return betahat

    def _compute_LMCV_score(self, betahat_lmcv: np.ndarray, one_tau: float):
        """
        Calculate the Local-Modified-Cross-Validation (LMCV) score.

        Parameters
        ----------
        betahat_lmcv : np.ndarray
            The estimated coefficients using LMCV.
        one_tau : float
            The time point.

        Returns
        -------
        float
            The LMCV score.
        """

        taut = np.arange(1 / self.n, (self.n + 1) / self.n, 1 / self.n)

        aa = (self.vY - (self.mX @ betahat_lmcv).diagonal()) ** 2
        b = self._omega(taut, one_tau)

        return np.sum(aa * b) / self.n

    def _get_optimalh_lmcv(self, lmcv_type, LB_bw, UB_bw):
        """
        Calculates the optimal value of h for local linear regression using Local-Modified-Cross-Validation (LMCV).

        Parameters
        ----------
        lmcv_type : int
            The type of LMCV to use.
        LB_bw : float
            The lower bound for the bandwidth selection.
        UB_bw : float
            The upper bound for the bandwidth selection

        Returns
        -------
        float
            The optimal value of h.
        """

        optimal_h_tau = []
        
        vh = np.arange(LB_bw, UB_bw, 0.005)
        
        betasss = np.zeros(shape=(len(vh), self.n_est, self.n))
        print('Progress LMCV:')
        for index, h in enumerate(tqdm(vh)):
            betasss[index] = self._beta_estimation_lmcv(
                h, tau=self.times, lmcv_type=lmcv_type)

        for one_tau in self.times:
            contain = []
            for index, h in enumerate(vh):
                contain.append(
                    self._compute_LMCV_score(betasss[index], one_tau)
                )
            optimal_h_tau.append(np.argmin(np.array(contain)))

        return vh[min(optimal_h_tau)]

    def _get_lmcv_bandwiths(self, LB_bw, UB_bw):
        """
        Calculates the local linear regression bandwidths using Local-Modified-Cross-Validation (LMCV-l) for different values of l.
        
        Parameters
        ----------
        LB_bw : float
            The lower bound for the bandwidth selection.
        UB_bw : float
            The upper bound for the bandwidth selection
        
        Returns
        -------
        h : list
            A list of bandwidths calculated using LMCV, including the average bandwidth.
        """
        if self.bw_selection != 'all':
            return self._get_optimalh_lmcv(self.lmcv_type, LB_bw, UB_bw)
        h = []
        lmcv_types = [0, 2, 4, 6]
        for r in range(len(lmcv_types)):
            lmcv_type = lmcv_types[r]
            h.append(self._get_optimalh_lmcv(lmcv_type, LB_bw, UB_bw))
        AVG = np.mean(h)
        h.append(AVG)
        if self.lmcv_type is not None:
            h.append(self._get_optimalh_lmcv(int(self.lmcv_type), LB_bw, UB_bw))
        return h

    def _AICmodx(self, h):
        '''
        Computes the AIC value for a given mean squared error and trace.

        Parameters
        ----------
        h : float
            The bandwidth parameter.

        Returns
        -------
        float
            AIC value.

        '''
        s2, traceh = self._get_loss_aic_gcv(h)
        return np.log(s2) + 2*(traceh+1)/(self.n-traceh-2)

    def _get_loss_aic_gcv(self, h):
        '''
        Computes the mean squared error and corresponding trace, for a given bandwidth.

        Parameters
        ----------
        h : float
            The bandwidth parameter.

        Returns
        -------
        s2 : float
            The sample variance of residuals.
        traceh : float
            The trace of projection matrix Q_h in yhat(h) = Q_h y.

        '''

        vYhat = np.zeros(self.n)
        traceh = 0
        for i in range(self.n):
            mS, mT = self._construct_S_matrix(
                h, self.times, self.tau_bw_selection[i], self.mX, 2 * self.n_est
            ), self._construct_T_matrix(h, self.times, self.tau_bw_selection[i], self.mX, self.vY, 2 * self.n_est)
            mul = np.linalg.pinv(mS) @ mT
            betahath = mul[:self.n_est]
            vYhat[i] = (self.mX[i, :]@betahath)[:]

            vE = self._get_vE_vector(h, self.tau_bw_selection[i], i)
            vHatMatSelect = np.linalg.pinv(mS) @ vE
            traceh = traceh + self.mX[i, :] @ vHatMatSelect[:self.n_est]

        s2 = np.mean((self.vY-vYhat)**2)

        return s2, traceh

    def _get_vE_vector(self, h, tau, i):
        """
        Constructs the vector vE for AIC and GCV bandwidth selection.

        Parameters
        ----------
        h : float
            The bandwidth parameter.
        tau : float
            The point at which the regression is evaluated.
        i : float
            The index at which the regression is evaluated.

        Returns
        -------
        np.ndarray
            The constructed E vector.
        """
        mE = np.zeros(shape=(2*self.n_est, self.n))
        En0 = self._compute_En(0, h, self.times, tau, self.mX)
        En1 = self._compute_En(1, h, self.times, tau, self.mX)
        size = En0.shape[0]

        if self.n_est == 1:
            mE[:size, 0] = En0
            mE[size:, 0] = En1
        else:
            mE[:size, 0] = En0[:, 0]
            mE[size:, 0] = En1[:, 0]

        vEx = np.zeros((self.n, 1))
        vEx[i] = 1

        vE = (mE @ vEx).flatten()

        return vE

    def _compute_En(
        self,
        k: int,
        h: float,
        times: np.ndarray,
        tau: float,
        mX: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the E_n vector for AIC and GCV bandwidth selection.

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
            The calculated E_n vector.
        """
        u = (times[None, :] - tau) / h
        K_u = self._kernel(u)

        if mX.ndim == 2:
            return np.sum(mX[:, :, None] * np.reshape((times[None, :] - tau) ** k * K_u, newshape=(len(times), 1, 1)), axis=0,) / (h)
        elif mX.ndim == 1:
            return np.sum(mX[:, None] * np.reshape((times[None, :] - tau) ** k * K_u, newshape=(len(times), 1)), axis=0,) / (h)

    def _get_aic_bandwidth(self,LB_bw, UB_bw):
        '''
        Calculates the LLR optimal bandwidth by minimizing modified AIC. 
        
        Parameters
        ----------
        LB_bw : float
            The lower bound for the bandwidth selection.
        UB_bw : float
            The upper bound for the bandwidth selection

        Returns
        -------
        h_opt
            Optimal bandwidth value

        '''
        res = scipy.optimize.minimize(self._AICmodx, 0.1, bounds=[(LB_bw, UB_bw)])
        h_opt = res.x[0]

        return h_opt

    def _GCVmodx(self, h):
        '''
        Computes the GCV value for a bandwidth value.

        Parameters
        ----------
        h : float
            The bandwidth parameter.

        Returns
        -------
        float
            GCV value.

        '''
        s2, traceh = self._get_loss_aic_gcv(h)
        return s2/(1-traceh/self.n)**2

    def _get_gcv_bandwidth(self,LB_bw, UB_bw):
        '''
        Calculates the LLR optimal bandwidth using Generalized CV (GCV)
        
        Parameters
        ----------
        LB_bw : float
            The lower bound for the bandwidth selection.
        UB_bw : float
            The upper bound for the bandwidth selection

        Returns
        -------
        h_opt
            Optimal bandwidth value

        '''        
        res = scipy.optimize.minimize(self._GCVmodx, 0.1, bounds=[(LB_bw, UB_bw)])
        h_opt = res.x[0]

        return h_opt

    def _bandwidth_selection(self, LB_bw, UB_bw):
        """
        Calculate the optimal bandwidth for the local linear regression model using LMCV, AIC and GCV.
        
        Parameters
        ----------
        LB_bw : float
            The lower bound for the bandwidth selection.
        UB_bw : float
            The upper bound for the bandwidth selection

        Returns
        -------
        dict
            The optimal bandwidths for each individual method.
        """
        d = {}
        if LB_bw is None:
            LB_bw = 0.06
        if UB_bw is None:
            UB_bw_aic_gcv = 0.7
            UB_bw_lmcv = 0.2
        d['aic'] = self._get_aic_bandwidth(LB_bw, UB_bw_aic_gcv)
        d['gcv'] = self._get_gcv_bandwidth(LB_bw, UB_bw_aic_gcv)
        if self.bw_selection == 'aic':
            return d['aic']
        if self.bw_selection == 'gcv':
            return d['gcv']
        if self.bw_selection[:4] == 'lmcv':
            return self._get_lmcv_bandwiths(LB_bw, UB_bw_lmcv)
        list_h = self._get_lmcv_bandwiths(LB_bw, UB_bw_lmcv)
        d['0'] = list_h[0]
        d['2'] = list_h[1]
        d['4'] = list_h[2]
        d['6'] = list_h[3]
        if self.lmcv_type is not None:
            d[self.lmcv_type] = list_h[-1]

        return d

    ############################################################################################################
    # Bootstrap
    ############################################################################################################

    def _AR(self, zhat, T, ic=None):
        """
        Estimate the AR model and compute residuals.

        Parameters
        ----------
        zhat : np.ndarray
            Array of predicted values.
        T : int
            Number of observations.
        ic : string
            Information criterion to select number of lags. Default criterion is AIC

        Returns
        -------
        epsilonhat: np.ndarray
            Array of residuals.
        max_lag : int
            Maximum lag order.
        armodel : AutoReg
            Fitted autoregressive model.

        """
        if ic is None:
            ic = "aic"
        maxp = 10 * np.log10(T)
        arm_selection = ar_select_order(
            endog=zhat, maxlag=int(maxp), ic=ic, trend="n")

        if arm_selection.ar_lags is None:
            armodel = AutoReg(zhat, trend="n", lags=0).fit()
            max_lag = 0
            epsilonhat = zhat
        else:
            armodel = arm_selection.model.fit()
            max_lag = max(arm_selection.ar_lags)
            epsilonhat = armodel.resid

        return epsilonhat, max_lag, armodel

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
        zstar : np.ndarray
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

    def _S_BT(
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
        Compute a bootstrap sample using the sieve bootstrap.

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
        vYstar : np.ndarray
            The bootstrap sample generated by the _S_BT algorithm.
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
    def _AW_BT(
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
        h : float
            The bandwidth parameter.
        gamma : float
            The AR(1) coefficient and the standard deviation of the wild component in the AWB.

        Returns
        -------
        vYstar : np.ndarray
            The bootstrap sample generated by the _AW_BT algorithm.
        """
        # v_nv_star=np.random.normal(0,np.sqrt(1-gamma**2),200)
        xi_star0 = np.random.normal(0, 1, 1)

        v_xi_star = np.zeros(T)
        v_xi_star[0] = xi_star0
        for i in range(1, T):
            v_xi_star[i] = gamma * v_xi_star[i - 1] + \
                np.random.normal(0, np.sqrt(1 - gamma ** 2))

        zstar = v_xi_star * np.array(zhat)

        vYstar = (mX @ betatilde + zstar).diagonal()

        return vYstar

    def _W_BT(
        self,
        zhat: np.ndarray,
        mX: np.ndarray,
        betatilde: np.ndarray,
        T: int,
        h: float,  # not used
        gamma: float
    ):
        """
        Wild Bootstrap
        Compute a bootstrap sample using the wild bootstrap.

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
        h : float
            The bandwidth parameter.
        gamma : float
            The AR(1) coefficient and the standard deviation of the wild component in the AWB.    

        Returns
        -------
        vYstar : np.ndarray
            The bootstrap sample generated by the _W_BT algorithm.
        """
        # generate zstar by
        zstar = zhat * np.random.normal(0, 1, T)

        if mX.ndim == 1:
            vYstar = mX * betatilde[0] + zstar
            return vYstar
        elif mX.ndim == 2:
            vYstar = (mX @ betatilde + zstar).diagonal()
            return vYstar

    def _SW_BT(
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
        Compute a bootstrap sample using the sieve wild bootstrap.

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
        vYstar : np.ndarray
            The bootstrap sample generated by the _SW_BT algorithm.
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

    def _LBW_BT(
        self, zhat: np.ndarray, mX: np.ndarray, betatilde: np.ndarray, T: int, h: float, gamma: float
    ):
        """
        Local Blockwise Wild Bootstrap
        Compute a bootstrap sample using the local blockwise wild bootstrap.

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
        gamma : float
            The AR(1) coefficient and the standard deviation of the wild component in the AWB.

        Returns
        -------
        vYstar : np.ndarray
            The bootstrap sample generated by the _LBW_BT algorithm.
        """
        l = int(4.5 * ((T * h) ** (0.25)))
        number_blocks = T - l + 1

        overlapping_blocks = np.zeros(shape=(number_blocks, l, 1))
        for i in range(number_blocks):
            overlapping_blocks[i] = np.array(zhat[i: i + l]).reshape(l, 1)

        zstar = np.zeros(shape=(l * int(np.ceil(T / l)), 1))
        for tau in range(0, T, l):
            local_number_blocks = np.shape(
                overlapping_blocks[max(tau - l, 0): tau + l]
            )[0]

            random_choice = np.random.choice(
                np.arange(0, local_number_blocks), 1)

            vWild = np.repeat(np.random.normal(0, 1, 1), l)

            overlapping_blocks_star = (overlapping_blocks[max(tau - l, 0): tau + l])[
                random_choice
            ]

            zstar[tau: tau + l] = overlapping_blocks_star.reshape(
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
        # Confidence Bands
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

    def _ABS_value(self, qtau, diff, tau, alpha, B):
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
        alpha : float
            The significance level.
        B : int
            The number of bootstrap samples.
        
        Returns
        -------
        float
            Absolute value for quantiles.
        """
        
        check = np.sum(
            (qtau[0][:, None] < diff[:, :, None])
            & (diff[:, :, None] < qtau[1][:, None]),
            axis=1,
        )

        return np.abs((np.sum(np.where(check == len(tau), 1, 0)) / B) - (1 - alpha))

    def _min_alphap(self, diff, tau, alpha, B):
        """
        Calculate the minimum alpha value for confidence bands.

        Parameters
        ----------
        diff : np.ndarray
            Difference array.
        tau : np.ndarray
            Array of tau values.
        alpha : float
            The significance level.
        B : int
            The number of bootstrap samples.
        
        Returns
        -------
        float
            Minimum alpha value.
        """
        
        last = self._ABS_value(self._get_qtau(
            1 / B, diff, tau), diff, tau, alpha, B)
        for index, alphap in enumerate(np.arange(2, int(B * alpha) + 2) / B):
            qtau = self._get_qtau(alphap, diff, tau)
            value = self._ABS_value(qtau, diff, tau, alpha, B)
            if value <= last:
                last = value
                if index == int(B * alpha) - 1:
                    return alpha
            else:
                if index == 0:
                    return (index + 1) / B
                else:
                    return (index) / B

    def _K_ZW_Q(self, vTi_t, h):
        """
        Kernel function for Multiplier Bootstrap quantile.

        Parameters
        ----------
        vTi_t : np.ndarray
            The array of time indices.
        h : float
            The bandwidth parameter.

        Returns
        -------
        np.ndarray
            Array of kernel values.
        """
        return 2 * np.sqrt(2) * self._kernel(np.sqrt(2) * vTi_t / h) - self._kernel(vTi_t / h)

    def _getQ(self, B, T, taut, h, alpha):
        """
        Calculate quantile for Multiplier Bootstrap samples.

        Parameters
        ----------
        B : int
            The number of bootstrap samples.
        T : int
            the number of observations.
        taut : np.ndarray
            The points at which the regression is evaluated.
        h : float
            Bandwidth parameter.
        alpha : float
            Alpha level for quantiles.

        Returns
        -------
        dQ : float
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
                vKernel = self._K_ZW_Q(vTi_t, h)
                vMu[t - 1] = vV @ vKernel
            vQ[i] = max(np.absolute(vMu)) / dDenum
        dQ = np.quantile(vQ, 1 - alpha)
        return dQ

    def _getLambda(self, mmDelta, iN, iM, taut, dT, dTau, dGamma):
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

    def _getDelta(self, mX, vEps, iM):
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
            mQi = (
                np.sum(mL[max(0, i - iM):min(i + iM, T), :], axis=0)).reshape(-1, 1)
            mDeltai = (mQi @ mQi.T) / dDenum
            mmDelta[:, :, i] = mDeltai
        return mmDelta

    def _getSigma(self, mM, mLam):
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
        vSigma = np.array([mM_inv[d, d] * np.sqrt(mLam[d, d])
                          for d in range(mLam.shape[0])])
        return vSigma

    def _getM(self, mX, T, dT, h):
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

    def _MC_ZW(self, alpha, h, vY, mX, T, B):
        """
        Perform the Multiplier Bootstrap bootstrap algorithm.
        Zhou, Z., & Wu, W. B. (2010). Simultaneous inference of linear models with time varying coefficients.
        Journal of the Royal Statistical Society. Series B (Statistical Methodology), 72(4), 513–531. http://www.jstor.org/stable/40802223

        Parameters
        ----------
        alpha : 
            The significance level for the bootstrap.
        h : float
            Bandwidth parameter.
        vY : np.ndarray
            Array of dependent variable values.
        mX : np.ndarray
            Array of covariates.
        T : int
            Number of observations.
        B : int
            Number of bootstrap iterations.

        Returns
        -------
        tuple
            Lower and upper confidence bands and beta coefficients.
        """

        if mX.ndim == 1:
            mX = mX.reshape(-1, 1)
        vY = vY.flatten()

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
        betahat_k = self._est_betas(vY, mX, h_jk/np.sqrt(2), taut, taut, self.n_est)
        betahat_jk = 2 * betahat_k - betahat_j

        # Simulation quantile
        dQ = self._getQ(B, T, taut, h, alpha)

        vEps = vY - (mX @ betahat).diagonal()
        mmDelta = self._getDelta(mX, vEps, iM)

        vT_star = np.maximum(np.array([h] * T), np.minimum(taut, 1 - h))

        mmSCT = np.zeros((T, 2, mX.shape[1]))

        for l in range(T):
            # Construct mMHat(tau)
            mM = self._getM(mX, T, vT_star[l], h)

            # Construct mLambdaTilde(tau)
            mLam = self._getLambda(mmDelta, T, iM, taut, taut[l], dTau, dGamma)

            # Construct LRV
            vSigma = self._getSigma(mM, mLam)
            for d in range(mX.shape[1]):
                mmSCT[l, :, d] = betahat_jk[d, l] + \
                    vSigma[d] * dQ * np.array([-1, 1])

        S_LB_beta = [
            mmSCT[:, 0, i] for i in range(self.n_est)
        ]
        S_UB_beta = [
            mmSCT[:, 1, i] for i in range(self.n_est)
        ]
        P_LB_beta = S_LB_beta
        P_UB_beta = S_UB_beta
        results = []
        results.append((S_LB_beta, S_UB_beta, P_LB_beta, P_UB_beta))
        results.append(betahat_jk)
        return results

    def _construct_confidence_bands(self, bootstraptype: str, alpha: float = None, gamma: float = None, ic: str = None, Gsubs: list = None, Chtilde: float = None, B: float = 1299):
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
        Gsubs : list of tuples
            List of sub-ranges for G. Each sub-range is a tuple (start_index, end_index).
            Default is None, which uses the full range (0, T).
        Chtilde : float
            Multiplication constant to determine size of oversmoothing bandwidth htilde.
            Default is 2, if none or negative is specified.
        B : int
            The number of bootstrap samples.
            Deafult is 1299, if not provided by the user.
            
        Raises
        ------
        ValueError
            No valid bootstrap type is provided.

        Returns
        -------
        list of tuples
            Each tuple contains simultaneous and pointwise lower and upper bands for each sub-range,
            and beta coefficients for each sub-range.
        """
        if alpha is None or alpha <= 0 or alpha >= 1:
            alpha = 0.05
        
        if self.h_gcv is not None:
            if bootstraptype == 'MB':
                self.h = self.h_gcv
            else:
                self.h = self.h
        else:
            self.h = self.h

        if bootstraptype == "MB":
            print("Calculating MB Bootstrap Samples")
            return self._MC_ZW(alpha, self.h, self.vY, self.mX, len(self.vY), B)

        if Chtilde is None or Chtilde <= 0:
            Chtilde = 2

        htilde = Chtilde * (self.h ** (5 / 9))
        T = len(self.vY)
        taut = np.arange(1 / T, (T + 1) / T, 1 / T)

        if Gsubs is None:
            Gsubs = [(0, T)]

        results = []


        # Determine the appropriate bootstrap function
        bootstrap_functions = {
            "SB": self._S_BT,
            "WB": self._W_BT,
            "SWB": self._SW_BT,
            "LBWB": self._LBW_BT,
            "AWB": self._AW_BT
        }
        
        if B < 1299:
            print('Note: It is recommended to use at least B = 1299 iterations.\n')

        if bootstraptype not in bootstrap_functions:
            raise ValueError(
                "Invalid bootstrap type. Choose one of 'SB','WB', 'SWB','MB' ,'LBWB', 'AWB'")

        bootstrap_function = bootstrap_functions[bootstraptype]

        if gamma is None or gamma <= 0 or gamma >= 1:
            l = int(4.5 * ((T * self.h) ** (0.25)))
            gamma = (0.01)**(1/l)

        # Calculate betatilde and betahat once
        betatilde = self._est_betas(
            self.vY, self.mX, htilde, taut, taut, self.n_est)
        betahat = self._est_betas(
            self.vY, self.mX, self.h, taut, taut, self.n_est)

        zhat = self.vY - (self.mX @ betatilde).diagonal().reshape(-1, 1)

        # Initialize storage for bootstrap samples
        betahat_star_G_all = {i: np.zeros(
            (B, self.n_est, end - start)) for i, (start, end) in enumerate(Gsubs)}

        print(f"Calculating {bootstraptype} Bootstrap Samples")
        for i in tqdm(range(B)):
            if bootstraptype in ["SB", "SWB"]:
                epsilonhat, max_lag, armodel = self._AR(zhat, T, ic)
                epsilontilde = epsilonhat - np.mean(epsilonhat)
                if bootstraptype == 'SWB':
                    vYstar = bootstrap_function(
                        epsilonhat, max_lag, armodel, self.mX, betatilde, T)
                elif bootstraptype == 'SB':
                    vYstar = bootstrap_function(
                        epsilontilde, max_lag, armodel, self.mX, betatilde, T)
            else:
                vYstar = bootstrap_function(
                    zhat, self.mX, betatilde, T, self.h, gamma)

            for j, (start_index, end_index) in enumerate(Gsubs):
                G = taut[start_index:end_index]
                betahat_star_G_all[j][i] = self._est_betas(
                    vYstar, self.mX, self.h, G, taut, self.n_est)

        for j, (start_index, end_index) in enumerate(Gsubs):
            G = taut[start_index:end_index]

            betatilde_G = self._est_betas(
                self.vY, self.mX, htilde, G, taut, self.n_est)
            betahat_G = self._est_betas(
                self.vY, self.mX, self.h, G, taut, self.n_est)

            diff_beta_G = np.zeros(shape=(self.n_est, B, len(G)))

            for i in range(B):
                diff_G = betahat_star_G_all[j][i] - betatilde_G
                for k in range(self.n_est):
                    diff_beta_G[k][i] = diff_G[k]

            optimal_alphap_G = [self._min_alphap(
                diff_beta_G[i], G, alpha, B) for i in range(self.n_est)]

            S_LB_beta = [betahat_G[i] - self._get_qtau(optimal_alphap_G[i], diff_beta_G[i], G)[
                1] for i in range(self.n_est)]
            S_UB_beta = [betahat_G[i] - self._get_qtau(optimal_alphap_G[i], diff_beta_G[i], G)[
                0] for i in range(self.n_est)]

            P_LB_beta = [
                betahat_G[i] - self._get_qtau(alpha, diff_beta_G[i], G)[1] for i in range(self.n_est)]
            P_UB_beta = [
                betahat_G[i] - self._get_qtau(alpha, diff_beta_G[i], G)[0] for i in range(self.n_est)]

            results.append((S_LB_beta, S_UB_beta, P_LB_beta, P_UB_beta))
        results.append(betahat)
        return results

    def summary(self):
        """
        Print a summary of the regression results.
        """
        print("Local Linear Regression Results")
        print("=" * 60)
        print(f'Kernel: {self.kernel}')
        if self.bw_selection is None:
            print(f'Bandwidth specified by user: {self.h: .4f}')
        elif self.bw_selection == 'all':
            print('Bandwidth selection method: AVG')
            print(f"Bandwidth used for estimation: {self.h: .4f}")
        else:
            print(f'Bandwidth selection method: {self.bw_selection.upper()}')
            print(f"Bandwidth used for estimation: {self.h: .4f}")
        print(f"Number of observations: {len(self.vY)}")
        print(f"Number of predictors: {self.betahat.shape[0]}")
        print("=" * 60)
        print(f"Beta coefficients (shape: {self.betahat.shape}):")
        print("Use the 'plot_betas()' method to plot the beta coefficients.")
        print("=" * 60)
        print(
            "Use the 'confidence_bands()' method to obtain the confidence bands and plots.")
        print(
            "You can choose out of 6 types of Bootstrap to construct confidence bands:")
        print("SB, WB, SWB, MB, LBWB, AWB")
        print("=" * 60)
        print("Use the 'plot_residuals()' method to plot the residuals.\n")


    def plot_betas(self, tau: list = None):

        """
        Plot the beta coefficients over a normalized x-axis from 0 to 1.
        
        Parameters
        ----------
        tau : list, optional
            The list looks the  following: tau = [start,end].
            The function will plot all data and estimates between start and end.
            
        Raises
        ------
        ValueError
            No valid tau is provided.
        
        """
        tau_index = None
        if tau is None:
            tau_index=np.array([0,self.n])
        elif isinstance(tau, list):
            if min(tau) <= 0:
                tau_index = np.array([int(0), int(max(tau) * self.n)])
            else:
                tau_index = np.array([int(min(tau)*self.n-1),int(max(tau)*self.n)])
        else:
            raise ValueError('The optional parameter tau is required to be a list.')
            
        x_vals = np.arange(1/self.n,(self.n+1)/self.n,1/self.n)
        
        if self.n_est == 1:
    
            plt.figure(figsize=(12, 6))
            plt.plot(x_vals[tau_index[0]:tau_index[1]], self.vY[tau_index[0]:tau_index[1]], label="True data", linewidth=1,color='black')
            plt.plot(x_vals[tau_index[0]:tau_index[1]], self.betahat.reshape(-1,1)[tau_index[0]:tau_index[1]], label="Estimated $\\beta_{0}$", linestyle="--", linewidth=2)
            
            plt.grid(linestyle='dashed')
            plt.xlabel('$t/n$',fontsize="xx-large")
            plt.tick_params(axis='both', labelsize=16)
            plt.legend(fontsize="x-large")
            plt.show()

        else:
            plt.figure(figsize=(10, 6 * self.n_est))
            for i in range(self.n_est):
                plt.subplot(self.n_est, 1, i + 1)
                plt.plot(x_vals[tau_index[0]:tau_index[1]], self.betahat[i][tau_index[0]:tau_index[1]],
                            label=f'Estimated $\\beta_{i}$', linestyle="--", color='black')

                
                plt.grid(linestyle='dashed')
                plt.xlabel('$t/n$',fontsize="xx-large")
                plt.tick_params(axis='both', labelsize=16)
                plt.legend(fontsize="x-large")
            plt.show()
    
    def plot_predicted(self, tau : list=None):

        """
        Plot the actual values of Y against the predicted values of Y over a normalized x-axis from 0 to 1.
        
        Parameters
        ----------
        tau : list, optional
            The list looks the  following: tau = [start,end].
            The function will plot all data and estimates between start and end.
        
        Raises
        ------
        ValueError
            No valid tau is provided.
        
        """
        tau_index = None
        if tau is None:
            tau_index=np.array([0,self.n])
        elif isinstance(tau, list):
            if min(tau) <= 0:
                tau_index = np.array([int(0), int(max(tau) * self.n)])
            else:
                tau_index = np.array([int(min(tau)*self.n-1),int(max(tau)*self.n)])
        else:
            raise ValueError('The optional parameter tau is required to be a list.')
            
        plt.figure(figsize=(12, 6))

        x_vals = np.arange(1/self.n,(self.n+1)/self.n,1/self.n)
        
        

        plt.plot(x_vals[tau_index[0]:tau_index[1]], self.vY[tau_index[0]:tau_index[1]], label="True data", linewidth=1,color='black')
        plt.plot(x_vals[tau_index[0]:tau_index[1]], self.predicted_y[tau_index[0]:tau_index[1]], label="Fit", linestyle="--",  linewidth=2)        
        plt.grid(linestyle='dashed')
        plt.xlabel('$t/n$',fontsize="xx-large")
        plt.tick_params(axis='both', labelsize=16)
        plt.legend(fontsize="x-large")
        
        plt.show()

    def plot_residuals(self, tau : list=None):
        '''
        Plot the residuals over a normalized x-axis from 0 to 1.
        
        Parameters
        ----------
        tau : list, optional
            The list looks the  following: tau = [start,end].
            The function will plot all data and estimates between start and end.
        
        Raises
        ------
        ValueError
            No valid tau is provided.
        
        Returns
        -------
        self.residuals : np.ndarray
            Array of residuals.

        '''
        plt.figure(figsize=(12, 6))

        x_vals = np.arange(1/self.n,(self.n+1)/self.n,1/self.n)
        tau_index = None
        if tau is None:
            tau_index=np.array([0,self.n])
        elif isinstance(tau, list):
            if min(tau) <= 0:
                tau_index = np.array([int(0), int(max(tau) * self.n)])
            else:
                tau_index = np.array([int(min(tau)*self.n-1),int(max(tau)*self.n)])
        else:
            raise ValueError('The optional parameter tau is required to be a list.')

        plt.plot(x_vals[tau_index[0]:tau_index[1]], self.residuals[tau_index[0]:tau_index[1]],  linestyle="--", label="Residuals")       
        plt.grid(linestyle='dashed')
        plt.xlabel('$t/n$',fontsize="xx-large")
        plt.tick_params(axis='both', labelsize=16)
        plt.legend(fontsize="x-large")
        
        plt.show()

        return self.residuals

    

    def confidence_bands(self, bootstrap_type: str = 'LBWB', alpha: float = None,
                         gamma: float = None, ic: str = None, Gsubs=None,
                         Chtilde: float = 2, B: float = 1299, plots: bool = False):
        '''
        

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
        Gsubs : list of tuples
            List of sub-ranges for G. Each sub-range is a tuple (start_index, end_index).
            Default is None, which uses the full range (0, T).
        Chtilde : float
            Multiplication constant to determine size of oversmoothing bandwidth htilde.
            Default is 2, if none or negative is specified.
        B : int
            The number of bootstrap samples.
            Deafult is 1299, if not provided by the user.
        plots : bool
            If True, plots are shown of the estimated coefficients and corresponding confidence bands.
            

        Returns
        -------
        S_LB : np.ndarray
            The lower simultaneous confidence bands.
        S_UB : np.ndarray
            The upper simultaneous confidence bands.
        P_LB : np.ndarray
            The lower pointwise confidence intervals.
        P_UB : np.ndarray
            The upper pointwise confidence intervals.

        '''
        bootstrap_type = bootstrap_type.upper()
        
        # Construct confidence bands
        if Gsubs is None:
            confidence_bands_list = self._construct_confidence_bands(
                bootstrap_type, alpha=alpha, gamma=gamma, ic=ic, Chtilde=Chtilde, B=B)
            confidence_bands = confidence_bands_list[:-1][0]
            betahat = confidence_bands_list[-1]
            S_LB = confidence_bands[0]
            S_UB = confidence_bands[1]
            P_LB = confidence_bands[2]
            P_UB = confidence_bands[3]
            G_full = np.linspace(0, 1, len(self.vY))
        else:
            confidence_bands_dict = self._construct_confidence_bands(
                bootstrap_type, alpha=alpha, gamma=gamma, ic=ic, Gsubs=Gsubs, Chtilde=Chtilde, B=B)
            confidence_bands_list = confidence_bands_dict[:-1]
            betahat = confidence_bands_dict[-1]
            G_full = np.linspace(0, 1, len(self.vY))
            for i, (start_index, end_index) in enumerate(Gsubs):
                S_LB, S_UB, P_LB, P_UB = confidence_bands_list[i]

        if plots == True:
            # Number of beta coefficients
            n_betas = self.betahat.shape[0]

            # Plotting
            if Gsubs is None:
                plt.figure(figsize=(10, 6 * n_betas))
                for j in range(n_betas):
                    S_LB_beta = confidence_bands[0][j]
                    S_UB_beta = confidence_bands[1][j]
                    P_LB_beta = confidence_bands[2][j]
                    P_UB_beta = confidence_bands[3][j]

                    plt.subplot(n_betas, 1, j + 1)
                    plt.plot(
                        G_full, betahat[j], label=f'Estimated $\\beta_{j}$', color='black')
                    plt.plot(G_full, S_LB_beta, 'r--',
                             label='Simultaneous')
                    plt.plot(G_full, S_UB_beta, 'r--')
                    plt.fill_between(
                        G_full, P_LB_beta, P_UB_beta, color='grey', alpha=0.3, label='Pointwise')
                    # plt.title(f'{bootstrap_type} - beta {j}')
                    
                    plt.grid(linestyle='dashed')
                    plt.xlabel('$t/n$',fontsize="xx-large")
                    plt.tick_params(axis='both', labelsize=16)
                    plt.legend(fontsize="x-large")
                    
                plt.show()
            else:
                fig, axes = plt.subplots(
                    n_betas, 1, figsize=(10, 6 * n_betas))

                if n_betas == 1:
                    axes = [axes]

                # Plot betahat for the full range and confidence bands for each beta coefficient
                for j in range(n_betas):
                    ax = axes[j]
                    ax.plot(
                        G_full, betahat[j], label=f'Estimated $\\beta_{j}$', color='black')

                    for i, (start_index, end_index) in enumerate(Gsubs):
                        G = G_full[start_index:end_index]
                        S_LB_beta, S_UB_beta, P_LB_beta, P_UB_beta = confidence_bands_list[i]

                        ax.plot(
                            G, S_LB_beta[j], 'r--', label='Simultaneous' if i == 0 else "")
                        ax.plot(G, S_UB_beta[j], 'r--')
                        ax.fill_between(
                            G, P_LB_beta[j], P_UB_beta[j], color='grey', alpha=0.3, label='Pointwise' if i == 0 else "")

                    ax.set_xlabel('$t/n$',fontsize="xx-large")
                    ax.tick_params(axis='both', labelsize=16)
                    ax.legend(fontsize="x-large")
                    ax.grid(linestyle='dashed')

                plt.show()

        return S_LB, S_UB, P_LB, P_UB
