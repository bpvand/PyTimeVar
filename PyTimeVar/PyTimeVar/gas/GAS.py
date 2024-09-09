import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import time
import matplotlib.pyplot as plt


class GAS:
    """
    Class for performing score-driven (GAS) filtering.
    
    Parameters
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    mX : np.ndarray
        The independent variable (predictor) matrix.
    method : string
        Method to estimate GAS model.
    vgamma0 : np.ndarray 
        Initial parameter vector.
    bounds : list
        List to define parameter space.
    options : dict
        Stopping criteria for optimization.
    maxiter : int
        Maximum number of repitions of the optimization algorithm.
        If not provided, default is set to five repitions with different initial parameters.
        
    Attributes
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    mX : np.ndarray
        The independent variable (predictor) matrix.
    n : int
        The length of vY.
    n_est : int
        The number of coefficients.
    method : string
        Method to estimate GAS model.
    vgamma0 : np.ndarray 
        The initial parameter vector.
    maxiter : int
        Maximum number of repitions of the optimization algorithm.
    bounds : list
        List to define parameter space.
    options : dict
        Stopping criteria for optimization.
    success : bool
        If True, optimization was successful.
    betas : np.ndarray
        The estimated coefficients.
    params : np.ndarray
        The estimated GAS parameters.
        
        
    Methods
    -------
    fit()
        Fit score-driven model, according to the specified method (’gaussian’ or ’student’)
    plot()
        Plot estimated coefficients against true data
    

    """

    def __init__(self, vY: np.ndarray, mX: np.ndarray, method: str = 'none', vgamma0: np.ndarray = None, bounds: list = None, options: dict = None, maxiter: int = 5):
        self.vY = vY.flatten()
        self.mX = mX
        self.n = len(vY)
        self.n_est = np.shape(mX)[1]
        self.method = method.lower()
        self.vgamma0 = vgamma0
        self.maxiter = maxiter
        if self.vgamma0 is not None:
            if self.method == 'gaussian' and len(self.vgamma0) == 3*self.n_est+1:
                ValueError(
                    "Incorrect number of initial parameters are provided. Provide either 3*n_est + 1 or no initial parameters.")
            if self.method == 'student' and len(self.vgamma0) == 3*self.n_est + 2:
                ValueError(
                    "Incorrect number of initial parameters are provided. Provide either 3*n_est + 2 or no initial parameters.")

        self.bounds = bounds
        self.options = {'maxfun': 5E3} if options is None else options
        self.success = None
        self.betas = None
        self.params = None

    def fit(self):
        '''
        Fit score-driven model, according to the specified method (’gaussian’ or ’student’)

        Returns
        -------
        mBetaHat : np.ndarray
            The estimated coefficients.
        vparaHat : np.ndarray
            The estimated GAS parameters.

        '''
        # np.random.seed(123)
        if self.method == 'none':
            print('Warning: no filter method is specified. A t-GAS filter is computed.')
            self.method = 'student'

        # set initial values
        dnInitial = int(np.ceil(self.n / 10))
        vbeta0 = np.linalg.inv((self.mX[:dnInitial, :]).T @ (self.mX[:dnInitial, :])) @ (
            (self.mX[:dnInitial, :]).T @ self.vY[:dnInitial])
        LB = np.concatenate(([0.001], -10 * np.ones(self.n_est), -
                            np.ones(self.n_est), -10 * np.ones(self.n_est)))
        UB = np.concatenate(([100], 10 * np.ones(self.n_est),
                            np.ones(self.n_est), 10 * np.ones(self.n_est)))

        start_time = time.time()

        if self.method == 'gaussian':  # MLE by Gaussian-GAS
            if self.bounds is None:
                self.bounds = list(zip(LB, UB))
            if self.vgamma0 is None:
                vdelta0 = 1
                vtheta0 = np.concatenate(
                    [0.1*np.ones(self.n_est),  0.1*np.ones(self.n_est), 0.1*np.ones(self.n_est)])
                self.vgamma0 = np.concatenate([[vdelta0], vtheta0])

            def fgGAS_lh(vpara): return - \
                self.construct_likelihood(vbeta0, vpara)
            mmBeta = np.zeros((self.maxiter, self.n_est, self.n))
            vMSE = np.zeros(self.maxiter)
            vSuccess = np.zeros(self.maxiter)
            mParaHat = np.zeros((self.maxiter, len(self.vgamma0)))
            for j in range(self.maxiter):
                result = minimize(fgGAS_lh, self.vgamma0 + np.random.normal(0, 0.2, len(self.vgamma0)),
                                  bounds=self.bounds, options=self.options)

                vparaHat_gGAS = result.x
                mParaHat[j,:] = vparaHat_gGAS
                vSuccess[j] = result.success

                # construct betat estimate
                vbetaNow = vbeta0
                mBetaHat_gGAS = np.zeros((self.n_est, self.n))
                vthetaHat_gGAS = vparaHat_gGAS[1:]
                vomegaHat_gGAS = vthetaHat_gGAS[:self.n_est]
                mBHat_gGAS = vthetaHat_gGAS[self.n_est:2*self.n_est]
                mAHat_gGAS = vthetaHat_gGAS[2*self.n_est:]

                for id in range(self.n):
                    vxt = self.mX[id, :].reshape(-1, 1)
                    yt = self.vY[id]
                    epst = yt - vbetaNow.T @ vxt

                    vMSE[j] = vMSE[j] + epst*epst/self.n

                    mNablat = vxt * epst
                    vbetaNow = vomegaHat_gGAS + mBHat_gGAS * \
                        vbetaNow + mAHat_gGAS * mNablat.squeeze()

                    mBetaHat_gGAS[:, id] = vbetaNow

                mmBeta[j, :, :] = mBetaHat_gGAS
                
            ind_opt = np.argmin(vMSE)
            mBetaHat = (mmBeta[ind_opt, :, :]).T
            vparaHat = mParaHat[ind_opt,:]
            self.success = vSuccess[ind_opt]
            # mBetaHat = mBetaHat_gGAS.T

        elif self.method == 'student':  # MLE by t-GAS
            LB = np.concatenate(([0.01], LB))
            UB = np.concatenate(([200], UB))
            if self.bounds is None:
                self.bounds = list(zip(LB, UB))

            if self.vgamma0 is None:
                vdelta0 = np.array([10, 1])
                vtheta0 = np.concatenate(
                    [np.ones(self.n_est), -0.1 * np.ones(self.n_est), 0.1 * np.ones(self.n_est)])
                self.vgamma0 = np.concatenate([vdelta0, vtheta0])

            def ftGAS_lh(vpara): return - \
                self.construct_likelihood(vbeta0, vpara)
            mmBeta = np.zeros((self.maxiter, self.n_est, self.n))
            vMSE = np.zeros(self.maxiter)
            vSuccess = np.zeros(self.maxiter)
            mParaHat = np.zeros((self.maxiter, len(self.vgamma0)))
            for j in range(self.maxiter):
                result = minimize(ftGAS_lh, self.vgamma0 + np.random.normal(0, 0.5, len(self.vgamma0)),
                                  bounds=self.bounds, options=self.options)

                vparaHat_tGAS = result.x
                mParaHat[j,:] = vparaHat_tGAS
                vSuccess[j] = result.success

                # construct betat estimate
                vbetaNow = vbeta0
                mBetaHat_tGAS = np.zeros((self.n_est, self.n))
                dnuHat_tGAS = vparaHat_tGAS[0]
                dsigmauHat_tGAS = vparaHat_tGAS[1]
                vthetaHat_tGAS = vparaHat_tGAS[2:]
                vomegaHat_tGAS = vthetaHat_tGAS[:self.n_est]
                mBHat_tGAS = vthetaHat_tGAS[self.n_est:2*self.n_est]
                mAHat_tGAS = vthetaHat_tGAS[2*self.n_est:]

                for id in range(self.n):
                    vxt = self.mX[id, :].reshape(-1, 1)
                    yt = self.vY[id]
                    epst = yt - vbetaNow.T @ vxt

                    vMSE[j] = vMSE[j] + epst*epst/self.n

                    temp1 = (1 + dnuHat_tGAS**(-1)) * (1 + dnuHat_tGAS**(-1)
                                                       * (epst / dsigmauHat_tGAS)**2)**(-1)
                    mNablat = (1 + dnuHat_tGAS)**(-1) * (3 + dnuHat_tGAS) * \
                        temp1 * vxt * epst
                    vbetaNow = vomegaHat_tGAS + mBHat_tGAS * \
                        vbetaNow + mAHat_tGAS * mNablat.squeeze()

                    mBetaHat_tGAS[:, id] = vbetaNow
                mmBeta[j, :, :] = mBetaHat_tGAS
                
            ind_opt = np.argmin(vMSE)
            mBetaHat = (mmBeta[ind_opt, :, :]).T
            vparaHat = mParaHat[ind_opt,:]
            self.success = vSuccess[ind_opt]
            # mBetaHat = mBetaHat_tGAS.T

        self.betas, self.params = mBetaHat, vparaHat
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return mBetaHat, vparaHat

    def construct_likelihood(self, vbeta0, vpara):
        '''
        Calculated the log-likelihood value, according to the specified self.method.

        Parameters
        ----------
        vbeta0 : np.ndarray
            Initial coefficients at time zero.
        vpara : np.ndarray
            Array of GAS parameters.

        Returns
        -------
        lhVal : float
            Log-likelihood value.

        '''
        # set initial values
        vbetaNow = vbeta0
        lhVal = 0

        if self.method == 'gaussian':
            dsigmau = vpara[0]
            vtheta = vpara[1:]
            vomega = vtheta[:self.n_est]
            mB = vtheta[self.n_est:2*self.n_est]
            mA = vtheta[2*self.n_est:]

            for id in range(self.n):
                vxt = self.mX[id, :].reshape(-1, 1)
                yt = self.vY[id]

                lhVal += ((yt - vbetaNow.T @ vxt) / dsigmau)**2

                mNablat = vxt * (yt - vbetaNow.T @ vxt)
                vbetaNow = vomega + mB * vbetaNow + mA * mNablat.squeeze()

            lhVal = -np.log(dsigmau) - 0.5 * lhVal / self.n

        elif self.method == 'student':
            dnu = vpara[0]
            dsigmau = vpara[1]
            vtheta = vpara[2:]
            vomega = vtheta[:self.n_est]
            mB = vtheta[self.n_est:2*self.n_est]
            mA = vtheta[2*self.n_est:]

            for id in range(self.n):
                vxt = self.mX[id, :].reshape(-1, 1)
                yt = self.vY[id]

                lhVal += np.log(1+dnu**(-1)*((yt - vbetaNow.T @ vxt) / dsigmau)**2)

                temp1 = (1 + dnu**(-1)) * (1 + dnu**(-1) *
                                           ((yt - vbetaNow.T @ vxt) / dsigmau)**2)**(-1)
                mNablat = (1 + dnu)**(-1) * (3 + dnu) * \
                    temp1 * vxt * (yt - vbetaNow.T @ vxt)
                vbetaNow = vomega + mB * vbetaNow + mA * mNablat.squeeze()

            lhVal = -0.5 * (dnu + 1) * lhVal / self.n + gammaln((dnu + 1) / 2) - \
                gammaln(dnu / 2) - 0.5 * np.log(np.pi * dnu) - np.log(dsigmau)

        return lhVal


    def plot(self):
        '''
        Plot the beta coefficients over a normalized x-axis from 0 to 1.

        '''
        
        x_vals = np.linspace(0, 1, self.n)
        
        if self.n_est == 1:
    
            plt.figure(figsize=(12, 6))
            plt.plot(x_vals, self.vY, label="True data", linewidth=2,color='black')
            if self.method=='student':
                plt.plot(x_vals, self.betas, label="Estimated $\\beta_{0}$ - tGAS", linestyle="--", linewidth=2)
            elif self.method=='gaussian':
                plt.plot(x_vals, self.betas, label="Estimated $\\beta_{0}$ - gGAS", linestyle="--", linewidth=2)
         
            plt.grid(linestyle='dashed')
            plt.xlabel('$t/n$',fontsize="xx-large")

            plt.tick_params(axis='both', labelsize=16)
            plt.legend(fontsize="x-large")
            plt.show()

        else:
            plt.figure(figsize=(10, 6 * self.n_est))
            for i in range(self.n_est):
                plt.subplot(self.n_est, 1, i + 1)
                if self.method=='student':
                    plt.plot(x_vals, self.betas[:, i],
                            label=f'Estimated $\\beta_{i} - tGAS$', color='black', linewidth=2)
                elif self.method=='gaussian':
                    plt.plot(x_vals, self.betas[:, i],
                            label=f'Estimated $\\beta_{i} - GGAS$', color='black', linewidth=2)
                
                plt.grid(linestyle='dashed')
                plt.xlabel('$t/n$',fontsize="xx-large")

                plt.tick_params(axis='both', labelsize=16)
                plt.legend(fontsize="x-large")
            plt.show()
