import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.optimize import basinhopping
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
        Method to estimate GAS model. Choose between 'gaussian' or 'student'.
    vgamma0 : np.ndarray 
        Initial parameter vector.
    bounds : list
        List to define parameter space.
    options : dict
        Stopping criteria for optimization.
    niter : int
        The number of basin-hopping iterations, for scipy.optimize.basinhopping()
        
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
    bounds : list
        List to define parameter space.
    options : dict
        Stopping criteria for optimization.
    niter : int
        The number of basin-hopping iterations, for scipy.optimize.basinhopping()
    success : bool
        If True, optimization was successful.
    betas : np.ndarray
        The estimated coefficients.
    params : np.ndarray
        The estimated GAS parameters.
    inv_hessian : np.ndarray
        The inverse Hessian after optimization.
        
    
    Raises
    ------
    ValueError
        No valid number of initial parameters is provided.
    

    """

    def __init__(self, vY: np.ndarray, mX: np.ndarray, method: str = 'none', vgamma0: np.ndarray = None, bounds: list = None, options: dict = None, niter : int=10):
        self.vY = vY.flatten()
        self.mX = mX
        self.n = len(vY)
        self.n_est = np.shape(mX)[1]
        self.method = method.lower()
        self.vgamma0 = vgamma0
        if self.vgamma0 is not None:
            if self.method == 'gaussian' and len(self.vgamma0) == 3*self.n_est+1:
                raise ValueError(
                    "Incorrect number of initial parameters are provided. Provide either 3*n_est + 1 or no initial parameters.")
            if self.method == 'student' and len(self.vgamma0) == 3*self.n_est + 2:
                raise ValueError(
                    "Incorrect number of initial parameters are provided. Provide either 3*n_est + 2 or no initial parameters.")

        self.bounds = bounds
        self.options = {'maxfun': 5E5} if options is None else options
        self.niter = niter
        
        self.success = None
        self.betas = None
        self.params = None
        
        self.inv_hessian = None

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
                self._construct_likelihood(vbeta0, vpara)
                
            min_kwargs = {"method": "L-BFGS-B", "bounds": self.bounds, "options": self.options}
            result_bh = basinhopping(fgGAS_lh, self.vgamma0,
                                        minimizer_kwargs=min_kwargs,
                                        niter=self.niter)
            
            vparaHat_gGAS = result_bh.x
            self.success = result_bh.success
            
            # Run the local minimizer again at the best point found
            local_result = minimize(fgGAS_lh, vparaHat_gGAS, **min_kwargs)
            self.inv_hessian = local_result.hess_inv.todense()
            vparaHat_gGAS = local_result.x
            
            # construct betat estimate
            mBetaHat = self._g_filter(vbeta0, vparaHat_gGAS)
            vparaHat = vparaHat_gGAS
                
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
                self._construct_likelihood(vbeta0, vpara)
                
            min_kwargs = {"method": "L-BFGS-B", "bounds": self.bounds, "options": self.options}
            result_bh = basinhopping(ftGAS_lh, self.vgamma0,
                                        minimizer_kwargs=min_kwargs,
                                        niter=self.niter)
            
            vparaHat_tGAS = result_bh.x
            self.success = result_bh.success
            
            # Run the local minimizer again at the best point found
            local_result = minimize(ftGAS_lh, vparaHat_tGAS, **min_kwargs)
            self.inv_hessian = local_result.hess_inv.todense()
            vparaHat_tGAS = local_result.x

            # construct betat estimate
            mBetaHat = self._t_filter(vbeta0, vparaHat_tGAS)
            vparaHat = vparaHat_tGAS

        self.betas, self.params = mBetaHat, vparaHat
        print(f"\nTime taken: {time.time() - start_time:.2f} seconds")
        return mBetaHat, vparaHat
    
    def _g_filter(self, vbeta0, vparams):
        '''
        Run Gaussian score-driven filter.

        Parameters
        ----------
        vbeta0 : np.ndarray
            The initial filter estimates.
        vparams : np.ndarray
            The parameter values that specify the filter recursion.

        Returns
        -------
        mBetaHat : np.ndarray
            The estimated coefficients.

        '''
        
        vbetaNow = vbeta0
        mBetaHat_gGAS = np.zeros((self.n_est, self.n))
        vthetaHat_gGAS = vparams[1:]
        vomegaHat_gGAS = vthetaHat_gGAS[:self.n_est]
        mBHat_gGAS = vthetaHat_gGAS[self.n_est:2*self.n_est]
        mAHat_gGAS = vthetaHat_gGAS[2*self.n_est:]

        for id in range(self.n):
            vxt = self.mX[id, :].reshape(-1, 1)
            yt = self.vY[id]
            epst = yt - vbetaNow.T @ vxt

            mNablat = vxt * epst
            vbetaNow = vomegaHat_gGAS + mBHat_gGAS * \
                vbetaNow + mAHat_gGAS * mNablat.squeeze()

            mBetaHat_gGAS[:, id] = vbetaNow

        mBetaHat = mBetaHat_gGAS.T
        return mBetaHat
    
    def _t_filter(self, vbeta0, vparams):
        '''
        Run Student-t score-driven filter.

        Parameters
        ----------
        vbeta0 : np.ndarray
            The initial filter estimates.
        vparams : np.ndarray
            The parameter values that specify the filter recursion.

        Returns
        -------
        mBetaHat : np.ndarray
            The estimated coefficients.

        '''
        
        vbetaNow = vbeta0
        mBetaHat_tGAS = np.zeros((self.n_est, self.n))
        dnuHat_tGAS = vparams[0]
        dsigmauHat_tGAS = vparams[1]
        vthetaHat_tGAS = vparams[2:]
        vomegaHat_tGAS = vthetaHat_tGAS[:self.n_est]
        mBHat_tGAS = vthetaHat_tGAS[self.n_est:2*self.n_est]
        mAHat_tGAS = vthetaHat_tGAS[2*self.n_est:]

        for id in range(self.n):
            vxt = self.mX[id, :].reshape(-1, 1)
            yt = self.vY[id]
            epst = yt - vbetaNow.T @ vxt

            temp1 = (1 + dnuHat_tGAS**(-1)) * (1 + dnuHat_tGAS**(-1)
                                               * (epst / dsigmauHat_tGAS)**2)**(-1)
            mNablat = (1 + dnuHat_tGAS)**(-1) * (3 + dnuHat_tGAS) * \
                temp1 * vxt * epst
            vbetaNow = vomegaHat_tGAS + mBHat_tGAS * \
                vbetaNow + mAHat_tGAS * mNablat.squeeze()

            mBetaHat_tGAS[:, id] = vbetaNow
        
        mBetaHat = mBetaHat_tGAS.T
        return mBetaHat
        

    def _construct_likelihood(self, vbeta0, vpara):
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
    
    def _confidence_bands(self, alpha, iM):
        '''
        Compute confidence intervals at each time points by simulation-based methods.

        Parameters
        ----------
        alpha : float
            Significance level for quantiles.
        iM : int
            The nunber of simulations for simulation-based confidence intervals.

        Returns
        -------
        mCI_l : np.ndarray
            The lower confidence bounds at each time point, for each parameter.
        mCI_u : np.ndarray
            The upper confidence bounds at each time point, for each parameter.

        '''
        
        
        # draw iM parameter values
        LB = np.concatenate(([0.001], -10 * np.ones(self.n_est), -
                            np.ones(self.n_est), -10 * np.ones(self.n_est)))
        UB = np.concatenate(([100], 10 * np.ones(self.n_est),
                            np.ones(self.n_est), 10 * np.ones(self.n_est)))
        if self.method == 'student':
            LB = np.concatenate(([0.01], LB))
            UB = np.concatenate(([200], UB))
            
        
        mDraws = np.zeros((iM, len(self.params)))
        count = 0
        while count < iM:
            mSamples = np.random.multivariate_normal(self.params, (-1/self.n)*self.inv_hessian, size=iM)
            
            mMask = np.all((LB <= mSamples) & (mSamples <= UB), axis=1)
            mValid = mSamples[mMask]
            
            iNum_valid = mValid.shape[0]
            iNum_to_fill = min(iM-count, iNum_valid)
            
            mDraws[count:count+iNum_to_fill] = mValid[:iNum_to_fill]
            count += iNum_to_fill
        
        # obtain filter for each simulation
        dnInitial = int(np.ceil(self.n / 10))
        vbeta0 = np.linalg.inv((self.mX[:dnInitial, :]).T @ (self.mX[:dnInitial, :])) @ (
            (self.mX[:dnInitial, :]).T @ self.vY[:dnInitial])
        mBetaDraws = np.zeros((iM, self.n, self.n_est))
        filt = self._g_filter if self.method == 'gaussian' else self._t_filter
        for m in range(iM):
            mBetaDraws[m,:,:] = filt(vbeta0, mDraws[m,:]) 
        
        # get confidence intervals
        mCI_l = np.percentile(mBetaDraws, 100*(alpha/2), axis=0)
        mCI_u = np.percentile(mBetaDraws, 100*(1-alpha/2), axis=0)
        
        return mCI_l, mCI_u
        


    def plot(self, tau: list = None, confidence_intervals: bool = False, alpha = 0.05, iM = 1000):
        '''
        Plot the beta coefficients over a normalized x-axis from 0 to 1.
        
        Parameters
        ----------
        tau : list, optional
            The list looks the  following: tau = [start,end].
            The function will plot all data and estimates between start and end.
        confidence_intervals : bool, optional
            If True, simulation-based confidence intervals will be plotted around the estimates.
        alpha : float
            Significance level for confidence intervals.
        iM : int
            The nunber of simulations for simulation-based confidence intervals.
            
        Raises
        ------
        ValueError
            No valid tau is provided.

        '''
        
        tau_index=None
        x_vals = np.arange(1/self.n,(self.n+1)/self.n,1/self.n)
        if tau is None:

            tau_index=np.array([0,self.n])
        elif isinstance(tau, list):
            if min(tau) <= 0:
                tau_index = np.array([int(0), int(max(tau) * self.n)])
            else:
                tau_index = np.array([int(min(tau)*self.n-1),int(max(tau)*self.n)])
        else:
            raise ValueError('The optional parameter tau is required to be a list.')
        
        if confidence_intervals:
            mCI_l, mCI_u = self._confidence_bands(alpha, iM)
        
        if self.n_est == 1:
    
            plt.figure(figsize=(12, 6))
            plt.plot(x_vals[tau_index[0]:tau_index[1]], self.vY[tau_index[0]:tau_index[1]], label="True data", linewidth=1,color='black')
            if self.method=='student':
                plt.plot(x_vals[tau_index[0]:tau_index[1]], self.betas[tau_index[0]:tau_index[1]], label="Estimated $\\beta_{0}$ - tGAS", linestyle="--", linewidth=2)
            elif self.method=='gaussian':
                plt.plot(x_vals[tau_index[0]:tau_index[1]], self.betas[tau_index[0]:tau_index[1]], label="Estimated $\\beta_{0}$ - gGAS", linestyle="--", linewidth=2)
         
            if confidence_intervals:
                plt.plot(x_vals[tau_index[0]:tau_index[1]], mCI_u[tau_index[0]:tau_index[1],0], label=f'{1-alpha}% confidence interval', color='blue', linewidth=2, linestyle='dashed')
                plt.plot(x_vals[tau_index[0]:tau_index[1]], mCI_l[tau_index[0]:tau_index[1],0], color='blue', linewidth=2, linestyle='dashed')
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
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], (self.betas[:, i])[tau_index[0]:tau_index[1]],
                            label=f'Estimated $\\beta_{i} - tGAS$', color='black', linewidth=2)
                elif self.method=='gaussian':
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], (self.betas[:, i])[tau_index[0]:tau_index[1]],
                            label=f'Estimated $\\beta_{i} - GGAS$', color='black', linewidth=2)
                if confidence_intervals:
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], mCI_u[tau_index[0]:tau_index[1],i], label=f'{1-alpha}% confidence interval', color='blue', linewidth=2, linestyle='dashed')
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], mCI_l[tau_index[0]:tau_index[1],i], color='blue', linewidth=2, linestyle='dashed')
                plt.grid(linestyle='dashed')
                plt.xlabel('$t/n$',fontsize="xx-large")

                plt.tick_params(axis='both', labelsize=16)
                plt.legend(fontsize="x-large")
            plt.show()
