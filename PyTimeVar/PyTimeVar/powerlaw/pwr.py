import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.ar_model import AutoReg

class PowerLaw:
    """
    Class for implementing the Power-Law method.
    
    Parameters
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    n_powers : int
        The number of powers.
    vgamma0 : np.ndarray
        The initial parameter vector.
    options : dict
        Stopping criteria for optimization.
        
    Attributes
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    n : int
        The length of vY.
    p : int
        The number of powers. Default is set to 2.
    vgamma0 : np.ndarray
        The initial parameter vector.
    bounds : list
        List to define parameter space.
    cons : dict
        Dictionary that defines the constraints.
    trendHat : np.ndarray
        The estimated trend.
    gammaHat : np.ndarray
        The estimated power parameters.
    coeffHat : np.ndarray
        The estimated coefficients.
    C_LB_trend : np.ndarray
        The lower bounds of the pointwise confidence intervals for the trend.
    C_UB_trend ; np.ndarray
        The upper bound of the pointwise confidence intervals for the trend.
        
    Raises
    ------
    ValueError
        No valid bounds are provided.
        
    
    """

    def __init__(self, vY: np.ndarray, n_powers: float = None, vgamma0: np.ndarray=None, bounds : tuple=None, options: dict=None):
        self.vY = vY.reshape(-1,1)
        self.n = len(self.vY)
        self.p = 2 if n_powers is None else n_powers
        if n_powers is None:
            print('The number of powers is set to 2 by default. \nConsider setting n_powers to 3 or higher if a visual inspection of the data leads you to believe the trend is curly.\n')

        self.vgamma0 =vgamma0 if vgamma0 is not None else np.arange(0, 1*self.p, 1)
        self.bounds = bounds if bounds is not None else ((-0.495, 8),)*self.p
        for j in range(self.p):
            if self.bounds[j][0]<= -0.5:
                raise ValueError('Parameters are not identified if the power is smaller or equal than -1/2.\n The lower bounds need to be larger than -1/2.')
        self.options = options if options is not None else {'maxiter': 5E5}
        self.cons = {'type': 'ineq', 'fun': self._nonlcon}

        self.trendHat = None
        self.gammaHat = None
        self.coeffHat = None
        
        self.C_LB_trend = None
        self.C_UB_trend = None

    def plot(self, tau : list=None, confidence_intervals=True):
        """
        Plots the original series and the trend component.
        
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
        if self.trendHat is None:
            print("Model is not fitted yet.")
            return
        
        if confidence_intervals:
            if self.C_LB_trend == None:
                print("Confidence intervals not computed yet.")
                print('Run first .confidence_intervals() with desired settings.')
                print('Plot is shown without confidence intervals.')
        
        
        tau_index = np.array([None,None])
        x_vals = np.arange(1/self.n,(self.n+1)/self.n,1/self.n)
        if tau is None:

            tau_index=np.array([0,self.n])
        elif isinstance(tau, list):
            if tau[0] == tau[1]:
                raise ValueError("Invalid input: a and b cannot be equal.")

            if tau[0] > 1 and tau[1] > 1:
                raise ValueError("The values of tau must be in [0,1].")

            if tau[0] < 0 and tau[1] < 0:
                raise ValueError("The values of tau must be in [0,1].")


            if tau[0] < 0 or tau[1] > 1:
                print("Warning: The values of tau must be in [0,1].")

            original_tau = tau.copy()
            tau[0] = max(0, min(tau[0], 1))
            tau[1] = max(0, min(tau[1], 1))

            if original_tau != tau:
                print(f"Set to {tau} automatically.")
            if tau[0] > tau[1]:
                print("Warning: tau[0] > tau[1]. Values are switched automatically.")
                tau[0], tau[1] = tau[1], tau[0]

            tau_index[0] = int(tau[0]*(self.n-1))
            tau_index[1] = int(tau[1]*(self.n))
        else:
            raise ValueError('The optional parameter tau is required to be a list.')
        

        plt.figure(figsize=(12, 6))
        plt.plot(x_vals[tau_index[0]:tau_index[1]], self.vY[tau_index[0]:tau_index[1]], label="True data", linewidth=1, color = 'black')
        plt.plot(x_vals[tau_index[0]:tau_index[1]], self.trendHat[tau_index[0]:tau_index[1]], label="Estimated $\\beta_{0}$", linestyle="--", linewidth=2)
        
        plt.grid(linestyle='dashed')
        plt.xlabel('$t/n$',fontsize="xx-large")

        plt.tick_params(axis='both', labelsize=16)
        plt.legend(fontsize="x-large")
        plt.show()   
        
    def summary(self):
        """
        Print the mathematical equation for the fitted model

        """
        
        def term(coef, power):
            coef = coef if coef != 1 else ''
            coef, power = round(coef, 3), round(power, 3)
            if power >0:
                power = (f'^{power}') if power > 1 else ''
                return f'{coef} t{power}'
            else:
                return f'{coef}'
        terms = []
        for j in range(len(self.coeffHat)):
          if self.coeffHat[j][0] != 0:
            terms.append(term(self.coeffHat[j][0], self.gammaHat[0][j]))
        print('\nPower-Law Trend Results:')
        print('='*30)
        print('yhat= ' + ' + '.join(terms))

    def fit(self):
        '''
        Fits the Power-Law model to the data.        

        Returns
        -------
        self.trendHat : np.ndarray
            The estimated trend.
        self.gammaHat : np.ndarray
            The estimated power parameters.

        '''
        res = minimize(self._construct_pwrlaw_ssr, self.vgamma0,
                       bounds=self.bounds, constraints=self.cons, options=self.options, args=(self.vY,))
        self.gammaHat = res.x.reshape(1, self.p)

        trend = np.arange(1, self.n+1, 1).reshape(self.n, 1)
        mP = trend ** self.gammaHat
        self.coeffHat = np.linalg.pinv(mP.T @ mP) @ mP.T @ self.vY
        self.trendHat = mP @ self.coeffHat

        return self.trendHat, self.gammaHat

    def _construct_pwrlaw_ssr(self, vparams, vY):
        '''
        Compute sum of squared residuals for a given parameter vector.

        Parameters
        ----------
        vparams : np.ndarray
            The parameter vector.

        Returns
        -------
        ssr : float
            Sum of squared residuals.

        '''
        trend = np.arange(1, self.n+1, 1).reshape(self.n, 1)

        vparams = np.array(vparams).reshape(1, self.p)
        mP = trend ** vparams
        coeff = np.linalg.pinv(mP.T @ mP) @ mP.T @ vY
        ssr = np.sum((vY - mP @ coeff)**2)
        return ssr

    def _nonlcon(self, params):
        '''
        Construct the nonlinear constraints for identification.

        Parameters
        ----------
        params : np.ndarray
            The parameter vector.

        Returns
        -------
        c : list
            List of non-linear parameter constraints.

        '''
        epsilon = 0.005
        c = []
        for id1 in range(self.p-1):
            for id2 in range(id1+1, self.p):
                c.append(params[id1] - params[id2] + epsilon)
        return c

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
        beta: np.ndarray,
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
        beta : np.ndarray
            Array of estimated coefficients.
        T : int
            Total number of observations.

        Returns
        -------
        vYstar : np.ndarray
            The bootstrap sample generated by the _S_BT algorithm.
        """
        epsilonstar = epsilontilde[np.random.choice(len(epsilontilde), T - max_lag + 50),:]
        if max_lag == 0:
            zstar = epsilonstar[50:]
            zstar_array = zstar
        else:
            max_lag = np.arange(1, max_lag + 1)
            zstar_array = self._get_Zstar_AR(max_lag, armodel, T, epsilonstar)

        vYstar = mX @ beta + zstar_array
        return vYstar

    ######### Autoregressive Bootstrap #########
    def _AW_BT(
        self,
        zhat: np.ndarray,
        mX: np.ndarray,
        beta: np.ndarray,
        T: int,
        gamma: float,
        C: float
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
        beta : np.ndarray
            Array of estimated coefficients.
        T : int
            Total number of observations.
        gamma : float
            The AR(1) coefficient and the standard deviation of the wild component in the AWB.
        C : float
            The constant to determine the window length for blocks bootstraps. 
            Default is 2.


        Returns
        -------
        vYstar : np.ndarray
            The bootstrap sample generated by the _AW_BT algorithm.
        """
        # v_nv_star=np.random.normal(0,np.sqrt(1-gamma**2),200)
        xi_star0 = np.random.normal(0, 1, 1)

        v_xi_star = np.zeros((T,1))
        v_xi_star[0] = xi_star0
        for i in range(1, T):
            v_xi_star[i] = gamma * v_xi_star[i - 1] + \
                np.random.normal(0, np.sqrt(1 - gamma ** 2))

        zstar = v_xi_star * zhat      

        vYstar = mX @ beta + zstar

        return vYstar

    def _W_BT(
        self,
        zhat: np.ndarray,
        mX: np.ndarray,
        beta: np.ndarray,
        T: int,
        gamma: float,
        C: float
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
        beta : np.ndarray
            Array of estimated coefficients.
        T : int
            Total number of observations.
        gamma : float
            The AR(1) coefficient and the standard deviation of the wild component in the AWB.    
        C : float
            The constant to determine the window length for blocks bootstraps. 
            Default is 2.


        Returns
        -------
        vYstar : np.ndarray
            The bootstrap sample generated by the _W_BT algorithm.
        """
        # generate zstar by
        zstar = zhat * np.random.normal(0, 1, size=(T,1))

        vYstar = mX @ beta + zstar
        return vYstar

    def _SW_BT(
        self,
        epsilontilde: np.ndarray,
        max_lag: int,
        armodel,
        mX: np.ndarray,
        beta: np.ndarray,
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
        beta : np.ndarray
            Array of estimated coefficients.
        T : int
            Total number of observations.

        Returns
        -------
        vYstar : np.ndarray
            The bootstrap sample generated by the _SW_BT algorithm.
        """
        epsilonstar = epsilontilde * np.random.normal(0, 1, (T - max_lag,1))
        epsilonstar = epsilonstar[np.random.choice(len(epsilonstar), T - max_lag + 50),:]
        if max_lag == 0:
            zstar = epsilonstar[50:]
            zstar_array = zstar
        else:
            max_lag = np.arange(1, max_lag + 1)
            zstar_array = self._get_Zstar_AR(max_lag, armodel, T, epsilonstar)

        vYstar = mX @ beta + zstar_array
        return vYstar

    def _LBW_BT(
        self, zhat: np.ndarray, mX: np.ndarray, beta: np.ndarray, T: int, gamma: float, C: float
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
        beta : np.ndarray
            The input array of shape (k,) or (k, 1) containing the estimated coefficients.
        T : int
            The number of observations.
        gamma : float
            The AR(1) coefficient and the standard deviation of the wild component in the AWB.
        C : float
            The constant to determine the window length for blocks bootstraps. 
            Default is 2.

        Returns
        -------
        vYstar : np.ndarray
            The bootstrap sample generated by the _LBW_BT algorithm.
        """
        l = C*int(T**(1/4))
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
        vYstar = mX @ beta + zstar
        return vYstar

    
    def _aux_fit(self, vYstar):
        '''
        Fits the Power-Law model to the auxiliary data.        

        Returns
        -------
        trendHat : np.ndarray
            The estimated trend.
        gammaHat : np.ndarray
            The estimated power parameters.

        '''
        res = minimize(self._construct_pwrlaw_ssr, self.vgamma0,
                       bounds=self.bounds, constraints=self.cons, options=self.options, args=(vYstar,))
        gammaHat = res.x.reshape(1, self.p)

        trend = np.arange(1, self.n+1, 1).reshape(self.n, 1)
        mP = trend ** gammaHat
        coeffHat = np.linalg.pinv(mP.T @ mP) @ mP.T @ vYstar
        trendHat = mP @ coeffHat

        return coeffHat, gammaHat, trendHat
    
    def confidence_intervals(self, bootstraptype: str, alpha: float = None, gamma: float = None, ic: str = None, B: float = 1299, block_constant: float = 2, verbose: bool = True):
        """
        Construct confidence intervals using bootstrap methods.

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
        B : int
            The number of bootstrap samples.
            Deafult is 1299, if not provided by the user.
        C : float
            The constant to determine the window length for blocks bootstraps. 
            Default is 2.
            
        Raises
        ------
        ValueError
            No valid bootstrap type is provided.

        Returns
        -------
        list of tuples
            Each tuple contains pointwise lower and upper bands.
        """
        if alpha is None or alpha <= 0 or alpha >= 1:
            alpha = 0.05


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
                "Invalid bootstrap type. Choose one of 'SB','WB', 'SWB' ,'LBWB', 'AWB'")

        bootstrap_function = bootstrap_functions[bootstraptype]

        if gamma is None or gamma <= 0 or gamma >= 1:
            l = block_constant*int(self.n**(1/4))
            gamma = (0.01)**(1/l)

        trend = np.arange(1, self.n+1, 1).reshape(self.n, 1)
        mP = trend ** self.gammaHat
        zhat = (self.vY - self.trendHat).reshape(-1,1)

        # Initialize storage for bootstrap samples
        mCoeffStar = np.zeros((B, self.p))
        mGammaStar = np.zeros((B, self.p))
        mTrendStar = np.zeros((B, self.n))

        print(f"Calculating {bootstraptype} Bootstrap Samples")
        for i in tqdm(range(B)):
            if bootstraptype in ["SB", "SWB"]:
                epsilonhat, max_lag, armodel = self._AR(zhat, self.n, ic)
                epsilontilde = epsilonhat - np.mean(epsilonhat)
                if bootstraptype == 'SWB':
                    vYstar = bootstrap_function(
                        epsilonhat, max_lag, armodel, mP, self.coeffHat, self.n)
                elif bootstraptype == 'SB':
                    vYstar = bootstrap_function(
                        epsilontilde, max_lag, armodel, mP, self.coeffHat, self.n)
            else:
                vYstar = bootstrap_function(
                    zhat, mP, self.coeffHat, self.n, gamma, block_constant)
                
            coeff_star, gamma_star, trend_star = self._aux_fit(vYstar.reshape(-1,1))
            mCoeffStar[i] = coeff_star.squeeze()
            mGammaStar[i] = gamma_star.squeeze()
            mTrendStar[i,:] = trend_star.squeeze()

        C_LB_coeff = np.quantile(mCoeffStar, alpha/2, axis=0)
        C_UB_coeff = np.quantile(mCoeffStar, 1-alpha/2, axis=0)
        C_LB_gamma = np.quantile(mGammaStar, alpha/2, axis=0)
        C_UB_gamma = np.quantile(mGammaStar, 1-alpha/2, axis=0)
        C_LB_trend = np.percentile(mTrendStar, 100*(alpha/2), axis=0)
        C_UB_trend = np.percentile(mTrendStar, 100*(1-alpha/2), axis=0)

        if verbose:
            print('\n--- Power Law: Estimated Parameters and Confidence Intervals ---')
    
            print('Coefficients (τ_i):')
            for i in range(self.p):
                print(f'  τ_{i+1}:')
                print(f'    Estimated: {self.coeffHat[i][0]:.3f}')
                print(f'    Confidence Interval: {C_LB_coeff[i]:.3f} to {C_UB_coeff[i]:.3f}')
                    
            print('=============================================================')
            
            print('Power Parameters (γ_i):')
            for i in range(self.p):
                print(f'  γ_{i+1}:')
                print(f'    Estimated: {self.gammaHat[0][i]:.3f}')
                print(f'    Confidence Interval: {C_LB_gamma[i]:.3f} to {C_UB_gamma[i]:.3f}')
            
            print('=============================================================')
            
        self.C_LB_trend = C_LB_trend
        self.C_UB_trend = C_UB_trend
        return C_LB_coeff, C_UB_coeff, C_LB_gamma, C_UB_gamma, C_LB_trend, C_UB_trend
