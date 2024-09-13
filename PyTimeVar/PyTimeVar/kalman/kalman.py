import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Kalman:
    """
    Class for performing Kalman filtering and smoothing.

    Parameters
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    T : np.ndarray, optional
        The transition matrix of the state space model.
    R : np.ndarray, optional
        The transition correlation matrix of the state space model.
    Q : np.ndarray, optional
        The transition covariance matrix of the state space model.
    sigma_u : np.ndarray, optional
        The observation noise variance of the state space model.
    b_1 : np.ndarray, optional
        The initial mean of the state space model.
    P_1 : np.ndarray, optional
        The initial covariance matrix of the state space model.
    mX : np.ndarray, optional
        The regressors to use in the model. If provided, the model will be a linear regression model.
        
        
    Attributes
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    n : int
        The length of vY.
    isReg : bool
        If True, regressors are provided by the user.
    T : np.ndarray
        The transition matrix of the state space model.
    Z : np.ndarray
        Auxiliary (1,1)-vector of a scalar 1. This is used in case there are no regressors.
    R : np.ndarray
        The transition correlation matrix of the state space model.
    Q : np.ndarray
        The transition covariance matrix of the state space model.
    H : np.ndarray
        The observation noise variance of the state space model.
    a_1 : np.ndarray, optional
        The initial mean of the state space model.
    P_1 : np.ndarray
        The initial covariance matrix of the state space model.
    mX : np.ndarray
        The regressors to use in the model. If provided, the model will be a linear regression model.
    Z_reg : np.ndarray
        The regressors matrix in correct format to use in filtering.
    p_dim : int
        The number of coefficients.
    m_dim : int
        The number of response variables. This is always 1.
    filt : np.ndarray
        The filtered coefficients.
    smooth : 
        The smoothed coefficients.
    
    
    Methods
    -------
    fit()
        Fit state-space model by Kalman filter or smoother, according to the specified option
        (’filter' or ’smoother’)
    summary()
        Print a summary of the Kalman filter/smoother specifications, including the values of H, Q, R and T
    plot()
        Plot filtered/smoothed estimates against true data
    
    
    
    """

    def __init__(self, vY: np.ndarray = None, T: np.ndarray = None, R: np.ndarray = None, Q: np.ndarray = None, sigma_u: float = None, b_1: np.ndarray = None, P_1: np.ndarray = None, mX: np.ndarray = None):
        self.vY = vY.reshape(-1,1)
        self.n = len(self.vY)
        self.isReg = False
        self.T = T
        if mX is not None:
            self.isReg = True
            if mX.ndim == 1:
                k = 1
                self.mX = mX.reshape(-1, 1)
            else:
                k = mX.shape[1]
                self.mX = mX

            self.T = T if T is not None else np.eye(k)

            # Create a Z array that can be indexed by t to return the appropriate regressor
            self.Z_reg = np.array([self.mX[t]
                                  for t in range(self.mX.shape[0])])
        self.T = self.T if self.T is not None else np.array(
            [[1]])                # Transition matrix
        self.Z = np.array(
            [[1]])                # Observation vector
        self.Q = Q
        self.bEst_Q, self.bEst_H = False, False
        if self.Q is None:
            # Transition covariance matrix
            self.bEst_Q = True
        self.H = sigma_u
        if self.H is None:
            # Observation covariance matrix
            self.bEst_H = True
        else: 
            self.H = np.array([sigma_u])
        self.a_1 = b_1 if b_1 is not None else np.zeros(
            self.T.shape[0])            # Initial state mean
        self.P_1 = P_1 if P_1 is not None else np.eye(
            self.T.shape[0])*1000       # Initial state covariance
        self.R = R if R is not None else np.eye(
            self.a_1.shape[0])         # Transition covariance matrix
        # Dimension of the state space
        self.p_dim = self.T.shape[0]
        # Dimension of the observation space
        self.m_dim = self.Z.shape[0]

        self.H, self.Q = self._estimate_ML()

        self.filt = None
        self.smooth = None

    def _estimate_ML(self):
        '''
        Esimates the H and Q values, if necessary.

        Returns
        -------
        np.ndarray
            Estimated value of H.
        np.ndarray
            Estimated value of Q.

        '''
        if self.bEst_H and self.bEst_Q:
            vH_init, vQ_init = np.ones(
                self.m_dim**2)*100, np.ones(self.p_dim**2)*100
            vTheta = np.hstack([vH_init, vQ_init])
        elif self.bEst_H and not self.bEst_Q:
            vTheta = np.ones(self.m_dim**2)*100
            mQ = self.Q
        elif not self.bEst_H and self.bEst_Q:
            vTheta = np.ones(self.p_dim**2)*100
            mH = self.H
        else:
            return self.H, self.Q

        LL_model = minimize(self.compute_likelihood_LL, vTheta, method='SLSQP', bounds=(
            (0.00001, None),)*len(vTheta), options={'maxiter': 3000})
        if LL_model.success == False:
            print('Optimization failed')
        else:
            print('Optimization success')

        if self.bEst_H and self.bEst_Q:
            mH, mQ = LL_model.x[:self.m_dim**2].reshape(
                self.m_dim, self.m_dim), LL_model.x[self.m_dim**2:].reshape(self.p_dim, self.p_dim)
        elif self.bEst_H and not self.bEst_Q:
            mH = LL_model.x.reshape(self.m_dim, self.m_dim)
        elif not self.bEst_H and self.bEst_Q:
            mQ = LL_model.x.reshape(self.p_dim, self.p_dim)

        return np.array([[mH]]), np.array([[mQ]])

    def compute_likelihood_LL(self, vTheta):
        '''
        Computes the negative log-likelihood value for a given parameter vector.

        Parameters
        ----------
        vTheta : np.ndarray
            The parameter vector.

        Returns
        -------
        float
            The negative log-likelihood value.

        '''
        if self.bEst_H and self.bEst_Q:
            self.H, self.Q = vTheta[:self.m_dim**2].reshape(
                self.m_dim, self.m_dim), vTheta[self.m_dim**2:].reshape(self.p_dim, self.p_dim)
        elif self.bEst_H and not self.bEst_Q:
            self.H = vTheta.reshape(self.m_dim, self.m_dim)
        elif not self.bEst_H and self.bEst_Q:
            self.Q = vTheta.reshape(self.p_dim, self.p_dim)

        a, P, v, F, K = self._filter()
        dLL = -(self.n*self.m_dim/2)*np.log(2*np.pi)
        for t in range(self.n):
            dLL = dLL - 0.5 * \
                (np.linalg.det(F[t, :, :]) + v[t, :, :].T @
                 np.linalg.inv(F[t, :, :])@v[t, :, :])

        return -dLL

    def _filter(self):
        """
        Performs the filtering step of the Kalman filter.

        Returns
        -------
        np.ndarray
            The filtered state at each time step.
        np.ndarray
            The filtered state covariances at each time step.
        np.ndarray
            The prediction errors at each time step.
        np.ndarray
            The prediction variances at each time step.
        np.ndarray
            The Kalman gains at each time step.

        """

        a = np.zeros((self.n + 1, self.p_dim, 1))
        P = np.zeros((self.n + 1, self.p_dim, self.p_dim))
        v = np.zeros((self.n, self.m_dim, 1))
        F = np.zeros((self.n, self.m_dim, self.m_dim))
        K = np.zeros((self.n, self.p_dim, self.m_dim))
        a[0] = self.a_1.reshape(self.T.shape[1], 1)
        P[0] = self.P_1
        for t in range(self.n):
            if self.isReg:
                self.Z = self.Z_reg[t].reshape(1, -1)
            v[t] = self.vY[t] - self.Z @ a[t]
            F[t] = self.Z @ P[t] @ self.Z.T + self.H

            if not np.isnan(self.vY[t]):
                K[t] = self.T @ P[t] @ self.Z.T @ np.linalg.inv(F[t])
                a[t + 1] = self.T @ a[t] + K[t] @ v[t]
            else:
                K[t] = 0
                a[t + 1] = self.T @ a[t]

            P[t + 1] = self.T @ P[t] @ (self.T - K[t]
                                        @ self.Z).T + self.R @ self.Q @ self.R.T
        return a[1:], P[1:], v, F, K

    def filter(self):
        """
        Performs the filtering step of the Kalman filter.

        Returns
        -------
        np.ndarray
            The filtered state means at each time step.
        """
        a, _, _, _, _ = self._filter()

        return a.squeeze()

    def _smoother(self):
        """
        Performs the smoothing steps of the Kalman filter.

        Returns
        -------
        a_s : np.ndarray
            The smoothed state means at each time step.
        V_s : np.ndarray
            The smoothed state covariances at each time step.
        """

        a, P, v, F, K = self._filter()
        a_s = np.zeros((self.n, self.p_dim, 1))
        V_s = np.zeros((self.n, self.p_dim, self.p_dim))

        r_prev, r_cur = np.zeros((self.p_dim, 1)), np.zeros((self.p_dim, 1))
        N_prev, N_cur = np.zeros((self.p_dim, self.p_dim)), np.zeros((self.p_dim, self.p_dim))

        for t in range(self.n-1, -1, -1):
            if self.isReg:
                self.Z = self.Z_reg[t].reshape(1, -1)
            L = self.T - K[t] @ self.Z
            if not np.isnan(self.vY[t]):
                r_prev = self.Z.T @ np.linalg.inv(F[t]) @ v[t] + L.T @ r_cur
                N_prev = self.Z.T @ np.linalg.inv(F[t]) @ self.Z + L.T @ N_cur @ L
            else:
                r_prev = self.T.T @ r_cur
                N_prev = self.T.T @ N_cur @ self.T
            
            a_s[t] = a[t] + P[t] @ r_prev
            V_s[t] = P[t] - P[t] @ N_prev @ P[t]
            
            r_cur, N_cur = r_prev, N_prev

        return a_s, V_s

    def smoother(self):
        """
        Performs the smoothing step of the Kalman filter.

        Returns
        -------
        np.ndarray
            The smoothed state means at each time step.
        """
        a_s, _ = self._smoother()
        return a_s.squeeze()

    def fit(self, option):
        '''
        Fits the Kalman filter or smoother to the data.

        Parameters
        ----------
        option : string
            Denotes the fitted trend: filter or smoother.

        Raises
        ------
        ValueError
            No valid option is provided.

        Returns
        -------
        np.ndarray
            Estimated trend.

        '''

        if option.lower() == 'filter':
            self.filt = self.filter()
            return self.filt
        elif option.lower() == 'smoother':
            self.smooth = self.smoother()
            return self.smooth
        else:
            raise ValueError(
                'Unknown option provided to fit(). Choose either filter or smoother')

    def summary(self):
        """
        Prints a summary of the state-space model specification.
        """

        print("Kalman Filter specification:")
        print(f"H: {self.H}")
        print(f"Q: {self.Q}")
        print(f"R: {self.R}")
        print(f"T: {self.T}")


    def plot(self):
        """
        Plot the beta coefficients over a normalized x-axis from 0 to 1 or over a date range.
        """

        x_vals = np.linspace(0, 1, self.n)
        
        if self.p_dim == 1:
            plt.figure(figsize=(12, 6))
            plt.plot(x_vals, self.vY, label="True data", linewidth=2,color='black')
            if self.smooth is not None:
                plt.plot(x_vals, self.smooth[:], label="Estimated $\\beta_{0}$ - Smoother", linestyle="--", linewidth=2)
            if self.filt is not None:
                plt.plot(x_vals[1:], self.filt[:-1], label="Estimated $\\beta_{0}$ - Filter", linestyle="-", linewidth=2)
            
            plt.grid(linestyle='dashed')
            plt.xlabel('$t/n$',fontsize="xx-large")

            plt.tick_params(axis='both', labelsize=16)
            plt.legend(fontsize="x-large")
            plt.show()
            
        else:
            plt.figure(figsize=(10, 6 * self.p_dim))
            for i in range(self.p_dim):
                plt.subplot(self.p_dim, 1, i + 1)
                if self.smooth is not None:
                    plt.plot(x_vals, self.smooth[:, i], label=r"Estimated $\\beta{i}$ - Smoother", linestyle="--", linewidth=2)
                if self.filt is not None:
                    plt.plot(x_vals[1:], self.filt[:-1, i], label=r"Estimated $\\beta{i}$ - Filter", linestyle="-", linewidth=2)

                plt.grid(linestyle='dashed')
                plt.xlabel('$t/n$',fontsize="xx-large")

                plt.tick_params(axis='both', labelsize=16)
                plt.legend(fontsize="x-large")
            plt.show()