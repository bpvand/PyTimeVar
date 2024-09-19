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
    pred : np.ndarray
        The predicted coefficients.
    smooth : 
        The smoothed coefficients.
    
    
    
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

            self.T = np.array([T]).reshape((k,k)) if T is not None else np.eye(k)

            # Create a Z array that can be indexed by t to return the appropriate regressor
            self.Z_reg = np.array([self.mX[t]
                                  for t in range(self.mX.shape[0])])
        else:
            k=1
        
        self.T = np.array(self.T).reshape((k,k)) if self.T is not None else np.array(
            [[1]])                # Transition matrix
        self.Z = np.array(
            [[1]])                # Observation vector
        self.Q = Q
        self.bEst_Q, self.bEst_H = False, False
        if self.Q is None:
            # Transition covariance matrix
            self.bEst_Q = True
        else: 
            self.Q = np.array(self.Q).reshape((k,k))
        self.H = sigma_u
        if self.H is None:
            # Observation covariance matrix
            self.bEst_H = True
        else: 
            self.H = np.array([sigma_u])
        self.a_1 = np.array(b_1).reshape(k,) if b_1 is not None else np.zeros(
            self.T.shape[0])            # Initial state mean
        self.P_1 = np.array(P_1).reshape((k,k)) if P_1 is not None else np.eye(
            self.T.shape[0])*1000      # Initial state covariance
        self.R = np.array(R).reshape((k,k)) if R is not None else np.eye(
            self.a_1.shape[0])         # Transition covariance matrix
        # Dimension of the state space
        self.p_dim = self.T.shape[0]
        # Dimension of the observation space
        self.m_dim = self.Z.shape[0]

        self.H, self.Q = self._estimate_ML()

        self.filt = None
        self.pred = None
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
            vH_init, vQ_init = np.ones(1)*1, np.ones(self.p_dim**2)*1
            vTheta = np.hstack([vH_init, vQ_init])
        elif self.bEst_H and not self.bEst_Q:
            vTheta = np.ones(1)*1
            mQ = self.Q
        elif not self.bEst_H and self.bEst_Q:
            vTheta = np.ones(self.p_dim**2)*1
            mH = self.H
        else:
            return self.H, self.Q
        
        LL_model = minimize(self._compute_likelihood_LL, vTheta, method='L-BFGS-B', bounds = ((1e-8, None),)*len(vTheta),options={'maxiter': 5e5})
        if LL_model.success == False:
            print('Optimization failed')
        else:
            print('Optimization success')

        if self.bEst_H and self.bEst_Q:
            mH, mQ = LL_model.x[0].reshape(1, 1), LL_model.x[1:].reshape(self.p_dim, self.p_dim)
        elif self.bEst_H and not self.bEst_Q:
            mH = LL_model.x.reshape(1,1)
        elif not self.bEst_H and self.bEst_Q:
            mQ = LL_model.x.reshape(self.p_dim, self.p_dim)
        
        k = self.T.shape[0]
        mH = np.array(mH).reshape((1,1))
        mQ = np.array(mQ).reshape((k,k))
        
        return mH, mQ

    def _compute_likelihood_LL(self, vTheta):
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
            self.H, self.Q = vTheta[0].reshape(1,1), vTheta[1:].reshape(self.p_dim, self.p_dim)
        elif self.bEst_H and not self.bEst_Q:
            self.H = vTheta.reshape(1,1)
        elif not self.bEst_H and self.bEst_Q:
            self.Q = vTheta.reshape(self.p_dim, self.p_dim)
        
        a_filt, a_pred, P, v, F, K = self._KalmanFilter()
        dLL = -(self.n*self.m_dim/2)*np.log(2*np.pi)
        for t in range(self.n):
            dLL = dLL - 0.5 * (np.log(np.linalg.det(F[t, :, :])) + v[t, :, :].T @
                 np.linalg.inv(F[t, :, :])@v[t, :, :])
        return -dLL

    def _KalmanFilter(self):
        """
        Performs the Kalman filter.

        Returns
        -------
        a_filt : np.ndarray
            The filtered state at each time step.
        a_pred : np.ndarray
            The predicted state at each time step.
        P : np.ndarray
            The filtered state covariances at each time step.
        v : np.ndarray
            The prediction errors at each time step.
        F : np.ndarray
            The prediction variances at each time step.
        K : np.ndarray
            The Kalman gains at each time step.

        """

        a_pred = np.zeros((self.n + 1, self.p_dim, 1))
        a_filt = np.zeros((self.n, self.p_dim, 1))
        P = np.zeros((self.n + 1, self.p_dim, self.p_dim))
        v = np.zeros((self.n, self.m_dim, 1))
        F = np.zeros((self.n, self.m_dim, self.m_dim))
        K = np.zeros((self.n, self.p_dim, self.m_dim))
        a_pred[0] = self.a_1.reshape(self.T.shape[1], 1)
        P[0] = self.P_1
        for t in range(self.n):
            if self.isReg:
                self.Z = self.Z_reg[t].reshape(1, -1)
            v[t] = self.vY[t] - self.Z @ a_pred[t]
            # print('H',self.H)
            # print('P',self.Z @ P[t] @ self.Z.T)
            # print('F',self.Z @ P[t] @ self.Z.T + self.H)
            F[t] = self.Z @ P[t] @ self.Z.T + self.H
            
            if not np.isnan(self.vY[t]):
                mAux = P[t] @ self.Z.T @ np.linalg.inv(F[t])
                K[t] = self.T @ mAux
                a_filt[t] = a_pred[t] + mAux @ v[t]
                a_pred[t + 1] = self.T @ a_pred[t] + K[t] @ v[t]
            else:
                K[t] = 0
                a_filt[t] = a_pred[t]
                a_pred[t + 1] = self.T @ a_pred[t]

            P[t + 1] = self.T @ P[t] @ (self.T - K[t]
                                        @ self.Z).T + self.R @ self.Q @ self.R.T  
        
        return a_filt, a_pred, P, v, F, K

    def _filter(self):
        """
        Computes the filtered states using the Kalman filter.

        Returns
        -------
        np.ndarray
            The filtered states at each time step.
        """
        a_filt , _, _, _, _, _ = self._KalmanFilter()

        return a_filt.squeeze()
    
    def _predict(self):
        """
        Computes the one-step ahead predictions using the Kalman filter.

        Returns
        -------
        np.ndarray
            The one-step ahead predicted state means at each time step.
        """
        a_filt , a_pred, _, _, _, _ = self._KalmanFilter()

        return a_pred.squeeze()
    

    def _KalmanSmoother(self):
        """
        Performs the smoothing steps of the Kalman filter.

        Returns
        -------
        a_s : np.ndarray
            The smoothed state means at each time step.
        V_s : np.ndarray
            The smoothed state covariances at each time step.
        """

        a_filt, a, P, v, F, K = self._KalmanFilter()
        a = a[:-1]
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
        
        

    def _smoother(self):
        """
        Computes the smoothed state estimates using the Kalman smoother.

        Returns
        -------
        np.ndarray
            The smoothed state means at each time step.
        """
        a_s, _ = self._KalmanSmoother()
        return a_s.squeeze()

    def fit(self, option='filter'):
        '''
        Computes the Kalman filtered states, one-step ahead predicted states or smoothed states for the data.

        Parameters
        ----------
        option : string
            Denotes the fitted trend: filter, predictor, smoother, or all.

        Raises
        ------
        ValueError
            No valid option is provided.

        Returns
        -------
        np.ndarray
            Estimated trend. 
            If option='all', a list of trends is returned:
                [filter, predictor, smoother]

        '''
        if option.lower() == 'filter':
            self.filt = self._filter()
            return self.filt
        if option.lower() == 'predictor':
            self.pred = self._predict()
            return self.pred
        elif option.lower() == 'smoother':
            self.smooth = self._smoother()
            return self.smooth
        elif option.lower() == 'all':
            self.filt = self._filter()
            self.pred = self._predict()
            self.smooth = self._smoother()
            return [self.filt, self.pred, self.smooth]
        else:
            raise ValueError(
                'Unknown option provided to fit(). Choose either filter, predictor, smoother or all')

    def summary(self):
        """
        Prints a summary of the state-space model specification.
        """

        print("State-space model specification")
        print('='*30)
        print(f"H: {self.H}")
        print(f"Q: {self.Q}")
        print(f"R: {self.R}")
        print(f"T: {self.T}\n")


    def plot(self, individual=False, tau: list = None):
        """
        Plot the estimated beta coefficients over a normalized x-axis from 0 to 1 or over a date range.
        
        Parameters
        ----------
        individual : bool, optional
            If True, the filtered states, the predictions, and smoothed states are shown in separate figures.
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
        
        if not individual:
            if self.p_dim == 1:
                plt.figure(figsize=(12, 6))
                plt.plot(x_vals[tau_index[0]:tau_index[1]], self.vY[tau_index[0]:tau_index[1]], label="True data", linewidth=2,color='black')
                if self.smooth is not None:
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], self.smooth[:], label="Estimated $\\beta_{0}$ - Smoother", linestyle="--", linewidth=2)
                if self.pred is not None:
                    plt.plot(x_vals[tau_index[0]+1:tau_index[1]], self.pred[tau_index[0]+1:tau_index[1]], label="Estimated $\\beta_{0}$ - Predictor", linestyle="-", linewidth=2)
                if self.filt is not None:
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], self.filt[tau_index[0]:tau_index[1]], label="Estimated $\\beta_{0}$ - Filter", linestyle="-.", linewidth=2)
                
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
                        plt.plot(x_vals[tau_index[0]:tau_index[1]], self.smooth[tau_index[0]:tau_index[1], i], label=r"Estimated $\\beta{i}$ - Smoother", linestyle="--", linewidth=2)
                    if self.smooth is not None:
                        plt.plot(x_vals[tau_index[0]+1:tau_index[1]], self.pred[tau_index[0]+1:tau_index[1], i], label="Estimated $\\beta_{0}$ - Predictor", linestyle="-", linewidth=2)
                    if self.filt is not None:
                        plt.plot(x_vals, self.filt[tau_index[0]:tau_index[1], i], label=r"Estimated $\\beta{i}$ - Filter", linestyle="-.", linewidth=2)
    
                    plt.grid(linestyle='dashed')
                    plt.xlabel('$t/n$',fontsize="xx-large")
    
                    plt.tick_params(axis='both', labelsize=16)
                    plt.legend(fontsize="x-large")
                plt.show()
                
        if individual:
            if self.p_dim == 1:
                if self.smooth is not None:
                    plt.figure(figsize=(12, 6))
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], self.vY[tau_index[0]:tau_index[1]], label="True data", linewidth=2,color='black')
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], self.smooth[tau_index[0]:tau_index[1]], label="Estimated $\\beta_{0}$ - Smoother", linestyle="--", linewidth=2)
                    plt.grid(linestyle='dashed')
                    plt.xlabel('$t/n$',fontsize="xx-large")
        
                    plt.tick_params(axis='both', labelsize=16)
                    plt.legend(fontsize="x-large")
                    plt.show()
                    
                if self.pred is not None:
                    plt.figure(figsize=(12, 6))
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], self.vY[tau_index[0]:tau_index[1]], label="True data", linewidth=2,color='black')
                    plt.plot(x_vals[tau_index[0]+1:tau_index[1]], self.pred[tau_index[0]+1:tau_index[1]], label="Estimated $\\beta_{0}$ - Predictor", linestyle="--", linewidth=2)
                    plt.grid(linestyle='dashed')
                    plt.xlabel('$t/n$',fontsize="xx-large")
        
                    plt.tick_params(axis='both', labelsize=16)
                    plt.legend(fontsize="x-large")
                    plt.show()
                    
                if self.filt is not None:
                    plt.figure(figsize=(12, 6))
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], self.vY[tau_index[0]:tau_index[1]], label="True data", linewidth=2,color='black')
                    plt.plot(x_vals[tau_index[0]:tau_index[1]], self.filt[tau_index[0]:tau_index[1]], label="Estimated $\\beta_{0}$ - Filter", linestyle="--", linewidth=2)
                    plt.grid(linestyle='dashed')
                    plt.xlabel('$t/n$',fontsize="xx-large")
        
                    plt.tick_params(axis='both', labelsize=16)
                    plt.legend(fontsize="x-large")
                    plt.show()
                    
            
            else:
                if self.smooth is not None:
                    plt.figure(figsize=(10, 6 * self.p_dim))
                    for i in range(self.p_dim):
                        plt.subplot(self.p_dim, 1, i + 1)
                        plt.plot(x_vals[tau_index[0]:tau_index[1]], self.smooth[tau_index[0]:tau_index[1], i], label=r"Estimated $\\beta{i}$ - Smoother", linestyle="--", linewidth=2)
                        plt.grid(linestyle='dashed')
                        plt.xlabel('$t/n$',fontsize="xx-large")
        
                        plt.tick_params(axis='both', labelsize=16)
                        plt.legend(fontsize="x-large")
                    plt.show()
                if self.pred is not None:
                    plt.figure(figsize=(10, 6 * self.p_dim))
                    for i in range(self.p_dim):
                        plt.subplot(self.p_dim, 1, i + 1)
                        plt.plot(x_vals[tau_index[0]+1:tau_index[1]], self.pred[tau_index[0]+1:tau_index[1], i], label="Estimated $\\beta_{0}$ - Predictor", linestyle="--", linewidth=2)
                        plt.grid(linestyle='dashed')
                        plt.xlabel('$t/n$',fontsize="xx-large")
        
                        plt.tick_params(axis='both', labelsize=16)
                        plt.legend(fontsize="x-large")
                    plt.show()
                if self.filt is not None:
                    plt.figure(figsize=(10, 6 * self.p_dim))
                    for i in range(self.p_dim):
                        plt.subplot(self.p_dim, 1, i + 1)
                        plt.plot(x_vals[tau_index[0]:tau_index[1]], self.filt[tau_index[0]:tau_index[1], i], label=r"Estimated $\\beta{i}$ - Filter", linestyle="--", linewidth=2)
                        plt.grid(linestyle='dashed')
                        plt.xlabel('$t/n$',fontsize="xx-large")
        
                        plt.tick_params(axis='both', labelsize=16)
                        plt.legend(fontsize="x-large")
                    plt.show()

            
