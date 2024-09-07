import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


class Kalman:
    """
    Class for performing Kalman filtering and smoothing.

    Parameters
    ----------
    T : np.ndarray, optional
        The transition matrix of the state space model.
    Z : np.ndarray, optional
        The observation matrix of the state space model.
    Q : np.ndarray, optional
        The transition covariance matrix of the state space model.
    H : np.ndarray, optional
        The observation covariance matrix of the state space model.
    a_1 : np.ndarray, optional
        The initial mean of the state space model.
    P_1 : np.ndarray, optional
        The initial covariance matrix of the state space model.
    regressors : np.ndarray, optional
        The regressors to use in the model. If provided, the model will be a linear regression model.

    Methods
    -------
    filter(observations)
        Performs the filtering step of the Kalman filter.
    smoother(observations)
        Performs the smoothing step of the Kalman filter.
    """

    def __init__(self, vY: np.ndarray = None, T: np.ndarray = None, Z: np.ndarray = None, R: np.ndarray = None, Q: np.ndarray = None, H: np.ndarray = None, a_1: np.ndarray = None, P_1: np.ndarray = None, regressors: np.ndarray = None):
        self.vY = vY
        # data
        self.n = len(self.vY)
        self.isReg = False
        self.T = T
        if regressors is not None:
            self.isReg = True
            if regressors.ndim == 1:
                k = 1
                self.regressors = regressors.reshape(-1, 1)
            else:
                k = regressors.shape[1]
                self.regressors = regressors

            self.T = T if T is not None else np.eye(k)

            # Create a Z array that can be indexed by t to return the appropriate regressor
            self.Z_reg = np.array([self.regressors[t]
                                  for t in range(self.regressors.shape[0])])
        self.T = self.T if self.T is not None else np.array(
            [[1]])                # Transition matrix
        self.Z = Z if Z is not None else np.array(
            [[1]])                # Observation matrix
        self.Q = Q
        self.bEst_Q, self.bEst_H = False, False
        if self.Q is None:
            # Transition covariance matrix
            self.bEst_Q = True
        self.H = H
        if self.H is None:
            # Observation covariance matrix
            self.bEst_H = True
        self.a_1 = a_1 if a_1 is not None else np.zeros(
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
                 np.linalg.inv(F[t, :, :]@v[t, :, :]))

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
        prediction = np.zeros((self.n, self.m_dim, 1))
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

        Parameters
        ----------
        observations : np.ndarray
            The observed values.

        Returns
        -------
        np.ndarray
            The filtered state means at each time step.
        """
        a, _, _, _, _ = self._filter()

        return a.squeeze()

    def _smoother(self):
        """
        Performs the smoothing step of the Kalman filter.

        Parameters
        ----------
        observations : np.ndarray
            The observed values.

        Returns
        -------
        np.ndarray
            The smoothed state means at each time step.
        np.ndarray
            The smoothed state covariances at each time step.
        """

        a, P, v, F, K = self._filter()
        r = np.zeros((self.n, self.p_dim, 1))
        N = np.zeros((self.n, self.p_dim, self.p_dim))
        a_s = np.zeros((self.n, self.p_dim, 1))
        V_s = np.zeros((self.n, self.p_dim, self.p_dim))

        r[self.n - 1] = 0
        N[self.n - 1] = 0

        for t in range(self.n-1, -1, -1):
            if self.isReg:
                self.Z = self.Z_reg[t].reshape(1, -1)
            L = self.T - K[t] @ self.Z
            if not np.isnan(self.vY[t]):
                r[t - 1] = self.Z.T @ np.linalg.inv(F[t]) @ v[t] + L.T @ r[t]
                N[t -
                    1] = self.Z.T @ np.linalg.inv(F[t]) @ self.Z + L.T @ N[t] @ L
            else:
                r[t - 1] = r[t]
                N[t - 1] = N[t]

            a_s[t] = a[t] + P[t] @ r[t - 1]
            V_s[t] = P[t] - P[t] @ N[t - 1] @ P[t]

        return a_s, V_s

    def smoother(self):
        """
        Performs the smoothing step of the Kalman filter.

        Parameters
        ----------
        observations : np.ndarray
            The observed values.

        Returns
        -------
        np.ndarray
            The smoothed state means at each time step.
        """
        a_s, _ = self._smoother()
        return a_s.squeeze()

    def fit(self, option):

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
        Prints a summary of the results.
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
            plt.plot(x_vals, self.vY, label="Original Series")
            if self.smooth is not None:
                plt.plot(x_vals, self.smooth[:], label="Kalman Smoother", linestyle="--", c='r')
            if self.filt is not None:
                plt.plot(x_vals, self.smooth[:], label="Kalman Filter", linestyle="-", c='k')
            plt.legend()
            plt.xlabel("$t/n$")
            plt.grid(linestyle='dashed')
            plt.show()
            
        else:
            plt.figure(figsize=(6.5, 5 * self.p_dim))
            for i in range(self.p_dim):
                plt.subplot(self.p_dim, 1, i + 1)
                if self.smooth is not None:
                    plt.plot(x_vals, self.smooth[:], label=r"Smooth $\\alpha_{i}$", linestyle="--", c='r')
                if self.filt is not None:
                    plt.plot(x_vals, self.smooth[:], label=r"Filter $\\alpha_{i}$", linestyle="-", c='k')

                plt.xlabel("$t/n$")
                plt.legend()
                plt.grid(linestyle='dashed')
            plt.show()



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def simulate_ar3_data(n_timesteps, phi, sigma):
        # Generate AR(3) data
        y = np.zeros(n_timesteps)
        for t in range(3, n_timesteps):
            y[t] = phi[0]*y[t-1] + phi[1]*y[t-2] + \
                phi[2]*y[t-3] + np.random.normal(0, sigma)
        return y

    def simulate_local_level_data(n_timesteps, sigma_w, sigma_v):
        # Generate local level model data
        x = np.zeros(n_timesteps)
        y = np.zeros(n_timesteps)
        for t in range(1, n_timesteps):
            x[t] = x[t-1] + np.random.normal(0, sigma_w)
            y[t] = x[t] + np.random.normal(0, sigma_v)
        return y

    def simulate_local_linear_trend_data(n_timesteps, sigma_w, sigma_v):
        # Generate local linear trend model data
        x = np.zeros(n_timesteps)
        v = np.zeros(n_timesteps)
        y = np.zeros(n_timesteps)
        for t in range(1, n_timesteps):
            v[t] = v[t-1] + np.random.normal(0, sigma_w)
            x[t] = x[t-1] + v[t-1] + np.random.normal(0, sigma_w)
            y[t] = x[t] + np.random.normal(0, sigma_v)
        return y

    # Parameters for the AR(3) model
    phi = [0.5, -0.2, 0.1]
    sigma_ar3 = 1.0
    n_timesteps = 100

    # Parameters for the local level model
    sigma_w_level = 0.1
    sigma_v_level = 1.0

    # Parameters for the local linear trend model
    sigma_w_trend = 0.1
    sigma_v_trend = 1.0

    # Simulate data
    ar3_data = simulate_ar3_data(n_timesteps, phi, sigma_ar3)
    local_level_data = simulate_local_level_data(
        n_timesteps, sigma_w_level, sigma_v_level)
    local_linear_trend_data = simulate_local_linear_trend_data(
        n_timesteps, sigma_w_trend, sigma_v_trend)

    # Define state space matrices for AR(3) model
    T_ar3 = np.array([[phi[0], 1, 0],
                      [phi[1], 0, 1],
                      [phi[2], 0, 0]])
    Z_ar3 = np.array([[1, 0, 0]])
    Q_ar3 = np.eye(3) * sigma_ar3**2
    H_ar3 = np.array([[1]])
    a_1_ar3 = np.zeros(3)
    P_1_ar3 = np.eye(3)

    # Define state space matrices for local level model
    T_level = np.array([[1]])
    Z_level = np.array([[1]])
    Q_level = np.array([[sigma_w_level**2]])
    H_level = np.array([[sigma_v_level**2]])
    a_1_level = np.array([0])
    P_1_level = np.array([[10**7]])

    # Define state space matrices for local linear trend model
    T_trend = np.array([[1, 1],
                        [0, 1]])
    Z_trend = np.array([[1, 0]])
    Q_trend = np.array([[sigma_w_trend**2, 0],
                        [0, sigma_w_trend**2]])
    H_trend = np.array([[sigma_v_trend**2]])
    a_1_trend = np.array([0, 0])
    P_1_trend = np.eye(2)

    # Apply Kalman filter and smoother for AR(3) model
    kalman_ar3 = Kalman(T=T_ar3, Z=Z_ar3, Q=Q_ar3,
                        H=H_ar3, a_1=a_1_ar3, P_1=P_1_ar3)
    filtered_ar3 = kalman_ar3.filter(ar3_data)
    smoothed_ar3 = kalman_ar3.smoother(ar3_data)

    # Apply Kalman filter and smoother for local level model
    kalman_level = Kalman(T=T_level, Z=Z_level, Q=Q_level,
                          H=H_level, a_1=a_1_level, P_1=P_1_level)
    filtered_level = kalman_level.filter(local_level_data)
    smoothed_level = kalman_level.smoother(local_level_data)

    # Apply Kalman filter and smoother for local linear trend model
    kalman_trend = Kalman(T=T_trend, Z=Z_trend, Q=Q_trend,
                          H=H_trend, a_1=a_1_trend, P_1=P_1_trend)
    filtered_trend = kalman_trend.filter(local_linear_trend_data)
    smoothed_trend = kalman_trend.smoother(local_linear_trend_data)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # AR(3) model
    plt.subplot(3, 1, 1)
    plt.plot(ar3_data, label='AR(3) Data')
    plt.plot(filtered_ar3, label='Filtered')
    plt.plot(smoothed_ar3, label='Smoothed')
    plt.legend()
    plt.title('AR(3) Model')

    # Local level model
    plt.subplot(3, 1, 2)
    plt.plot(local_level_data, label='Local Level Data')
    plt.plot(filtered_level, label='Filtered')
    plt.plot(smoothed_level, label='Smoothed')
    plt.legend()
    plt.title('Local Level Model')

    # Local linear trend model
    plt.subplot(3, 1, 3)
    plt.plot(local_linear_trend_data, label='Local Linear Trend Data')
    plt.plot(filtered_trend, label='Filtered')
    plt.plot(smoothed_trend, label='Smoothed')
    plt.legend()
    plt.title('Local Linear Trend Model')

    plt.tight_layout()
    plt.show()
