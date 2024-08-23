import numpy as np

def kalman_filter(y, X=None, Q_t=None, R_t=None, T_t=None, initial_beta=None, initial_P=None, sigma_u=None):
    """
    Kalman filter for estimating time-varying coefficients in a linear regression model.

    Parameters:
        y (array-like): Observations vector of size (time_steps,).
        X (array-like or None): Regressor matrix of size (time_steps, d). If None, a vector of ones is used (d = 0).
        Q_t (ndarray or None): Process noise covariance matrix. Defaults to a small identity matrix.
        R_t (ndarray or None): Observation noise covariance matrix. Defaults to an identity matrix.
        T_t (ndarray or None): State transition matrix. Defaults to identity matrix.
        initial_beta (array-like or None): Initial state estimate for the coefficients. If None, defaults to zeros.
        initial_P (ndarray or None): Initial error covariance matrix. Defaults to an identity matrix.
        sigma_u (float or None): Observation noise variance. Defaults to a small scalar.

    Returns:
        beta_filtered (list of ndarray): Filtered state estimates for each time step.
        P_filtered (list of ndarray): Filtered error covariance matrices for each time step.
        F (list of ndarray): Innovation covariances for each time step.
        v (list of ndarray): Innovation or measurement residuals for each time step.
    """
    time_steps = len(y)

    if X is None:
        # When X is None, use a vector of ones as the regressor matrix (d = 0)
        X = np.ones((time_steps, 1))
    elif X.ndim == 1:
        # If X is 1D, reshape it to be a column vector (single regressor)
        X = X.reshape(-1, 1)

    n = X.shape[1]  # Number of regressors (including intercept if d = 0)

    # Default T_t (Identity matrix if not provided)
    if T_t is None:
        T_t = np.eye(n)

    # Default Q_t (Small identity matrix if not provided)
    if Q_t is None:
        Q_t = np.eye(n) * 0.001

    # Default R_t (Identity matrix if not provided)
    if R_t is None:
        R_t = np.eye(n)

    # Default sigma_u (Scalar observation noise variance if not provided)
    if sigma_u is None:
        sigma_u = 0.1

    # Initial state estimate (coefficients)
    if initial_beta is None:
        beta_t = np.zeros((n, 1))  # Default to zeros if no prior knowledge
    else:
        beta_t = np.array(initial_beta).reshape(-1, 1)  # Use provided initial estimate

    # Initial error covariance matrix (identity if not provided)
    if initial_P is None:
        P_t = np.eye(n)  # Default to identity matrix
    else:
        P_t = np.array(initial_P)  # Use provided initial error covariance matrix

    # Kalman Filter implementation
    beta_filtered = []
    P_filtered = []
    F = []
    v = []
    K = []

    for t in range(time_steps):
        Z_t = X[t, :].reshape(-1, 1)  # Ensure X_t is a column vector

        # Innovation or measurement residual
        v_t = y[t] - Z_t.T @ beta_t

        # Innovation (or residual) covariance
        F_t = Z_t.T @ P_t @ Z_t + sigma_u

        # Optimal Kalman gain
        K_t = P_t @ Z_t @ np.linalg.inv(F_t)

        # Updated state estimate
        beta_t = T_t @ beta_t + K_t @ v_t

        # Updated error covariance
        P_t = T_t @ P_t @ T_t.T + R_t @ Q_t @ R_t.T - K_t @ F_t @ K_t.T

        # Store the filtered state estimates
        beta_filtered.append(beta_t.copy())
        P_filtered.append(P_t.copy())
        F.append(F_t.copy())
        v.append(v_t.copy())
        K.append(K_t.copy())

    beta_estimates = np.hstack(beta_filtered)
    return beta_estimates, beta_filtered, P_filtered, F, v, K

def kalman_smoother(beta_filtered, P_filtered, F, v, K, X=None, T_t=None, Q_t=None):
    """
    Kalman smoother for obtaining smoothed state estimates.

    Parameters:
        beta_filtered (list of ndarray): Filtered state estimates for each time step.
        P_filtered (list of ndarray): Filtered error covariance matrices for each time step.
        F (list of ndarray): Innovation covariances for each time step.
        v (list of ndarray): Innovation or measurement residuals for each time step.
        K (list of ndarray): Kalman gain for each time step.
        X (ndarray): Regressor matrix.
        T_t (ndarray): State transition matrix.
        Q_t (ndarray): Process noise covariance matrix.

    Returns:
        beta_smoothed (ndarray): Smoothed state estimates.
    """
    # Default T_t (Identity matrix if not provided)
    time_steps = len(beta_filtered)

    if X is None:
        # When X is None, use a vector of ones as the regressor matrix (d = 0)
        X = np.ones((time_steps, 1))
    elif X.ndim == 1:
        # If X is 1D, reshape it to be a column vector (single regressor)
        X = X.reshape(-1, 1)

    n = X.shape[1]
    if T_t is None:
        T_t = np.eye(n)

    # Default Q_t (Small identity matrix if not provided)
    if Q_t is None:
        Q_t = np.eye(n) * 0.001



    # Initialize smoother variables
    r_t = np.zeros((len(X), n, 1))
    N_t = np.zeros((n, n))

    beta_smoothed = np.zeros((n, time_steps))

    for t in range(time_steps - 1, -1, -1):

        Z_t = X[t, :].reshape(-1, 1)

        # Compute L_t
        L_t = T_t - K[t] @ Z_t.T

        # Smoothing recursion
        r_t[t-1] = Z_t @ np.linalg.inv(F[t]) @ v[t] + L_t.T @ r_t[t]
        N_t = Z_t @ np.linalg.inv(F[t]) @ Z_t.T + L_t.T @ N_t @ L_t

        # Smoothed state estimate
        beta_smoothed[:, t] = (beta_filtered[t] + P_filtered[t] @ r_t[t-1]).ravel()

    return beta_smoothed





# beta_smoothed contains the smoothed state estimates

beta_estimates, beta_filtered, P_filtered, F, v, K = kalman_filter(y,initial_beta=7)
beta_smoothed = kalman_smoother(beta_filtered, P_filtered, F, v, K)