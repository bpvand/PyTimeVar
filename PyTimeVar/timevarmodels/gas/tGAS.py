import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as t_dist
from numpy.linalg import inv
from scipy.special import gammaln, logsumexp

class tGAS:
    """
    A class representing a dynamic Student's t location model.
    
    Attributes:
        y (numpy.ndarray): The input data used for fitting the model.
        nu (float): The degrees of freedom parameter.
        phi (float): The parameter for the autoregressive component.
        k (float): The parameter for the dynamic component.
        T (int): The length of the input data.
        model_type (str): The type of model to fit (e.g., 'local_level', 'local_linear_trend').

    Methods:
        log_likelihood_t: Computes the log-likelihood of the Student's t distribution.
        calculate_ut: Calculates the innovation term.
        calculate_mu: Calculates the mean of the model.
        calculate_beta: Calculates the beta term for the local linear trend model.
        log_likelihood: Computes the log-likelihood of the model.
        fit: Fits the model to the data.
        filter: Filters the data using the fitted model.
        covariance_matrix: Computes the covariance matrix of the model.

    References:
    Based on the following paper:
    Found in this document: 
    Harvey, A., & Luati, A. (2014). Filtering With Heavy Tails. Journal of the American Statistical Association, 109(507), 1112–1122. https://doi.org/10.1080/01621459.2014.887011

    """
    def __init__(self, y, model_type=None, nu=5, phi=0.9, k=0.9):
        self.y = y
        self.nu = nu
        self.phi = phi
        self.k = k
        self.T = len(y)
        self.model_type = model_type

    def log_likelihood_t(self, y, mu, nu, lamb):
        # Use a more stable approach to calculate the constant term
        logC = gammaln((nu + 1) / 2) - (0.5 * np.log(np.pi * nu) + gammaln(nu / 2) + 0.5 * logsumexp([0, 2 * lamb]))

        # Calculate the exponent term
        term = 1 + ((y - mu) ** 2) / (nu * np.exp(2 * lamb) + 1e-10)
        
        # Calculate the log likelihood for each observation
        log_likelihood = logC - ((nu + 1) / 2) * np.log(term + 1e-10)
        return log_likelihood
    
    def calculate_ut(self, y_t, mu_t, nu, lamb):
        numerator = (y_t - mu_t) ** 2 / (nu * np.exp(2 * lamb) + 1e-10)
        denominator = 1 + numerator
        bt = numerator / (denominator + 1e-10)
        ut = (1 - bt) * (y_t - mu_t)
        return ut 

    def calculate_mu(self, mu_t, k, u, omega=None, phi=None, beta_t=None):
        if self.model_type is None:
            mu = omega + phi * mu_t + k * u
        elif self.model_type == 'local_level':
            mu =  mu_t + k * u
        elif self.model_type == 'local_linear_trend':
            mu =  mu_t + beta_t + k * u
        return mu
    
    def calculate_beta(self, beta_t, k, u):
        k2 = k**2/(2-k)
        beta = beta_t + k2 * u
        return beta
    
    def log_likelihood(self, params):
        y = self.y
        if self.model_type is None:
            k,lambda_val, nu,mu_init, omega, phi = params
        elif self.model_type == 'local_level':
            k, lambda_val, nu, mu_init = params
        elif self.model_type == 'local_linear_trend':
            k, lambda_val, nu, mu_init, beta_init = params
            betas = np.zeros(self.T)
            betas[0] = beta_init
        
        T = len(y)
        mu = np.zeros(T)
        mu[0] = mu_init
        u = np.zeros(T)
        log_likelihood = 0
        for i in range(1, T):
            u[i] = self.calculate_ut(y[i], mu[i-1], nu, lambda_val)
            if self.model_type is None:
                mu[i] = self.calculate_mu(mu[i-1], k, u[i],omega, phi)
            elif self.model_type == 'local_level':
                mu[i] = self.calculate_mu(mu[i-1], k, u[i])
            elif self.model_type == 'local_linear_trend':
                mu[i] = self.calculate_mu(mu[i-1], k, u[i],beta_t=betas[i-1])
                betas[i] = self.calculate_beta(betas[i-1], k, u[i])
            log_likelihood += self.log_likelihood_t(y[i], mu[i], nu, lambda_val)
        return -log_likelihood

    def fit(self):
        if self.model_type == None:
            initial_params = [self.k, 0.1, self.nu, self.y[0], 0.1, 0.9]
            bounds = [(1e-5, None), (1e-5, None), (1e-5, None), (None, None), (1e-5, None), (0, 1)]
        elif self.model_type == 'local_level':
            initial_params = [self.k, 0.1, self.nu, self.y[0]]
            bounds = [(1e-5, None), (1e-5, None), (1e-5, None), (None, None)]
        elif self.model_type == 'local_linear_trend':
            initial_params = [self.k, 0.1, self.nu, self.y[0], 0.1]
            bounds = [(1e-5, None), (1e-5, None), (1e-5, None), (None, None), (1e-5, None)]
        
        result = minimize(self.log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')

        if self.model_type == None:
            self.k, self.lambda_val, self.nu, self.mu_init, self.omega, self.phi = result.x
            self.mu = self.filter(self.k, self.lambda_val, self.nu, self.omega, self.phi)
        elif self.model_type == 'local_level':
            self.k, self.lambda_val, self.nu, self.mu_init = result.x
            self.mu = self.filter(self.k, self.lambda_val, self.nu)
        elif self.model_type == 'local_linear_trend':
            self.k, self.lambda_val, self.nu, self.mu_init, self.beta_init = result.x
            self.mu, self.betas = self.filter(self.k, self.lambda_val, self.nu, beta_init=self.beta_init)
        return result

    def filter(self, k, lambda_val, nu, omega=None, phi=None, beta_init=None):
        y = self.y
        T = len(y)
        mu = np.zeros(T)
        u = np.zeros(T)
        if self.model_type == None:
            mu[0] = omega
        elif self.model_type == 'local_level':
            mu[0] = self.mu_init
        elif self.model_type == 'local_linear_trend':
            mu[0] = self.mu_init
            betas = np.zeros(T)
            betas[0] = beta_init
            
        for i in range(1, T):
            u[i] = self.calculate_ut(y[i], mu[i-1], nu, lambda_val)
            if self.model_type == None:
                mu[i] = self.calculate_mu(mu[i-1], k, u[i], omega, phi)
            elif self.model_type == 'local_level':
                mu[i] = self.calculate_mu(mu[i-1], k, u[i])
            elif self.model_type == 'local_linear_trend':
                mu[i] = self.calculate_mu(mu[i-1], k, u[i], beta_t=betas[i-1])
                betas[i] = self.calculate_beta(betas[i-1], k, u[i])
        if self.model_type == 'local_linear_trend':
            return mu, betas
        else:
            return mu
        
       

    def covariance_matrix(self):
        result = self.fit()
        hessian_inv = inv(result.hess_inv.todense())
        return hessian_inv

# Example usage
if __name__ == '__main__':
    from ..datasets.gold.data import load
    gold_data = load(['USD']).dropna()
    load(['USD']).dropna().to_clipboard()
    y = gold_data['USD'].values
    y = np.log(y[:500])
    print('y:', y)
    model = tGAS(y, model_type='local_linear_trend')
    result = model.fit()
    print("Estimated parameters: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(*result.x))
    print('log-likelihood:', result.fun)
    import matplotlib.pyplot as plt
    plt.plot(y, label='Observed')
    plt.plot(model.mu, label='mu')
    plt.legend()
    plt.show()
