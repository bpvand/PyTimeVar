# -*- coding: utf-8 -*-
"""
Title: 
    
Project: 
    
Author(s):
    
Date modified:
    
"""
# In[]
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class PowerLaw():
    '''
    Class for implementing the Power Law.
    '''

    def __init__(self, vY: np.ndarray, n_powers: float = None):
        self.vY = vY
        self.n = len(self.vY)
        self.p = 2 if n_powers is None else n_powers
        if n_powers is None:
            print('The number of powers is set to 2 by default. \nConsider setting n_powers to 3 or higher if a visual inspection of the data leads you to believe the trend is curly.')

        self.vgamma0 = np.arange(0, 0.01*self.p, 0.01)
        self.bounds = ((-0.495, 8),)*self.p
        self.options = {'maxiter': 5E5}
        self.cons = {'type': 'ineq', 'fun': self._nonlcon}

        self.trendHat = None
        self.gammaHat = None
        self.coeffHat = None

    def fit(self):

        res = minimize(self._construct_pwrlaw_ssr, self.vgamma0,
                       bounds=self.bounds, method='Powell', constraints=self.cons, options=self.options)
        self.gammaHat = res.x.reshape(1, self.p)

        trend = np.arange(1, self.n+1, 1).reshape(self.n, 1)
        mP = trend @ self.gammaHat

        self.coeffHat = np.linalg.pinv(mP.T @ mP) @ mP.T @ self.vY
        self.trendHat = mP @ self.coeffHat

        return self.trendHat

    def _construct_pwrlaw_ssr(self, vparams):
        trend = np.arange(1, self.n+1, 1).reshape(self.n, 1)

        vparams = np.array(vparams).reshape(1, self.p)
        mP = trend @ vparams
        coeff = np.linalg.pinv(mP.T @ mP) @ mP.T @ self.vY

        ssr = np.sum((self.vY - mP @ coeff)**2)
        return ssr

    def _nonlcon(self, params):
        epsilon = 0.005
        c = []
        for id1 in range(self.p-1):
            for id2 in range(id1+1, self.p):
                c.append(params[id1] - params[id2] + epsilon)
        return c

        # In[]
if __name__ == '__main__':
    print('hello')
