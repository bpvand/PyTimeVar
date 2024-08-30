import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t, multivariate_normal
import time
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


class GAS:
    """
    Class for performing GAS filtering.

    """

    def __init__(self, vY: np.ndarray, mX: np.ndarray, method: str = 'none'):
        self.vY = vY.flatten()
        self.mX = mX
        self.n = len(vY)
        self.n_est = np.shape(mX)[1]
        self.method = method.lower()
        self.betas = None
        self.params = None

    def fit(self):

        if self.method == 'none':
            print('Warning: no filter method is specified. A t-GAS filter is computed.')
            self.method = 'student'

        # set initial values
        dnInitial = int(np.ceil(self.n / 10))
        vbeta0 = np.linalg.inv((self.mX[:dnInitial, :]).T @ (self.mX[:dnInitial, :])) @ (
            (self.mX[:dnInitial, :]).T @ self.vY[:dnInitial])
        LB = np.concatenate(([0.001], -10 * np.ones(self.n_est), -
                            np.ones(self.n_est), -50 * np.ones(self.n_est)))
        UB = np.concatenate(([100], 10 * np.ones(self.n_est),
                            np.ones(self.n_est), 50 * np.ones(self.n_est)))

        options = {'maxfun': 5E5}

        start_time = time.time()

        if self.method == 'gaussian':  # MLE by Gaussian-GAS

            vdelta0 = 1
            vtheta0 = np.concatenate(
                [0.1*np.ones(self.n_est),  0.1*np.ones(self.n_est), 0.1*np.ones(self.n_est)])
            vgamma0 = np.concatenate([[vdelta0], vtheta0])

            def fgGAS_lh(vpara): return - \
                self.construct_likelihood(vbeta0, vpara)
            result = minimize(fgGAS_lh, vgamma0, bounds=list(
                zip(LB, UB)), options=options)
            vparaHat_gGAS = result.x
            vparaHat = vparaHat_gGAS

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

                mNablat = vxt * (yt - vbetaNow.T @ vxt)
                vbetaNow = vomegaHat_gGAS + mBHat_gGAS * \
                    vbetaNow + mAHat_gGAS * mNablat.squeeze()

                mBetaHat_gGAS[:, id] = vbetaNow

            mBetaHat = mBetaHat_gGAS.T

        elif self.method == 'student':  # MLE by t-GAS
            LB = np.concatenate(([0.01], LB))
            UB = np.concatenate(([200], UB))

            vdelta0 = np.array([10, 1])
            vtheta0 = np.concatenate(
                [np.ones(self.n_est), -0.1 * np.ones(self.n_est), 0.1 * np.ones(self.n_est)])
            vgamma0 = np.concatenate([vdelta0, vtheta0])

            def ftGAS_lh(vpara): return - \
                self.construct_likelihood(vbeta0, vpara)
            result = minimize(ftGAS_lh, vgamma0, bounds=list(
                zip(LB, UB)), options=options)
            vparaHat_tGAS = result.x
            vparaHat = vparaHat_tGAS

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

                temp1 = (1 + dnuHat_tGAS**(-1)) * (1 + dnuHat_tGAS**(-1)
                                                   * ((yt - vbetaNow.T @ vxt) / dsigmauHat_tGAS)**2)**(-1)
                mNablat = (1 + dnuHat_tGAS)**(-1) * (3 + dnuHat_tGAS) * \
                    temp1 * vxt * (yt - vbetaNow.T @ vxt)
                vbetaNow = vomegaHat_tGAS + mBHat_tGAS * \
                    vbetaNow + mAHat_tGAS * mNablat.squeeze()

                mBetaHat_tGAS[:, id] = vbetaNow

            mBetaHat = mBetaHat_tGAS.T

        self.betas, self.params = mBetaHat, vparaHat
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return mBetaHat, vparaHat

    def construct_likelihood(self, vbeta0, vpara):
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

                lhVal += ((yt - vbetaNow.T @ vxt) / dsigmau)**2

                temp1 = (1 + dnu**(-1)) * (1 + dnu**(-1) *
                                           ((yt - vbetaNow.T @ vxt) / dsigmau)**2)**(-1)
                mNablat = (1 + dnu)**(-1) * (3 + dnu) * \
                    temp1 * vxt * (yt - vbetaNow.T @ vxt)
                vbetaNow = vomega + mB * vbetaNow + mA * mNablat.squeeze()

            lhVal = -0.5 * (dnu + 1) * lhVal / self.n + gammaln((dnu + 1) / 2) - \
                gammaln(dnu / 2) - 0.5 * np.log(np.pi * dnu) - np.log(dsigmau)

        return lhVal

    def _format_x_axis(self, ax, date_list):
        def format_func(x, pos):
            if len(date_list) > 0:
                index = int(x)
                if 0 <= index < len(date_list):
                    date = date_list[index]
                    if date_list[-1] - date_list[0] > pd.Timedelta(days=730):
                        return date.strftime('%Y')
                    else:
                        return date.strftime('%b %Y')
            return ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    def _generate_dates(self, length, start_date, end_date):
        # Calculate the total duration in days
        total_days = (end_date - start_date).days

        # Generate a date range with the appropriate number of periods
        dates = pd.date_range(start=start_date, end=end_date, periods=length)

        return dates

    def plot(self, date_range=None):
        """
        Plot the beta coefficients over a normalized x-axis from 0 to 1 or over a date range.
        """
        fig, axs = plt.subplots(self.n_est, 1, figsize=(10, 6))

        # Ensure axs is always an array even if there's only one subplot
        if self.n_est == 1:
            axs = [axs]

        if date_range:
            start_date, end_date = [datetime.strptime(
                date, "%Y-%m-%d") for date in date_range]
            x_vals = self._generate_dates(
                self.betas.shape[0], start_date, end_date)
        else:
            x_vals = np.linspace(0, 1, self.betahat.shape[1])

        for i in range(self.n_est):
            axs[i].plot(x_vals, self.betas[:, i],
                        label=r'$\beta_{{{:2d}}}$'.format(i+1))
            axs[i].set_title(r'$\beta_{{{:2d}}}$'.format(i+1))
            axs[i].set_xlabel("Date" if date_range else "t/n")
            axs[i].set_ylabel(r"$\beta$ Value")
            axs[i].legend()
            if date_range:
                self._format_x_axis(axs[i], x_vals)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Parameter specification for simulating data
    n = 2000
    nu = 2.1
    burnin = 20
    sigmau = 0.1
    mA = np.array([[0.3, 0.1], [0.1, 0.2]])

    # Simulate data
    def beta0x(x): return -4 * x**3 + 9 * x**2 - 6 * x + 2
    def beta1x(x): return 1.5 * np.exp(-10 * (x - 0.2)**2) + \
        1.6 * np.exp(-8 * (x - 0.8)**2)

    def beta2x(x): return -0.5 * x - 0.5 * np.exp(-5 * (x - 0.8)**2)
    trend = np.arange(1, n + 1) / n
    mbeta = np.vstack([beta0x(trend), beta1x(trend), beta2x(trend)]).T

    mXi = multivariate_normal.rvs(mean=np.zeros(
        2), cov=np.eye(2), size=n + burnin).T
    mX = np.zeros((2, n + burnin))
    for i in range(1, n + burnin):
        mX[:, i] = mA @ mX[:, i - 1] + mXi[:, i]
    mX = np.hstack([np.ones((n, 1)), mX[:, -n:].T])
    vu = sigmau * t.rvs(nu, size=n)
    vy = np.sum(mbeta * mX, axis=1) + vu

    # MLE by Gaussian-GAS
    gGAS = GAS(vy, mX, method='gaussian')
    mBetaHat_gGAS, vparaHat_gGAS = gGAS.fit()

    # MLE by t-GAS
    tGAS = GAS(vy, mX, method='student')
    mBetaHat_tGAS, vparaHat_tGAS = tGAS.fit()

    # Make plots
    for id2 in range(mX.shape[1]):
        plt.figure(figsize=(12, 6))
        plt.plot(mbeta[:, id2], '-k', linewidth=3, label='true')
        plt.plot(mBetaHat_gGAS[:, id2], '-.r',
                 linewidth=3, label='$\\mathcal{N}$-GAS')
        plt.plot(mBetaHat_tGAS[:, id2], '--b', linewidth=3, label='$t$-GAS')
        plt.grid(which='minor')
        plt.legend(fontsize=20, loc='best', frameon=False)
        plt.gca().tick_params(axis='both', which='major', labelsize=20)

        plt.show()
