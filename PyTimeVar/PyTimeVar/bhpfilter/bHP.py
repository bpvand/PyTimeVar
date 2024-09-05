import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from tqdm import tqdm

class BoostedHP:
    """
    Class for performing the boosted HP filter

    Attributes
    ----------
    vY : array-like
        The input time series data.
    dLambda : float
        The smoothing parameter lambda.
    iMaxIter : int
        The maximum number of iterations for the boosting algorithm.
    results : tuple
        A tuple containing the results of the Boosted HP filter.

    Methods
    -------
    __init__(self, vY, dLambda=1600, iMaxIter=100)
        Initializes the BoostedHP object.
    fit(self, boost=True, stop='adf', dAlpha=0.05, verbose=False)
        Fits the Boosted HP filter to the data.
    summary(self)
        Prints a summary of the results.
    plot(self)
        Plots the original series and the trend component.
    """

    def __init__(self, vY, dLambda=1600, iMaxIter=100):
        """
        Initializes the BoostedHP object.

        Parameters
        ----------
        vY : array-like
            The input time series data.
        dLambda : float
            The smoothing parameter lambda.
        iMaxIter : int
            The maximum number of iterations for the boosting algorithm.
        """
        self.vY = vY.flatten()
        self.dLambda = dLambda
        self.iMaxIter = iMaxIter
        self.results = None

    def fit(self, boost=True, stop="adf", dAlpha=0.05, verbose=False):
        """
        Fits the Boosted HP filter to the data.

        Parameters
        ----------
        boost : bool
            Whether to use boosting.
        stop : str
            Stopping criterion ('adf', 'bic', 'aic', 'hq').
        dAlpha : float
            The significance level for the stopping criterion.
        verbose : bool
            Whether to display a progress bar.

        Returns
        -------
        tuple
            The trend component of the time series and the residuals.
        """
        self.dAlpha = dAlpha
        self.stop = stop
        self.results = self._bHP(
            self.vY, boost, self.dLambda, stop, self.dAlpha, self.iMaxIter, verbose
        )
        mTrends, vCurrentRes, vIC_values, iM, vBHP = self.results
        return vBHP, vCurrentRes

    def summary(self):
        """
        Prints a summary of the results.
        """
        if self.results is None:
            print("Model is not fitted yet.")
            return

        mTrends, vCurrentRes, vIC_values, iM, vBHP = self.results
        print(f"Boosted HP Filter Results:")
        print(f"Stopping Criterion: {self.stop}")
        print(f"Max Iterations: {self.iMaxIter}")
        print(f"Iterations Run: {iM}")
        print(f"Lambda: {self.dLambda}")
        print(f"Alpha: {self.dAlpha}")
        print("\nInformation Criteria Values:")
        print(vIC_values)
        print("\nBoosted HP Filter Output:")
        print(vBHP)

    def plot(self):
        """
        Plots the original series and the trend component.
        """
        if self.results is None:
            print("Model is not fitted yet.")
            return

        _, _, _, _, vBHP = self.results
        plt.figure(figsize=(12, 6))
        plt.plot(self.vY, label="Original Series")
        plt.plot(vBHP, label="Trend Component", linestyle="--")
        plt.legend()
        plt.grid(linestyle='dashed')
        plt.show()

    def _bHP(self, vY, bBoost, dLambda, sStop, dAlpha, iMaxIter, verbose):
        """
        Internal method to fit the Boosted HP filter to the data.

        Parameters
        ----------
        vY : array-like
            The input time series data.
        bBoost : bool
            Whether to use boosting.
        dLambda : float
            The smoothing parameter lambda.
        sStop : str
            Stopping criterion.
        dAlpha : float
            The significance level for the stopping criterion.
        iMaxIter : int
            The maximum number of iterations.
        verbose : bool
            Whether to display a progress bar.

        Returns
        -------
        tuple
            The trend component, residuals, information criteria values, number of iterations, and trend component.
        """
        dT = len(vY)
        mS, mI_S = self._comp_matrix_operators(dT, dLambda)

        if not bBoost:
            vHP = mS @ vY
            vRes = vY - vHP
            return 0, vRes, 0, 1, vHP

        if sStop == "adf":
            return self._bHP_adf(vY, iMaxIter, dT, mI_S, dAlpha, verbose)
        elif sStop == "bic" or sStop == "bicnone":
            return self._bHP_bic(vY, iMaxIter, dT, mI_S, mS, sStop, verbose)
        elif sStop == "aic" or sStop == "aicnone":
            return self._bHP_aic(vY, iMaxIter, dT, mI_S, mS, sStop, verbose)
        elif sStop == "hq" or sStop == "hqnone":
            return self._bHP_hq(vY, iMaxIter, dT, mI_S, mS, sStop, verbose)

    def _bHP_adf(self, vY, iMaxIter, dT, mI_S, dAlpha, verbose):
        """
        Internal method for the ADF stopping criterion.

        Parameters
        ----------
        vY : array-like
            The input time series data.
        iMaxIter : int
            The maximum number of iterations.
        dT : int
            The length of the time series.
        mI_S : np.ndarray
            The I-S matrix.
        dAlpha : float
            The significance level for the stopping criterion.
        verbose: bool
            Whether to display a progress bar.

        Returns
        -------
        tuple
            The trend component, residuals, ADF p-values, number of iterations, and trend component.
        """
        bStationary = False
        vCurrentRes = vY
        mTrends = np.zeros((dT, iMaxIter))
        vAdf_pvalues = np.zeros(iMaxIter)
        if verbose:
            for i in tqdm(range(iMaxIter)):
                vCurrentRes = mI_S @ vCurrentRes
                mTrends[:, i] = vY - vCurrentRes
                vAdf_pvalues[i] = ts.adfuller(vCurrentRes, regression="ct", maxlag=1)[1]
                if vAdf_pvalues[i] <= dAlpha:
                    mTrends = mTrends[:, : i + 1]
                    vAdf_pvalues = vAdf_pvalues[: i + 1]
                    break
        else:
            for i in range(iMaxIter):
                vCurrentRes = mI_S @ vCurrentRes
                mTrends[:, i] = vY - vCurrentRes
                vAdf_pvalues[i] = ts.adfuller(vCurrentRes, regression="ct", maxlag=1)[1]
                if vAdf_pvalues[i] <= dAlpha:
                    mTrends = mTrends[:, : i + 1]
                    vAdf_pvalues = vAdf_pvalues[: i + 1]
                    break

        vBHP = vY - vCurrentRes
        return mTrends, vCurrentRes, vAdf_pvalues, i + 1, vBHP

    def _bHP_bic(self, vY, iMaxIter, dT, mI_S, mS, sStop, verbose):
        """
        Internal method for the BIC stopping criterion.

        Parameters
        ----------
        vY : array-like
            The input time series data.
        iMaxIter : int
            The maximum number of iterations.
        dT : int
            The length of the time series.
        mI_S : np.ndarray
            The I-S matrix.
        mS : np.ndarray
            The S matrix.
        sStop : str
            Stopping criterion.
        verbose: bool
            Whether to display a progress bar.

        Returns
        -------
        tuple
            The trend component, residuals, BIC values, number of iterations, and trend component.
        """
        vCurrentRes = vY
        mTrends = np.zeros((dT, iMaxIter))
        vIC_values = np.zeros(iMaxIter)

        vC_HP = mI_S @ vY
        mCurrentI_S = mI_S

        if verbose:
            for i in tqdm(range(iMaxIter)):
                vCurrentRes = mCurrentI_S @ vY
                mTrends[:, i] = vY - vCurrentRes
                mCurrentBm = np.eye(dT) - mCurrentI_S
                vIC_values[i] = np.var(vCurrentRes) / np.var(vC_HP) + np.log(dT) * (
                    np.sum(np.diag(mCurrentBm)) / (dT - np.sum(np.diag(mS)))
                )
                mCurrentI_S = mI_S @ mCurrentI_S
                if i >= 1 and sStop == "bic" and vIC_values[i - 1] < vIC_values[i]:
                    break
        else:
            for i in range(iMaxIter):
                vCurrentRes = mCurrentI_S @ vY
                mTrends[:, i] = vY - vCurrentRes
                mCurrentBm = np.eye(dT) - mCurrentI_S
                vIC_values[i] = np.var(vCurrentRes) / np.var(vC_HP) + np.log(dT) * (
                    np.sum(np.diag(mCurrentBm)) / (dT - np.sum(np.diag(mS)))
                )
                mCurrentI_S = mI_S @ mCurrentI_S
                if i >= 1 and sStop == "bic" and vIC_values[i - 1] < vIC_values[i]:
                    break

        mTrends = mTrends[:, :i]
        vBHP = mTrends[:, i - 1]
        return mTrends, vCurrentRes, vIC_values[: i + 1], i + 1, vBHP

    def _bHP_aic(self, vY, iMaxIter, dT, mI_S, mS, sStop, verbose):
        """
        Internal method for the AIC stopping criterion.

        Parameters
        ----------
        vY : array-like
            The input time series data.
        iMaxIter : int
            The maximum number of iterations.
        dT : int
            The length of the time series.
        mI_S : np.ndarray
            The I-S matrix.
        mS : np.ndarray
            The S matrix.
        sStop : str
            Stopping criterion.
        verbose: bool
            Whether to display a progress bar.

        Returns
        -------
        tuple
            The trend component, residuals, AIC values, number of iterations, and trend component.
        """
        vCurrentRes = vY
        mTrends = np.zeros((dT, iMaxIter))
        vIC_values = np.zeros(iMaxIter)

        vC_HP = mI_S @ vY
        mCurrentI_S = mI_S

        if verbose:
            for i in tqdm(range(iMaxIter)):
                vCurrentRes = mCurrentI_S @ vY
                mTrends[:, i] = vY - vCurrentRes
                mCurrentBm = np.eye(dT) - mCurrentI_S
                vIC_values[i] = np.var(vCurrentRes) / np.var(vC_HP) + 2 * (
                    np.sum(np.diag(mCurrentBm)) / (dT - np.sum(np.diag(mS)))
                )
                mCurrentI_S = mI_S @ mCurrentI_S
                if i >= 1 and vIC_values[i - 1] < vIC_values[i]:
                    break
        else:
            for i in range(iMaxIter):
                vCurrentRes = mCurrentI_S @ vY
                mTrends[:, i] = vY - vCurrentRes
                mCurrentBm = np.eye(dT) - mCurrentI_S
                vIC_values[i] = np.var(vCurrentRes) / np.var(vC_HP) + 2 * (
                    np.sum(np.diag(mCurrentBm)) / (dT - np.sum(np.diag(mS)))
                )
                mCurrentI_S = mI_S @ mCurrentI_S
                if i >= 1 and vIC_values[i - 1] < vIC_values[i]:
                    break

        mTrends = mTrends[:, :i]
        vBHP = mTrends[:, i - 1]
        return mTrends, vCurrentRes, vIC_values[: i + 1], i + 1, vBHP

    def _bHP_hq(self, vY, iMaxIter, dT, mI_S, mS, sStop, verbose):
        """
        Internal method for the HQ stopping criterion.

        Parameters
        ----------
        vY : array-like
            The input time series data.
        iMaxIter : int
            The maximum number of iterations.
        dT : int
            The length of the time series.
        mI_S : np.ndarray
            The I-S matrix.
        mS : np.ndarray
            The S matrix.
        sStop : str
            Stopping criterion.
        verbose: bool
            Whether to display a progress bar.

        Returns
        -------
        tuple
            The trend component, residuals, HQ values, number of iterations, and trend component.
        """
        vCurrentRes = vY
        mTrends = np.zeros((dT, iMaxIter))
        vIC_values = np.zeros(iMaxIter)

        vC_HP = mI_S @ vY
        mCurrentI_S = mI_S

        if verbose:
            for i in tqdm(range(iMaxIter)):
                vCurrentRes = mCurrentI_S @ vY
                mTrends[:, i] = vY - vCurrentRes
                mCurrentBm = np.eye(dT) - mCurrentI_S
                vIC_values[i] = np.var(vCurrentRes) / np.var(vC_HP) + 2 * np.log(
                    np.log(dT)
                ) * (np.sum(np.diag(mCurrentBm)) / (dT - np.sum(np.diag(mS))))
                mCurrentI_S = mI_S @ mCurrentI_S
                if i >= 1 and vIC_values[i - 1] < vIC_values[i]:
                    break
        else:
            for i in range(iMaxIter):
                vCurrentRes = mCurrentI_S @ vY
                mTrends[:, i] = vY - vCurrentRes
                mCurrentBm = np.eye(dT) - mCurrentI_S
                vIC_values[i] = np.var(vCurrentRes) / np.var(vC_HP) + 2 * np.log(
                    np.log(dT)
                ) * (np.sum(np.diag(mCurrentBm)) / (dT - np.sum(np.diag(mS))))
                mCurrentI_S = mI_S @ mCurrentI_S
                if i >= 1 and vIC_values[i - 1] < vIC_values[i]:
                    break

        mTrends = mTrends[:, :i]
        vBHP = mTrends[:, i - 1]
        return mTrends, vCurrentRes, vIC_values[: i + 1], i + 1, vBHP

    def _comp_matrix_operators(self, dT, dLambda):
        """
        Compute matrix operators for the Boosted HP filter.

        Parameters
        ----------
        dT : int
            The length of the time series.
        dLambda : float
            The smoothing parameter lambda.

        Returns
        -------
        tuple
            The S and I-S matrices.
        """
        mIdentity = np.eye(dT)
        mD_temp = np.vstack((np.zeros([1, dT]), np.eye(dT - 1, dT)))
        mD_temp = (mIdentity - mD_temp) @ (mIdentity - mD_temp)
        mD = mD_temp[2:dT].T
        mS = np.linalg.inv(mIdentity + dLambda * mD @ mD.T)
        mI_S = mIdentity - mS
        return mS, mI_S