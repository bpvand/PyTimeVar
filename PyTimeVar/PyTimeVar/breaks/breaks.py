import numpy as np
import matplotlib.pyplot as plt


class Breaks:
    """
    Class for structural breaks estimation.
    
    Parameters
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    mX : np.ndarray
        The independent variable (predictor) matrix.
    iH : int
        The minimal length of a segment.
    iM : int
        The maximum number of structural breaks allowed.
    dPara_trimming : float
        The value of the trimming (in percentage).
        
    Attributes
    ----------
    vY : np.ndarray
        The dependent variable (response) array.
    mX : np.ndarray
        The independent variable (predictor) matrix.
    iH : int
        The minimal length of a segment.
    iM : int
        The maximum number of structural breaks allowed.
    dPara_trimming : float
        The value of the trimming (in percentage).
    n_est : int
        The number of regressors.
    n : int
        The sample size.
    mBetaHat : np.ndarray
        The matrix of estimated coefficients between all break dates.
    glb : np.ndarray
        The vector of optimal SSR values.
    datevec : np.ndarray
        The matrix with estimated break dates (loc matrix, MATLAB-style).
    bigvec : np.ndarray
        ((bigt*(bigt+1)//2) x 1) vector storing SSR values.
    break_location : np.ndarray
        The (iMx1) vector of break locations.
    
    

    """

    def __init__(self, vY: np.ndarray, mX: np.ndarray, iH: int = 1, iM: int = 1, dPara_trimming: float = 0.15):
        self.vY = vY.reshape(-1,1)
        self.mX = mX
        self.iH = iH
        self.iM = iM
        self.dPara_trimming = dPara_trimming
        
        self.n = len(vY)
        self.n_est = np.shape(mX)[1]
        
        self.mBetaHat = None
        self.glb = None
        self.datevec = None
        self.bigvec = None
        self.break_location = None


    def _ssr(self, start, last):
        """
        Compute the recursive sum of squared residuals (SSR)
        for observations from index 'start' to 'last'.
        """
        vecssr = np.zeros((last, 1))
        z_seg = self.mX[start-1 : start-1+self.iH, :]
        inv1 = np.linalg.inv(z_seg.T @ z_seg)
        delta1 = inv1 @ (z_seg.T @ self.vY[start-1 : start-1+self.iH, :])
        res = self.vY[start-1 : start-1+self.iH, :] - z_seg @ delta1
        vecssr[start+self.iH-2, 0] = (res.T @ res)[0, 0]
        
        r = start + self.iH
        while r <= last:
            v = self.vY[r-1, 0] - (self.mX[r-1, :] @ delta1)[0]
            invz = inv1 @ self.mX[r-1, :].reshape(-1, 1)
            f = 1 + (self.mX[r-1, :].reshape(1, -1) @ invz)[0, 0]
            delta2 = delta1 + invz * v        
            inv2 = inv1 - (invz @ invz.T) / f
            inv1 = inv2
            delta1 = delta2
            vecssr[r-1, 0] = vecssr[r-2, 0] + v*v/f
            r += 1
        return vecssr
    
    def _parti(self, start, b1, b2, last, bigvec):
        """
        Determine the optimal one-break partition for a segment starting at 'start'
        and ending at 'last', considering break dates from b1 to b2.
        """
        dvec_local = np.zeros((self.n, 1))
        ini = (start - 1) * self.n - (start - 2) * (start - 1) // 2 + 1
        ini_idx = int(ini - 1) 
        j = b1
        while j <= b2:
            k = int(j * self.n - (j - 1) * j // 2 + last - j)
            dvec_local[j-1, 0] = bigvec[ini_idx + (j - start), 0] + bigvec[k - 1, 0]
            j += 1
        sub_dvec = dvec_local[b1-1 : b2, 0]
        ssrmin = np.min(sub_dvec)
        minindcdvec = int(np.argmin(sub_dvec))
        dx = (b1 - 1) + (minindcdvec + 1)
        return ssrmin, dx
    
    def _estimate_coefficients(self):
        mBeta = np.zeros((self.n, self.n_est))
        print(self.datevec.shape)
        vSegments = np.concatenate(([0], self.break_location, [self.n])).astype(int)
        for i in range(self.iM+1):
            start, end = vSegments[i], vSegments[i+1]
            vY_seg = self.vY[start:end]
            mX_seg = self.mX[start:end,:]
            vBeta_seg = np.linalg.inv(mX_seg.T@mX_seg)@(mX_seg.T@vY_seg)
            mBeta[start:end,:] = vBeta_seg
        
        return mBeta
    
    def fit(self):
        """
        Main procedure that computes break points that globally minimize the SSR.
        
        Returns:
          glb     : (m x 1) array of optimal SSR values.
          datevec : (m x m) array with break dates (loc matrix, MATLAB-style).
          bigvec  : ((bigt*(bigt+1)//2) x 1) vector storing SSR values.
          
        """
        datevec = np.zeros((self.iM, self.iM))
        optdat  = np.zeros((self.n, self.iM))
        optssr  = np.zeros((self.n, self.iM))
        dvec    = np.zeros((self.n, 1))
        glb     = np.zeros((self.iM, 1))
        bigvec  = np.zeros((self.n * (self.n + 1) // 2, 1))
        
        start_trimming = int(np.floor(self.dPara_trimming * self.n))
        end_trimming   = int(np.floor((1 - self.dPara_trimming) * self.n))
        
        # --- Build bigvec ---
        for i in range(1, self.n - self.iH + 2):
            vecssr = self._ssr(i, self.n)
            start_idx = int((i - 1) * self.n + i - ((i - 1) * i) // 2) - 1
            end_idx   = int(i * self.n - ((i - 1) * i) // 2)
            bigvec[start_idx:end_idx, 0] = vecssr[i-1:self.n, 0]
        
        # --- Main algorithm ---
        if self.iM == 1:
            ssrmin, datx = self._parti(1, self.iH, self.n - self.iH, self.n, bigvec)
            datevec[0, 0] = datx
            glb[0, 0] = ssrmin
        else:
            # First stage: one-break partitions (first column)
            for j1 in range(2 * self.iH, self.n + 1):
                ssrmin, datx = self._parti(1, self.iH, j1 - self.iH, j1, bigvec)
                optssr[j1 - 1, 0] = ssrmin
                optdat[j1 - 1, 0] = datx
            glb[0, 0] = optssr[self.n - 1, 0]
            datevec[0, 0] = optdat[self.n - 1, 0]
            
            # Subsequent stages: for ib = 2 to m
            for ib in range(2, self.iM + 1):
                if ib == self.iM:
                    jlast = self.n
                    for jb in range(ib * self.iH, jlast - self.iH + 1):
                        dvec[jb - 1, 0] = optssr[jb - 1, ib - 2] + \
                            bigvec[int((jb + 1) * self.n - jb * (jb + 1) // 2) - 1, 0]
                    sub_range = dvec[ib * self.iH - 1 : jlast - self.iH, 0]
                    optssr[jlast - 1, ib - 1] = np.min(sub_range)
                    minindcdvec = int(np.argmin(sub_range))
                    optdat[jlast - 1, ib - 1] = (ib * self.iH - 1) + (minindcdvec + 1)
                else:
                    for jlast in range((ib + 1) * self.iH, self.n + 1):
                        for jb in range(ib * self.iH, jlast - self.iH + 1):
                            dvec[jb - 1, 0] = optssr[jb - 1, ib - 2] + \
                                bigvec[int(jb * self.n - jb * (jb - 1) // 2 + jlast - jb) - 1, 0]
                        sub_range = dvec[ib * self.iH - 1 : jlast - self.iH, 0]
                        optssr[jlast - 1, ib - 1] = np.min(sub_range)
                        minindcdvec = int(np.argmin(sub_range))
                        optdat[jlast - 1, ib - 1] = (ib * self.iH - 1) + (minindcdvec + 1)
                
                # Backward recursion to set earlier break dates.
                datevec[ib - 1, ib - 1] = optdat[self.n - 1, ib - 1]
                for i_inner in range(1, ib):
                    xx = ib - i_inner
                    prev_break = int(datevec[xx, ib - 1])
                    datevec[xx - 1, ib - 1] = optdat[prev_break - 1, xx - 1]
                glb[ib - 1, 0] = optssr[self.n - 1, ib - 1]
        
        # --- Final trimming adjustments ---
        if datevec[0, 1] < start_trimming:
            datevec[0, 1] = start_trimming
        elif datevec[1, 1] > end_trimming:
            datevec[1, 1] = end_trimming
        
        self.glb = glb
        self.datevec = datevec
        self.bigvec = bigvec
        self.break_location = self.datevec[:,-1]
        
        # Fit coefficients between break points
        mBetaHat = self._estimate_coefficients()
        self.mBetaHat = mBetaHat
        
        return mBetaHat, glb, self.break_location
    
    def plot(self, tau: list = None):
        '''
        Plot the beta coefficients over a normalized x-axis from 0 to 1.
        
        Parameters
        ----------
        tau : list, optional
            The list looks the  following: tau = [start,end].
            The function will plot all data and estimates between start and end.
            
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
            
        vBreaks_normalized = self.break_location / self.n
        
        if self.n_est == 1:
    
            plt.figure(figsize=(12, 6))
            plt.plot(x_vals[tau_index[0]:tau_index[1]], self.vY[tau_index[0]:tau_index[1]], label="True data", linewidth=1,color='black')
            plt.plot(x_vals[tau_index[0]:tau_index[1]], self.mBetaHat[tau_index[0]:tau_index[1]], label="Estimated $\\beta_{0}$", linestyle="--", linewidth=2)
            
            # Add vertical lines for break dates
            for break_point in vBreaks_normalized:
                if tau is None or (min(tau) <= break_point <= max(tau)):
                    plt.axvline(x=break_point, color='r', linestyle='--', linewidth=0.8, label='Break date' if break_point == vBreaks_normalized[0] else "")
            
            
            plt.grid(linestyle='dashed')
            plt.xlabel('$t/n$',fontsize="xx-large")
            plt.tick_params(axis='both', labelsize=16)
            plt.legend(fontsize="x-large")
            plt.show()

        else:
            plt.figure(figsize=(10, 6 * self.n_est))
            for i in range(self.n_est):
                plt.subplot(self.n_est, 1, i + 1)
                plt.plot(x_vals[tau_index[0]:tau_index[1]], (self.mBetaHat[:, i])[tau_index[0]:tau_index[1]], label=f'Estimated $\\beta_{i}$', color='black', linewidth=2)
                
                # Add vertical lines for break dates
                for break_point in vBreaks_normalized:
                    if tau is None or (min(tau) <= break_point <= max(tau)):
                        plt.axvline(x=break_point, color='r', linestyle='--', linewidth=0.8, label='Break date' if break_point == vBreaks_normalized[0] else "" )
                
                
                plt.grid(linestyle='dashed')
                plt.xlabel('$t/n$',fontsize="xx-large")
                plt.tick_params(axis='both', labelsize=16)
                plt.legend(fontsize="x-large")
            plt.show()
