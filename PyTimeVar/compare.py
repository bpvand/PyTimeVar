from PyTimeVar.datasets import temperature, co2_panel
from PyTimeVar import BoostedHP, LocalLinear, Kalman, GAS
import numpy as np
import matplotlib.pyplot as plt


class Compare:
    """
    Class for performing comparison study on a dataset
    """
    
    def __init__(self, vY : np.ndarray, mX : np.ndarray, dates : np.ndarray, name: str='none'):
        self.vY = vY
        self.mX = mX
        self.dates = dates
        self.name = name
        if self.name == 'none':
            self.name = input("Provide a short name of your dataset here: ")
        
    def compare_kernels(self):
        plt.figure(figsize=(20,10))
        kernels = ['Epanechnikov', 'Gaussian', 'uniform', 'triangular', 'quartic', 'tricube']
        for kernel in kernels:
            llr_model = LocalLinear(self.vY, self.mX, kernel = kernel)
            res = llr_model.fit()
            plt.plot(self.dates, res.betas()[0], linewidth=2, label=kernel)
        
        title = self.name + ' : LLR estimates for different kernel functions'
        plt.legend(prop={'size': 20})
        plt.grid(linewidth = 3)
        plt.title(title, fontsize=25)
        plt.xlabel('Date', fontsize=25)
        plt.ylabel(r'$\beta$', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.show()


    def compare_trends(self):
        LLRModel = LocalLinear(self.vY, self.mX)
        res = LLRModel.fit()
        
        HPmodel = BoostedHP(self.vY)
        HPtrend, HPresiduals = HPmodel.fit(boost=False)
        
        bHPmodel = BoostedHP(self.vY)
        bHPtrend, bHPresiduals = bHPmodel.fit()
        
        Q = np.array([[0.03]])
        kalmanmodel = Kalman(Q=Q)
        smooth_trend = kalmanmodel.smoother(self.vY)
        
        model = GAS(self.vY, self.mX, 'student')
        tGAStrend, _ = model.fit()
        
        plt.plot(self.dates, self.vY, label='Original Series')
        plt.plot(self.dates, HPtrend, label='HP', linestyle='--')
        plt.plot(self.dates, bHPtrend, label='bHP', linestyle='--')
        plt.plot(self.dates, res.betas()[0], label='LLR', linestyle='--')
        plt.plot(self.dates, tGAStrend, label='tGAS', linestyle='--')
        plt.plot(self.dates, smooth_trend, label='Kalman Smoother', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel(self.name.lower())
        plt.title(self.name)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # download data
    temp_data = temperature.load(regions=['World'],start_date='1961-01-01', end_date='2023-12-31')
    vY_temp = temp_data.values
    X_temp = np.ones_like(vY_temp)
    
    co2_data = co2_panel.load(start_date='1900-01-01', end_date='2017-01-01', regions=['AUSTRIA'])
    vY_co2 = co2_data.values
    X_co2 = np.ones_like(vY_co2)
    
    # Figure 1
    # local linear estimation for each dataset
    LLRModel_temp = LocalLinear(vY_temp, X_temp)
    res_temp = LLRModel_temp.fit()
    cb_temp = res_temp.plot_confidence_bands(bootstrap_type="LBWB")

    LLRModel_co2 = LocalLinear(vY_co2, X_co2)
    res_co2 = LLRModel_co2.fit()
    cb_co2 = res_co2.plot_confidence_bands(bootstrap_type="LBWB")
    
    # Figure 2
    # local linear estimation using different kernels. CO2 dataset is used here
    co2_comp = Compare(vY_co2, X_co2, co2_data.index)
    co2_comp.compare_kernels()
    
    # Figure 3
    # LLR and alternative trend estimates. Temperature and CO2 datasets are used here
    # Temperature
    temp_comp = Compare(vY_temp, X_temp, temp_data.index)
    temp_comp.compare_trends()
    
    # CO2
    co2_comp.compare_trends()
    
    
    
