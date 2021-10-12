# Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt   
## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

########################### data我也不知道是什么样子 导入的你自己写一下 #################################

	# data = ？
def my_method(data,tau_max=3):
    dataframe = pp.DataFrame(data)
	#####################################################################################################
    N = len( data[0])
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=CMIknn())
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.1)
    p_matrices = results['p_matrix']
    val_matrices = results['val_matrix']


	### 主要添了这一段，其他的你改一改 ######
    lag_matrix = np.zeros([N,N])
    p_matrix = np.zeros([N,N])
    score_matrix = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
	        argmin_p = np.argmin(p_matrices[i][j])
	        lag_matrix[i][j] = argmin_p
	        p_matrix[i][j] = np.min(p_matrices[i][j])
	        score_matrix[i][j] = np.abs(val_matrices[i][j][argmin_p])
    return score_matrix, p_matrix,lag_matrix
	       
if __name__ == '__main__':    
        # Fix random seed
    np.random.seed(42)
    
    # Number of variables N
    N = 4
    
    # Time series length T
    T = 150
    
    # Initialize Gaussian noise
    data = np.random.randn(T, N) 
    
    # Consider a time series coming from the following data generating process
    for t in range(5, T): 
        data[t,0] = 0.7*data[t-1,0] - 0.5*data[t-3,1] + np.random.randn()
        data[t,1] = 0.8*data[t-1,1] + 0.4*data[t-1,2] + np.random.randn()
        data[t,2] = 0.5*data[t-1,2] + np.random.randn() 
        data[t,3] = 0.4*data[t-1,3] + 0.2*data[t-2,2] + np.random.randn()
    
    a,b,c = my_method(data,5)
    print(a)
    print(b)
    print(c)