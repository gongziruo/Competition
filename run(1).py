# -*- coding: UTF-8 -*-
"""
This script can be used to iterate over the datasets of a particular experiment.
Below you import your function "my_method" stored in the module causeme_my_method.

Importantly, you need to first register your method on CauseMe.
Then CauseMe will return a hash code that you use below to identify which method
you used. Of course, we cannot check how you generated your results, but we can
validate a result if you upload code. Users can filter the Ranking table to only
show validated results.
"""

if __name__ == '__main__':

    # Imports
    import numpy as np
    import json
    import zipfile
    import bz2
    import time
    from tqdm import tqdm
    start_t = time.time()
    import numpy
    import matplotlib
    from matplotlib import pyplot as plt
    import sklearn
    import tigramite
    from tigramite import data_processing as pp
    from tigramite import plotting as tp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
    from tigramite.models import LinearMediation, Prediction
    import warnings

    warnings.filterwarnings('ignore')
    results = {}

    '''
    可改参数：显著性，maxlag，minlag
    '''

    # from pcmci_with_imm import my_method

    # from BASE_algorithm import base_lingam as my_method
    # method_name,results['method_sha']  = ['score-base',"d9373c93ce4744a9848067442fc9c858"]

    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--maxlag', type=int, default=3)
    parser.add_argument('--method', type=str, default='SELF2')
    parser.add_argument('--process', type=int, default=10)
    parser.add_argument('--type', type=str, default= None)
    parser.add_argument('--size', type=str, default=None)
    parser.add_argument('--s_val', type=float, default=0)
    args = parser.parse_args()

    print(args.maxlag, args.method, args.process, args.type, args.size)

    maxlags = args.maxlag
    results['model'] = args.type
    experimental_setup = args.size
    process = args.process

    method_hash = dict()
    method_hash['SELF1'] = "d9373c93ce4744a9848067442fc9c858"
    method_hash['SELF2'] = "6c90954eb4504e9d91b444c3c4d3938c"
    method_hash['VAR_L'] = "293098534d1047f4af05e11830c23070"
    method_hash['PCMCI_L'] = "4d11e387d7af45bf9ac87fb5f1d569cb"
    method_hash['PCMCI_N'] =  "b85469e159094514862b029bc960e730"

    if (args.method not in method_hash.keys()):
        raise('Please choose existed method:'+str(method_hash.keys()))
    if (args.method=='SELF2'):
        from Self_version2 import my_method
        method_name = args.method
        results['method_sha']  = method_hash[args.method]
    elif(args.method=='SELF1'):
        from Self_ import my_method
        method_name = args.method
        results['method_sha']  = method_hash[args.method]
    elif(args.method=='VAR_L'):
        from causeme_my_method import my_method
        method_name = args.method
        results['method_sha']  = method_hash[args.method]
    elif(args.method=='PCMCI_L'):
        from pcmci import pc as my_method
        method_name = args.method
        results['method_sha']  = method_hash[args.method]
    elif(args.method=='PCMCI_N'):
        from pcmci_nonlinear import pc as my_method
        method_name = args.method
        results['method_sha']  = method_hash[args.method]


    '''
    TestCLIM_N-5_T-250
    '''
    # results['model'] = 'TestCLIM'
    # experimental_setup = 'N-5_T-250'


    results['parameter_values'] = "maxlags=%d" % maxlags
    results['experiment'] = results['model'] + '_' + experimental_setup

    # Adjust save name if needed
    save_name = '{}_{}_{}'.format(method_name,
                                  results['parameter_values'],
                                  results['experiment'])

    # Setup directories (adjust to your needs)
    experiment_zip = 'experiments/%s.zip' % results['experiment']
    results_file = 'results/%s.json.bz2' % (save_name)

    # Start of script
    scores = []
    pvalues = []
    lags = []
    runtimes = []

    # (Note that runtimes on causeme are only shown for validated results, this is more for
    # your own assessment here)

    # Loop over all datasets within an experiment
    # Important note: The datasets need to be stored in the order of their filename
    # extensions, hence they are sorted here
    print("Load data")

    def get_result(data_with_name):
        print(data_with_name[0])
        data = data_with_name[1]
        start_time = time.time()
        val_matrix, p_matrix, lag_matrix = my_method(data, maxlags)
        # val_matrix = ((val_matrix/val_matrix.max()*100)*10**(0.5))*val_matrix.max()/100
        print(val_matrix)
        t = time.time() - start_time
        print(t)
        return val_matrix, p_matrix, lag_matrix,t

    with zipfile.ZipFile(experiment_zip, "r") as zip_ref:
        data_list = []
        for name in tqdm(sorted(zip_ref.namelist())):
            string = "Run {} on {}".format(method_name, name)
            data = np.loadtxt(zip_ref.open(name))
            data_list.append([string,data])

    import multiprocessing as mp

    pool = mp.Pool(processes=process)
    multi_res = [pool.apply_async(get_result, (i,)) for i in data_list]
    results_list = [res.get() for res in multi_res]
    for i in results_list:
            val_matrix, p_matrix, lag_matrix ,runtime = i
    # Now we convert the matrices to the required format

            runtimes.append(runtime)
            # and write the results file
            scores.append(val_matrix.flatten())

            # pvalues and lags are recommended for a more comprehensive method evaluation,
            # but not required. Then you can leave the dictionary field empty
            if p_matrix is not None: pvalues.append(p_matrix.flatten())
            if lag_matrix is not None: lags.append(lag_matrix.flatten())

    #
    # with zipfile.ZipFile(experiment_zip, "r") as zip_ref:
    #     pool = mp.Pool(processes=20)
    #     for name in tqdm(sorted(zip_ref.namelist())):
    #
    #         print("Run {} on {}".format(method_name, name))
    #         data = np.loadtxt(zip_ref.open(name))
    #
    #         # Runtimes for your own assessment
    #         start_time = time.time()
    #         # import pandas as pd
    #         # import seaborn as sns
    #         # sns.pairplot(pd.DataFrame(data))
    #         # plt.show()
    #         # continue
    #
    #         # Run your method (adapt parameters if needed)
    #         r = pool.apply_async(get_result,args=(data,))
    #         val_matrix, p_matrix, lag_matrix = r.get()
    #
    #         # Now we convert the matrices to the required format
    #         # and write the results file
    #         scores.append(val_matrix.flatten())
    #
    #         # pvalues and lags are recommended for a more comprehensive method evaluation,
    #         # but not required. Then you can leave the dictionary field empty
    #         if p_matrix is not None: pvalues.append(p_matrix.flatten())
    #         if lag_matrix is not None: lags.append(lag_matrix.flatten())

    # Store arrays as lists for json
    results['scores'] = np.array(scores).tolist()
    if len(pvalues) > 0: results['pvalues'] = np.array(pvalues).tolist()
    if len(lags) > 0: results['lags'] = np.array(lags).tolist()
    results['runtimes'] = np.array(runtimes).tolist()


    print('Runtime:',start_t-time.time())
    # Save data
    print('Writing results ...')
    results_json = bytes(json.dumps(results), encoding='latin1')
    with bz2.BZ2File(results_file, 'w') as mybz2:
        mybz2.write(results_json)

    # if __name__ == '__main__':
    #     dataframe = pp.DataFrame(data)
    #     datatime = numpy.arange(len(data))
    #     parcorr = ParCorr(significance='analytic')
    #     pcmci = PCMCI(
    #         dataframe=dataframe,
    #         cond_ind_test=parcorr,
    #         verbosity=1)
    #     correlations = pcmci.get_lagged_dependencies(tau_max=10)
    #     lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'x_base': 5, 'y_base': .5})
    #     lag_func_matrix.fig.show()