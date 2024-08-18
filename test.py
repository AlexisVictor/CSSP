import numpy as np
import file_txt as ft
from CSSP import *
import time
import pandas as pd


# np.set_printoptions(precision=3)

def test_find_optimal_params_SLS(Lambdas=[1.0], steps=[1e-1], n1 = 20, n2 = 50, delta = 10, s = 10, n_test = 10, itermax = 2000, stochastic = True, save = False, bigsize= True, size = 'large', tol = 1e-4): # L_1O vs SLS 
    if (size != 'large'):
        np.printoptions(precision=3) 
    ld_max = Lambdas[-1]
    ld_min = Lambdas[0]
    n_ld = len(Lambdas)
    step_min = steps[0]
    step_max = steps[-1]
    n_steps = len(steps)
    results_dict_SLS = {}
    results_dict_SLS_std = {}
    j = 0
    for ld in Lambdas:  
        for step in steps:
            nn = []
            j += 1
            # print(j)
            err_SLS_arr = np.zeros(n_test)
            nn = np.zeros(n_test)
            start_time = time.time()
            print('ld : ', ld, 'step : ', step)
            if bigsize:
                for i in range(n_test):
                    # print('test ', i)
                    # print('test ', i)
                    (indexes, m, n, t) = SLS( X, n1, n2, s, ld, step, itermax, display=False, tol = tol, stochastic=stochastic, info = True)
                    err_SLS_arr[i] = Approximation_factor(X, s, indexes)
                    nn[i] = n
                    # print(n)
            else:
                for i in range(n_test):
                    X = ft.matrix_from_file(i, n1, n2)
                    (bt_ind, bt) = brute_force(X, n1, n2, s)
                    (indexes, m, n, t) = SLS( X, n1, n2, s, ld, step, itermax, display=False, tol=tol, info = True)
                    err_SLS_arr[i] = ( m - bt) / bt
                    nn[i] = n
            # print(nn)
            end_time = time.time()
            print(f"Time taken for this configuration : {end_time - start_time} seconds")
            results_dict_SLS[(ld, step)] = (err_SLS_arr.mean(), nn.mean(), err_SLS_arr.std(), True) if nn.mean() < (0.90*itermax) else (err_SLS_arr.mean(), nn.mean(), err_SLS_arr.std(), False)
            results_dict_SLS_std[(ld, step)] = err_SLS_arr.std()
            print('standard deviation of SLS methof : ', err_SLS_arr.std())
            print('average error of SLS method : ', err_SLS_arr.mean())
            print('averge niter : ', nn.mean())
    # Store the results into a file
    if bigsize:
        compare = 'svd'
    else:
        compare = 'bt'
    if save:
        if stochastic:
            ft.store_dict_in_file_best_params(results_dict_SLS, f'results/{n1}x{n2}_s_{s}_SLS_stoch_{size}.txt')
        else:
            ft.store_dict_in_file_best_params(results_dict_SLS, f'results/{n1}x{n2}_s_{s}_SLS_{size}.txt')

    ### Visualization
    if stochastic:
        ft.plot_dict(results_dict_SLS, n1, n2, 'SLS Stochastic',  save = save, filename = 'stoch')
    else: 
        ft.plot_dict(results_dict_SLS, n1, n2, 'SLS',  save = save, filename = 'not stoch')
    return None 


def test_find_optimal_params_SLS_real_matrix(X, Lambdas=[1.0], steps=[1e-1], n1 = 20, n2 = 50, delta = 10, s = 10, 
                                                n_test = 10, itermax = 2000, stochastic = True, save = False, bigsize= True, 
                                                size = 'large', tol = 1e-4, X_list = False, filename = ''): # L_1O vs SLS 
    ld_max = Lambdas[-1]
    ld_min = Lambdas[0]
    n_ld = len(Lambdas)
    step_min = steps[0]
    step_max = steps[-1]
    n_steps = len(steps)
    if X_list:
        n1 = np.shape(X[0])[0]
        n2 = np.shape(X[0])[1]
    else:
        n1 = np.shape(X)[0]
        n2 = np.shape(X)[1]
    results_dict_SLS = {}
    results_dict_SLS_std = {}
    j = 0
    if X_list:
        pass
    else: 
        X_ = X
    for ld in Lambdas:  
        for step in steps:
            nn = []
            j += 1
            # print(j)
            err_SLS_arr = np.zeros(n_test)
            nn = np.zeros(n_test)
            start_time = time.time()
            print('ld : ', ld, 'step : ', step)
            for i in range(n_test):
                if X_list:
                    X_ = X[i]
                # print('test ', i)
                print('test ', i)
                (indexes, m, n, t) = SLS( X_, n1, n2, s, ld, step, itermax, display=False, tol = tol, stochastic=stochastic, info = True)
                err_SLS_arr[i] = Approximation_factor(X_, s, indexes)
                nn[i] = n
                print(n)
                print('error : ', err_SLS_arr[i])
            # print(nn)
            end_time = time.time()
            print(f"Time taken for this configuration : {end_time - start_time} seconds")
            results_dict_SLS[(ld, step)] = (err_SLS_arr.mean(), nn.mean(), err_SLS_arr.std(), True) if nn.mean() < (0.90*itermax) else (err_SLS_arr.mean(), nn.mean(), err_SLS_arr.std(), False)
            results_dict_SLS_std[(ld, step)] = err_SLS_arr.std()
            print('standard deviation of SLS methof : ', err_SLS_arr.std())
            print('average error of SLS method : ', err_SLS_arr.mean())
            print('averge niter : ', nn.mean())
    # Store the results into a file
    if bigsize:
        compare = 'svd'
    else:
        compare = 'bt'
    if save:
        if stochastic:
            ft.store_dict_in_file_best_params(results_dict_SLS, f'results/{n1}x{n2}_s_{s}_SLS_stoch_'+ filename +'.txt')
        else:
            ft.store_dict_in_file_best_params(results_dict_SLS, f'results/{n1}x{n2}_s_{s}_SLS_'+ filename +'.txt')

    ### Visualization
    if stochastic:
        ft.plot_dict(results_dict_SLS, n1, n2, 'SLS Stochastic',  save = save, filename = 'stoch')
    else: 
        ft.plot_dict(results_dict_SLS, n1, n2, 'SLS',  save = save, filename = filename)
    return None 
#here 
# Lambda: 6.309573444801933, step: 0.001584893192461114
#  Lambda: 2.51188643150958, step: 0.000630957344480193,
# test_find_optimal_params_SLS(np.array([2.5, 6.3]), np.array([0.000630957344480193, 0.001584893192461114]))


#used
def test_find_optimal_params_L_1O(Lambdas, steps, n1 = 20, n2 = 50, s = 10, n_test = 10, itermax = 2000, filename = '_',
                                     save = False, bigsize= True, size = 'large', tol = 1e-3, smart_start = False): 
    results_dict_L_1O = {}
    results_dict_L_1O_std = {}
    j = 0
    
    for ld in Lambdas:  
        for step in steps:
            nn = np.zeros(n_test)
            j += 1
            print(j)
            print('ld : ', ld, 'step : ', step)
            err_L_1O_arr = np.zeros(n_test)
            start_time = time.time()
            if bigsize:
                for i in range(n_test):
                    X = ft.matrix_from_file(i, n1, n2)
                    svd_error = CSSP_approximation_svd(X, n1, n2, s)[1]
                    # print('test ', i)
                    (indexes, error, n) = L_1O( X, n1, n2, s, ld, step, itermax, display=False, tol = tol, info = True)
                    err_L_1O_arr[i] = error/svd_error
                    nn[i] = n
            else:
                for i in range(n_test):
                    X = ft.matrix_from_file(i, n1, n2)
                    (bt_ind, bt) = brute_force(X, n1, n2, s)
                    # (indexes, m, n) = SLS( X, n1, n2, s, ld, step, itermax, display=False, tol=tol, info = True)
                    # err_L_1O_arr[i] = ( m - bt) / bt
                    # nn[i] = n
            # print(nn)
            print(nn)
            end_time = time.time()
            print(f"Time taken for this configuration : {end_time - start_time} seconds")
            results_dict_L_1O[(ld, step)] = (err_L_1O_arr.mean(), nn.mean(), err_L_1O_arr.std(), True) if nn.mean() < (0.95*itermax) else (err_L_1O_arr.mean(), nn.mean(), err_L_1O_arr.std(), False) # to verify 
            results_dict_L_1O_std[(ld, step)] = err_L_1O_arr.std()
            print('standard deviation of L_1O methof : ', err_L_1O_arr.std())
            print('average error of L_1O method : ', err_L_1O_arr.mean())
    # Store the results into a file
    if bigsize:
        compare = 'svd'
    else:
        compare = 'bt'
    if save:
        ft.store_dict_in_file_best_params(results_dict_L_1O, f'results/{n1}x{n2}_s_{s}_L_1O_{size}_{filename}.txt')
    ### Visualization
    ft.plot_dict(results_dict_L_1O, n1, n2, 'L_1O', save = save, filename = filename)
    return None

#used
def test_find_optimal_params_L_1O_real_matrix(X, Lambdas, steps, s = 10, itermax = 2000, filename = '_unknown',
                                                  save = False, tol = 1e-3, display = False, n_trials = 1): 
    """
    store in a file the dictionary of the results of the hyperparameters search"""
    (n1, n2) = np.shape(X)
    results_dict_L_1O = {}
    j = 0
    start_time = time.time()
    svd_error = CSSP_approximation_svd(X, n1, n2, s)[1]
    print('time taken by the svd : ', time.time() - start_time)
    for ld in Lambdas:  
        for step in steps:
            # results_dict_L_1O[(ld, step)] 
            nn=0
            err_L_1O = 0
            for i in range(n_trials):
                start_time = time.time()
                    # print('test ', i)
                (indexes, error, n) = L_1O( X, n1, n2, s, ld, step, itermax, display=display, tol = 1e-10)
                print(n)
                nn += n
                err_L_1O += error/svd_error
                end_time = time.time()
                print('config ld, step : ', (ld, step))
                print(f"Time taken : {end_time - start_time} seconds")
                 # to verify 
                print(' error of L_1O method : ',err_L_1O)
                print('number of iterations : ', n)
            err_L_1O /= n_trials
            nn /= n_trials
            results_dict_L_1O[(ld, step)] = (err_L_1O, nn, 0, True) if n < (itermax)*1.1 else (err_L_1O, nn, 0, False)
            # results_dict_L_1O[(ld, step)] = tuple([x/n_trials for x in results_dict_L_1O[(ld, step)]])
    # Store the results into a file
    compare = 'svd'
    if save:
        ft.store_dict_in_file_best_params(results_dict_L_1O, f'results/{n1}x{n2}_s_{s}_L_1O_{filename}.txt')
    ### Visualization
    ft.plot_dict(results_dict_L_1O, n1, n2, r'L_1 orthogonal regularization', save = save, filename = filename, s=s)
    return None


def influence_of_M_SLS(X, n1, n2, s, ld, step, n_test = 12, itermax = 2000, real_matrix = False, filename = ''): #
    """
    Test the influence of M on the performance of the SLS algorith
    """
    M = 10
    save = True
    M = np.arange(1,M,2)
    print(M)
    results_mat = np.zeros((len(M), n_test))
    error_svd = np.zeros(n_test)
    if real_matrix == False:
        for j in range(n_test):
            X = ft.matrix_from_file(j, n1, n2)
            error_svd[j] = CSSP_approximation_svd(X, n1, n2, s)[1]
    else :
        n1, n2 = np.shape(X)
        error_svd = np.ones(n_test)*CSSP_approximation_svd(X, n1, n2, s)[1]
    for i,M_ in enumerate(M):
        print('M : ', M_)
        for j in range(n_test):
            if real_matrix == False:
                X = ft.matrix_from_file(j, n1, n2)
            error = SLS(X, n1, n2, s, ld, step, itermax, delta = 1, M = M_, display=False, tol = 1e-3, info = True)
            # print('nite : ', error[2])
            results_mat[i, j] = error[1]/error_svd[j]
            # print('M : ', M_, 'error : ', error[1]/error_svd[j])
    np.savetxt('results/M_'+filename+'.txt', results_mat)
    return None

def plot_influence_of_M_SLS(filename = '', save = False):
    mat = np.loadtxt('results/M_'+filename+'.txt')
    M = 10
    M = np.arange(1,M,2)
    print(mat.mean(axis = 1))
    add = np.array([1.305154, 1.29416, 1.28435, 1.2793, 1.279])
    y = np.concatenate((mat.mean(axis = 1)[:3], add))
    #decide the size of the fig 
    plt.figure(figsize=(10, 4))
    plt.plot(M, add, color = 'black')
    # plt.errorbar(M, mat.mean(axis = 1), mat.std(axis=1), color = 'black')
    plt.xlabel('M')
    plt.ylabel('Approximation factor')
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig('results/M_'+filename+'.eps')
    else:
        plt.show()


def L_1O_SLS_comparison(n1 = 20, n2 = 50, s = 10, itermax = 12, n_test = 10, save = False, filename = '_'):
    ld_SLS_stoch = 1e0
    step_SLS_stoch = 1e-2
    # ld_SLS_not_stoch = 1e1
    # step_SLS_not_stoch = 1e-4
    # ld_L_1O_smart = 1e-8
    # step_L_1O_smart = 1e5
    ld_L_1O = 1e-8
    step_L_1O = 1e5
    tol = 1e-3
    s = np.arange(1,15,1)
    err_SLS = np.zeros(len(s))
    err_L_1O_smart = np.zeros(len(s))
    err_L_1O = np.zeros(len(s)) 
    err_SLS_not_stoch = np.zeros(len(s))
    err_uniform = np.zeros(len(s))
    err_Boutsidis = np.zeros(len(s))
    n_test_fast = 20
    svd_error = np.zeros(n_test)
    for (i,s_) in zip(range(len(s)),s):
        # print('s ', i)
        timer = time.time()
        for j in range(n_test):
            X = ft.matrix_from_file(j, n1, n2)
            svd_error[j] = CSSP_approximation_svd(X, n1, n2, s_)[1]
            (indexes, m, n) = L_1O( X, n1, n2, s_, ld_L_1O, step_L_1O, itermax, display=False, tol = tol, info = True)
            err_L_1O[i] += m/svd_error[j]
            # (indexes, m, n) = L_1O( X, n1, n2, s_, ld_L_1O_smart, step_L_1O_smart, itermax, display=False, tol = tol, smart_start=True)
            # err_L_1O_smart[i] += m/svd_error[j]
            (indexes, m, n, t) = SLS( X, n1, n2, s_, ld_SLS_stoch, step_SLS_stoch, itermax, display=False, tol = tol, stochastic = True, info = True)
            err_SLS[i] += m/svd_error[j]
            # (indexes, m, n, t) = SLS( X, n1, n2, s_, ld_SLS_not_stoch, step_SLS_not_stoch, itermax, display=False, tol = tol, stochastic = True)
            # err_SLS_not_stoch[i] += m/svd_error[j]
            (C_Boutsidis, m) = Boutsidis_Mahoney_Drineas(X, s_, c = 10*s_,info = True)
            err_Boutsidis[i] += m/svd_error[j]
            # print('Boutsidis done')
            # err_Boutsidis[i] += Approximation_factor(X, s_, C_Boutsidis)
            (C_uniform, m) = Uniform_sampling(X, s_, info = True)
            err_uniform[i] += Approximation_factor_C(X, s_, C_uniform)
        # print(f"Time taken for this configuration : {time.time() - timer} seconds")
    err_L_1O /= n_test
    err_L_1O_smart /= n_test
    err_SLS /= n_test
    err_Boutsidis /= n_test
    err_uniform /= n_test
    s = (s/n1)*100
    ##
    #save the results in a file .txt
    if save: 
        with open('results/comparison_perf_random_mat.txt', 'w') as f:
            f.write('s : ')
            f.write(np.array2string(np.array(s), max_line_width=np.inf))
            f.write('\n')
            f.write('err_L_1O : ')
            f.write(np.array2string(np.array(err_L_1O), max_line_width=np.inf))
            f.write('\n')
            f.write('err_SLS : ')
            f.write(np.array2string(np.array(err_SLS), max_line_width=np.inf))
            f.write('\n')
            f.write('err_Boutsidis : ')
            f.write(np.array2string(np.array(err_Boutsidis), max_line_width=np.inf))
            f.write('\n')
            f.write('err_uniform : ')
            f.write(np.array2string(np.array(err_uniform), max_line_width=np.inf))
            f.write('\n')
    return None


def L_1O_SLS_comparison_MNIST(X, itermax = 12, n_test = 10, save = False, filename = '_MNIST_'):
    n1, n2 = np.shape(X)
    ld_SLS_stoch = 1e1
    step_SLS_stoch = 1e-3
    # ld_SLS_not_stoch = 1e1
    # step_SLS_not_stoch = 1e-4
    ld_L_1O = 0.1 #ok
    step_L_1O = 0.001 #ok
    tol = 1e-3
    s = np.linspace(n1*0.05,n1*0.7,11)
    err_SLS = np.zeros(len(s))
    err_L_1O_smart = np.zeros(len(s))
    err_L_1O = np.zeros(len(s)) 
    err_leverage_scores = np.zeros(len(s))
    err_SLS_not_stoch = np.zeros(len(s))
    err_uniform = np.zeros(len(s))
    err_Boutsidis = np.zeros(len(s))
    n_test_fast = 20
    svd_error = np.zeros(n_test)
    for (i,s_) in zip(range(len(s)),s):
        print('test ', i)
        s_ = int(s_)
        print('s : ', s_)
        # timer_big = time.time()
        for j in range(n_test):
            # X = ft.matrix_from_file(j, n1, n2)
            timer = time.time()
            svd_error[j] = CSSP_approximation_svd(X, n1, n2, s_)[1]
            # print('time taken by the svd : ', time.time() - timer)
            # timer = time.time()
            (indexes, m, n) = L_1O( X, n1, n2, s_, ld_L_1O, step_L_1O, itermax, display=False, tol = tol, info = True)
            err_L_1O[i] += m/svd_error[j]
            # print('time taken by L_1O : ', time.time() - timer)
            # timer = time.time()
            # (indexes, m, n) = L_1O( X, n1, n2, s_, ld_L_1O_smart, step_L_1O_smart, itermax, display=False, tol = tol, smart_start=True)
            # err_L_1O_smart[i] += m/svd_error[j]
            # print('time taken by L_1O smart : ', time.time() - timer)
            # timer = time.time()
            (indexes, m, n, t) = SLS( X, n1, n2, s_, ld_SLS_stoch, step_SLS_stoch, itermax, display=False, tol = tol, stochastic = True, info = True)
            err_SLS[i] += m/svd_error[j]
            # print('time taken by SLS stoch : ', time.time() - timer)
            # timer = time.time()
            # (indexes, m, n, t) = SLS( X, n1, n2, s_, ld_SLS_not_stoch, step_SLS_not_stoch, itermax, display=False, tol = tol, stochastic = True)
            # err_SLS_not_stoch[i] += m/svd_error[j]
            # print('time taken by SLS not stoch : ', time.time() - timer)
            # timer = time.time()
            (C_Boutsidis, m) = Boutsidis_Mahoney_Drineas(X, s_, c = 1, info = True)
            err_Boutsidis[i] += m/svd_error[j]
            # print('Boutsidis done')
            # err_Boutsidis[i] += Approximation_factor(X, s_, C_Boutsidis)
            # timer = time.time()
            (C_uniform, m) = Uniform_sampling(X, s_, info = True)
            err_uniform[i] += m/svd_error[j]
            # print('time taken by leverage scores : ', time.time() - timer)
        # print(f"Time taken for this configuration : {time.time() - timer_big} seconds")
    err_L_1O /= n_test
    err_L_1O_smart /= n_test
    err_SLS /= n_test
    err_Boutsidis /= n_test
    err_uniform /= n_test
    s = (s/n1)*100
    ##
    #save the results in a file .txt
    if save:
        with open('results/comparison_perf_'+filename+'_mat.txt', 'w') as f:
            f.write('s : ')
            f.write(np.array2string(np.array(s), max_line_width=np.inf))
            f.write('\n')
            f.write('err_L_1O : ')
            f.write(np.array2string(np.array(err_L_1O), max_line_width=np.inf))
            f.write('\n')
            f.write('err_SLS : ')
            f.write(np.array2string(np.array(err_SLS), max_line_width=np.inf))
            f.write('\n')
            f.write('err_Boutsidis : ')
            f.write(np.array2string(np.array(err_Boutsidis), max_line_width=np.inf))
            f.write('\n')
            f.write('err_uniform : ')
            f.write(np.array2string(np.array(err_uniform), max_line_width=np.inf))
            f.write('\n')
    return None


def L_1O_SLS_comparison_arrithmia(X, itermax = 12, n_test = 10, save = False, filename = '_arrithmia_'):
    n1, n2 = np.shape(X)
    # ld_SLS_stoch = 1e-5
    # step_SLS_stoch = 1e-1
    ld_SLS_stoch = 0.01
    step_SLS_stoch =  0.001
    ld_L_1O = 1 #ok
    step_L_1O = 0.001 #ok
    tol = 1e-3
    s = np.linspace(n1*0.05,n1*0.7,11)
    err_SLS = np.zeros(len(s))
    err_L_1O = np.zeros(len(s)) 
    err_Boutsidis = np.zeros(len(s))
    err_uniform = np.zeros(len(s))
    n_test_fast = 20
    svd_error = np.zeros(n_test)
    for (i,s_) in zip(range(len(s)),s):
        print('test ', i)
        s_ = int(s_)
        print('s : ', s_)
        timer_big = time.time()
        for j in range(n_test):
            # X = ft.matrix_from_file(j, n1, n2)
            # timer = time.time()
            svd_error[j] = CSSP_approximation_svd(X, n1, n2, s_)[1]
            # print('time taken by the svd : ', time.time() - timer)
            # timer = time.time()
            (indexes, m, n) = L_1O( X, n1, n2, s_, ld_L_1O, step_L_1O, itermax, display=False, tol = tol, info = True)
            err_L_1O[i] += m/svd_error[j]
            (indexes, m, n, t) = SLS( X, n1, n2, s_, ld_SLS_stoch, step_SLS_stoch, itermax, display=False, tol = tol, stochastic = True, info = True)
            err_SLS[i] += m/svd_error[j]
            (C_Boutsidis,m) = Boutsidis_Mahoney_Drineas(X, s_, c = 1, info = True)
            # print('Boutsidis done')
            err_Boutsidis[i] += m/svd_error[j]
            # timer = time.time()
            (C_uniform, m) = Uniform_sampling(X, s_, info = True)
            err_uniform[i] += m/svd_error[j]
            # print('time taken by leverage scores : ', time.time() - timer)
        print(f"Time taken for this configuration : {time.time() - timer_big} seconds")
    err_L_1O /= n_test
    err_L_1O_smart /= n_test
    err_SLS /= n_test
    err_Boutsidis /= n_test
    # err_leverage_scores /= n_test
    err_uniform /= n_test
    s = (s/n1)*100
    ##
    #save the results in a file .txt
    if save:
        with open('results/comparison_perf_'+filename+'_mat.txt', 'w') as f:
            f.write('s : ')
            f.write(np.array2string(s, max_line_width=np.inf))
            f.write('\n')
            f.write('err_L_1O : ')
            f.write(np.array2string(err_L_1O, max_line_width=np.inf))
            f.write('\n')
            f.write('err_SLS : ')
            f.write(np.array2string(err_SLS, max_line_width=np.inf))
            f.write('\n')
            f.write('err_uniform : ')
            f.write(np.array2string(err_uniform, max_line_width=np.inf))
            f.write('\n')
            f.write('err_Boutsidis : ')
            f.write(np.array2string(err_Boutsidis, max_line_width=np.inf))
            f.write('\n')
    return None

def test_comparison_plot(filename1 = '_', uniform = True, save = False, random = False):
    # read the file comparison_perf_MNIST_mat.txt
    with open('results/' + filename1 + '.txt', 'r') as f:
        lines = f.readlines()
        print(lines)
        s = np.array(lines[0].split('[')[1].split(']')[0].split()).astype(float)
        err_L_1O = np.array(lines[1].split('[')[1].split(']')[0].split()).astype(float)
        # err_L_1O_smart = np.array(lines[2].split('[')[1].split(']')[0].split()).astype(float)
        err_SLS = np.array(lines[2].split('[')[1].split(']')[0].split()).astype(float)
        # err_SLS_not_stoch = np.array(lines[4].split('[')[1].split(']')[0].split()).astype(float)
        err_uniform = np.array(lines[3].split('[')[1].split(']')[0].split()).astype(float)
        err_boutsidis = np.array(lines[4].split('[')[1].split(']')[0].split()).astype(float)
    
    plt.figure(figsize=(10, 3))
    plt.plot(s, err_L_1O, label = r'$L_1$ Orthogonal Regularization (LOR)', color = 'blue')
    # plt.plot(s, err_L_1O_smart, label = r'$L_1$OR with PCA starint point')
    plt.plot(s, err_SLS, label = 'Stochatic Landmark Selection (SLS)', color = 'green')
    # plt.plot(s, err_SLS_not_stoch, label = 'Landmark Selection')
    plt.plot(s, err_boutsidis, label = 'Double-Phase Algorithm', color = 'red')
    # plt.plot(s, err_leverage_scores, label = 'Leverage Scores')
    if uniform:
        plt.plot(s, err_uniform, label = 'Uniform Sampling', color ='black')
    # plt.title('Comparison of the different methods')
    plt.xlim(s[0], s[-1])
    plt.xlabel('s/m %')
    # plt.yscale('log')
    plt.ylabel('Approximation Factor')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    if save:
        plt.savefig('results/' + filename1 + '.eps')
    else:
        plt.show()

def L_1O_real_perf(X, itermax = 1000, save = False, filename = '_', tol = 1e-3):
    ld_L_1O_smart = 1e-3
    step_L_1O_smart = 1e-1
    ld_L_1O = 1e-3
    step_L_1O = 1e-1
    (n1, n2) = np.shape(X)
    s = np.round(np.linspace(0,n2*0.7,11))
    print('s : ', s)
    err_L_1O_smart = np.zeros(len(s))
    err_L_1O = np.zeros(len(s)) 
    for (i,s_) in zip(range(len(s)),s):
            if s_==0:
                err_L_1O[i] = 1
                err_L_1O_smart[i] = 1
                continue
            s_ = int(s_)
            print(('s : ', s_))
            error_C_svd = CSSP_approximation_svd(X, n1, n2, s_)[1]
            # print('shape of C_svd : ', np.shape(C_svd))
            # error_C_svd = function_objectif_C(X, C_svd)
            timer = time.time()
            print('s : ', s_)
            (indexes, error, n) = L_1O( X, n1, n2, s_, ld_L_1O, step_L_1O, itermax, display=True, tol = tol, info = True)
            err_L_1O[i] = error/error_C_svd
            print('first method done')
            (indexes, error, n) = L_1O( X, n1, n2, s_, ld_L_1O_smart, step_L_1O_smart, itermax, display=True, tol = tol, smart_start=True, info = True)
            err_L_1O_smart[i] = error/error_C_svd
            print('second method done')
            print(f"Time taken for this configuration : {time.time() - timer} seconds")
            print('err_L_1O : ', err_L_1O)
            print('err_L_1O_smart : ', err_L_1O_smart)
    s = (s/np.max(s))*100
    #save the results in a text file 
    if save:
        pass
    print('err_L_1O : ', err_L_1O)
    print('err_L_1O_smart : ', err_L_1O_smart)
    plt.plot(s, err_L_1O, label = 'L_1O')
    plt.plot(s, err_L_1O_smart, label = 'L_1O Smart')
    plt.xlabel('n/s %')
    plt.ylabel('Approximation factor')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig('results/L_1O_mnist_perf'+filename+'.svg')
    else:
        plt.show()


def test_Uniform_Sampling(images_std, s=100):
    split = np.linspace(0.10,1.0,10)
    (m,n) = np.shape(images_std)
    print('m : ', m, 'n : ', n)
    s = split*n
    print(s)
    (U, Sigma,W_svdT) = svd(images_std)
    W_svd = W_svdT.T
    err = []
    for s_ in s:
        s_ = int(s_)
        W_svds = W_svd[:,:s_]
        # print('shape of W_svd : ', np.shape(W_svd))
        error_svd = function_objectif_W(images_std, W_svd)
        C_uniform = Uniform_sampling(images_std, s_)
        error_uniform = function_objectif_C(images_std, C_uniform)
        err.append(error_uniform/error_svd)
    plt.plot(split*100, err)
    plt.xlabel('s')
    plt.ylabel('Approximation factor')
    plt.grid()
    plt.show()
    return None



# def artificial_big_matrix_test_comparison(n_test, n1, n2, s, params, itermax,tol=1e-5, methods = ['L_1O', 'SLS']):
#     """
#     Test the L_1O function on a big matrix. Create a file that 
#     Parameters:
#         n_test (int): The number of test to perform.
#         n1 (int): The number of lines.
#         n2 (int): The number of columns.
#         s (int): The number of non-null columns.
#         noise (float): The standard deviation of the noise we add to every element of the matrix
#     Returns:
#         """
#     # ft.create_artificial_matrix(n_test, n1, n2, s)
#     n_sigma = 10
#     sigma = np.linspace(0.1, 1.0, n_sigma)
#     print('this is sigma : ', sigma)
#     # ft.create_matrix_file(n_test, 5, 8)
#     # Lambdas = np.linspace(ld_min, ld_max, n_ld)
#     # steps = np.linspace(0.001,0.5, n_steps)
#     # iter1 = 200
#     # print(i)
#     results_dict = {}
#     count = 0
#     for method in methods:
#         results_dict[method] = {}
#         ld = params[method]['ld']
#         step = params[method]['step']
#         for sig in sigma: 
#             # ft.create_artificial_matrix(n_test, n1, n2, s, sig)
#             print(sig)
#             for i in range(n_test):
#                 X = ft.read_artificial_matrix_from_file(i, n1, n2, s, sig)
#                 if (method == 'L_1O'):
#                     (relevant_col_indices, err, niter) = L_1O(X, n1, n2, s, ld, step, itermax, display = True, tol = 1e-5)
#                     print('the number of iterations for L_1O is : ', niter)
#                     print('error of L_1O : ', err)
#                 if(method == 'SLS'):
#                     (relevant_col_indices, err,  niter) = SLS( X, n1, n2, s, ld, step, itermax, tol=1e-5)
#                     print('the number of iterations for SLS is : ', niter)    
#                     print('error of SLS : ', err)
#                 count += np.sum((relevant_col_indices < n2 - s))
#             results_dict[method][sig] = (1-(count/(n_test*s)))*100
#             count = 0
#         print(results_dict[method])
#         print(list(results_dict[method].values()))
#         a = list(results_dict[method].values())
#         plt.plot(sigma, list(results_dict[method].values()), label = method)
#     plt.xlabel('Sigma')
#     plt.xlim(0.0, 1.0)
#     plt.ylim(0.0, 1.0)
#     plt.ylabel(r'% of relevant columns found')
#     plt.title(r'Methods performances on $X \in \mathbb{R}^{20,50}$ with $s = 5$')
#     plt.legend()
#     plt.grid()
#     plt.savefig(f'results/n1_{n1}_n2_{n2}_s_{s}_sigma.svg')
#     plt.close()
#     return None



def std_data(filename, file_std = 'data_std.data'):
    """
    read the data in the file , replace the nan values by 0 and standardize the data
    store it in a file named file_std.data
    """
    data = pd.read_csv(filename, header=None)
    print(data)
    data = data.fillna(0)
    data_std = (data - data.mean())/data.std()
    data_std.to_csv(file_std, header = False, index = False)
    return None




def test_speed_1(r = 9):
    # m = np.geomspace(1, 1024, 9)
    m = np.geomspace(4, 2**(r+1), r)
    print(m)
    # m = np.array([2**(7+1)])
    n = m*2
    s = m/2
    n_test = 9
    time_L_1O = np.zeros(r)  
    iter_L_1O = np.zeros(r)
    time_SLS = np.zeros(r)
    iter_SLS = np.zeros(r)
    for (i, m_, n_, s_) in zip(range(r),m, n, s):
        if i > 4:
            n_test = 2
        if i > 5:
            n_test = 1
        ft.create_matrix_file(n_test, int(m_), int(n_))
        for j in range(n_test):
            X = ft.matrix_from_file(j, int(m_), int(n_))
            timer = time.time()
            results_L_1O = L_1O(X, int(m_), int(n_), int(s_), 1e-8, 1e5, 400, display=False, tol = 1e-5,info = True)
            time_L_1O[i] += (time.time() - timer)/(results_L_1O[2]*n_test)
            iter_L_1O[i] += results_L_1O[2]/n_test   
            timer = time.time()
            results_SLS = SLS(X, int(m_), int(n_), int(s_), 100, 1e-5, 400, display=False, tol = 1e-10, stochastic=True, info = True)
            time_SLS[i] += (time.time() - timer)/(results_SLS[2]*n_test)
            iter_SLS[i] += results_SLS[2]/n_test
        
        print('r : ', i)
        print('time L_1O : ', time_L_1O)
        print('n iter L_1O : ', results_L_1O[2])
        print('time SLS : ', time_SLS)
        print('n iter SLS : ', results_SLS[2])
        # print(Approximation_factor_C(X, int(s_), X[:,results_L_1O[0]]))
        # print(Approximation_factor_C(X, int(s_), Uniform_sampling(X, int(s_))[0]))
        # print('error L_1O : ', results_L_1O[1])
    #write the results in a file .txt
    with open('results/speed1_comparison_2.txt', 'w') as f:
        f.write('m : ')
        f.write(np.array2string(np.array(m), max_line_width=np.inf))
        f.write('\n')
        f.write('time_L_1O : ')
        f.write(np.array2string(np.array(time_L_1O), max_line_width=np.inf))
        f.write('\n')
        f.write('iter_L_1O : ')
        f.write(np.array2string(np.array(iter_L_1O), max_line_width=np.inf))
        f.write('\n')
        f.write('time_SLS : ')
        f.write(np.array2string(np.array(time_SLS), max_line_width=np.inf))
        f.write('\n')
        f.write('iter_SLS : ')
        f.write(np.array2string(np.array(iter_SLS), max_line_width=np.inf))
        f.write('\n')
        f.close()


def plot_speed_1(filename = '_', save = False):
    fig, ax = plt.subplots(figsize=(8, 4))
    with open('results/speed1_comparison_2.txt', 'r') as f:
        lines = f.readlines()
        # s = np.array(lines[0].split('[')[1].split(']')[0].split()).astype(float)
        m = np.array(lines[0].split('[')[1].split(']')[0].split()).astype(float)
        print(m)
        n = m*2
        time_L_1O = np.array(lines[1].split('[')[1].split(']')[0].split()).astype(float)
        iter_L_1O = np.array(lines[2].split('[')[1].split(']')[0].split()).astype(float)
        time_SLS = np.array(lines[3].split('[')[1].split(']')[0].split()).astype(float)
        iter_SLS = np.array(lines[4].split('[')[1].split(']')[0].split()).astype(float)
        two_n = np.array([time_L_1O[0]*(i/4)**2 for i in m])
        three_n = np.array([time_L_1O[0]*(i/4)**3 for i in m])
    ax.plot(m, time_L_1O, label = r'$L_1$ Orthogonal Regularization', color = 'blue')#, base = 2)
    ax.plot(m, time_SLS/5, label = 'Stochastic Landmark Selection', color = 'green')#, base = 2)
    ax.plot(m, two_n, label = r'$\mathcal{O}(n^2 )$', color = 'black', linestyle='--')
    ax.plot(m, three_n, label = r'$\mathcal{O}(n^3 )$', color = 'black', linestyle=':')
    ax.set_xlabel('n')
    #xscale = log
    ax.set_xscale('log')
    ax.set_yscale('log')
    #set xlim
    ax.set_xlim(4, 2**12)
    #set ylim
    ax.set_ylim(time_L_1O[0], time_SLS[-1])
    ax.set_ylabel('Time [s] / iteration')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    if save:
        plt.savefig('results/definitif/speed1_comparison_by'+filename+'.eps')
    else:
        plt.show()
    return None
    

# s = np.arange(1, 2

import mnist
from mnist import MNIST
def test_speed_2(X, Xlist = True, filename=''):
    mndata = MNIST('samples')

    images, labels = mndata.load_training()

    #shuflle the images using a seed
    np.random.seed(42)
    np.random.shuffle(images)


    images = np.array(images)
    images_std = (images - images.mean())/images.std()


    X = np.array(images_std[:700])
    # X = np.random.rand(20, 50)
    (m, n) = np.shape(X)
    print('shape of X : ', np.shape(X))
    s = np.arange(0.05,1,0.1)*m
    print(s)
    n_test = 5
    time_L_1O = np.zeros(len(s))
    time_SLS = np.zeros(len(s))
    for (j,s_) in zip(range(len(s)),s): 
        for i in range(n_test):
            timer = time.time()
            results_L_1O = L_1O(X, m, n, int(s_), 1e-8, 1e5, 100, display=False, tol = 1e-3, info = True)
            time_L_1O[j] += (time.time() - timer)/(results_L_1O[2])
            print('L_1O done')
            timer = time.time()
            results_SLS = SLS(X, m, n, int(s_),  1, 1e-3, 100, display=False, tol = 1e-5, info = True)
            time_SLS[j] += (time.time() - timer)/(results_SLS[2])
        print('time SLS : ', time_SLS)
        print('time L_1O : ', time_L_1O)
        print('n iter L_1O : ', results_L_1O[2])
        print('n iter SLS : ', results_SLS[2])
        # print('error L_1O : ', results_L_1O[1])
    time_L_1O = time_L_1O/n_test
    time_SLS = time_SLS/n_test
    #store the time_L_1O and time_SLS in a file .txt
    with open('results/speed2_comparison'+filename+'.txt', 'w') as f:
        f.write('s : ')
        f.write(np.array2string(s, max_line_width=np.inf))
        f.write('\n')
        f.write('time_L_1O : ')
        f.write(np.array2string(time_L_1O, max_line_width=np.inf))
        f.write('\n')
        f.write('time_SLS : ')
        f.write(np.array2string(time_SLS, max_line_width=np.inf))
        f.write('\n')
        f.close()
    plt.plot(s, time_L_1O, label = r'$L_1$ Orthogonal Regularization',color = 'blue')
    plt.plot(s, time_SLS, label = 'Stochatic Landmark Selection',color = 'green')
    plt.legend()
    plt.xlabel('s') 
    plt.ylabel('Time (s)/iteration')
    plt.grid()
    plt.savefig('results/speed_2_test.svg')
    plt.show()
    plt.close()
    return None
    
def plot_speed_2(filename = '_', save = False):
    fig, ax = plt.subplots(figsize=(8, 4))
    with open('results/definitif/speed2.txt', 'r') as f:
        lines = f.readlines()
        s = np.array(lines[0].split('[')[1].split(']')[0].split()).astype(float)
        time_SLS = np.array(lines[1].split('[')[1].split(']')[0].split()).astype(float)
        time_L_1O = np.array(lines[2].split('[')[1].split(']')[0].split()).astype(float)
    s = s/784
    ax.plot(s, time_L_1O, label = r'$L_1$ Orthogonal Regularization', color = 'blue')#, base = 2)
    ax.plot(s, time_SLS, label = 'Stochastic Landmark Selection', color = 'green')#, base = 2)
    ax.set_xlabel('s/m %')
    #xscale = log
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    #set xlim
    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(time_L_1O[0], time_L_1O[-1])
    #set ylim
    # ax.set_ylim(time_L_1O[0], time_L_1O[-1])
    ax.set_ylabel('Time [s]/iteration')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    if save:
        plt.savefig('results/definitif/speed2_comparison'+filename+'.eps')
    else:
        plt.show()
    return None


# test_speed_2()

def test_speed_3(Xlist, s, itermax,mod,  ld_L_1O, step_L_1O, ld_SLS, step_SLS, filename = ''):
    (m, n) = np.shape(Xlist[0])
    # np.random.seed(42)
    # X = np.random.randn(m, n)
    n_test = len(Xlist)
    #to change
    time_L_1O = 0
    time_SLS = 0
    # mod = 10

    indices_L_1O = np.zeros(itermax//mod+1)
    Time_array_L_1O = np.zeros(itermax//mod+1)
    Error_array_L_1O = np.zeros((n_test, itermax//mod+1))
    n_array_L_1O = np.zeros(itermax//mod+1)
    Time_array_SLS = np.zeros(itermax//mod+1)
    Error_array_SLS = np.zeros((n_test, itermax//mod+1))
    n_array_SLS = np.zeros(itermax//mod+1)

    idxi_L_1O = itermax//mod+1
    idxi_SLS = itermax//mod+1
    for (i,X) in enumerate(Xlist):
        print('test number : ', i)
        # X = np.random.randn(m, n)
        error_svd = CSSP_approximation_svd(X, m, n, s)[1]
        print('svd done')
        results_L_1O = L_1O_part1_speed(X, m, n, s, ld_L_1O, step_L_1O, itermax ,mod, display=False, tol = 1e-3)
        print('L_1O done')
        results_SLS = SLS_speed(X, m, n, s, ld_SLS, step_SLS, itermax, mod, display=False, tol = 1e-3)
        print('SLS done')
        # print('results_SLS : ', results_SLS)
        # print('Error_array shape: ', np.shape(results_L_1O[2]))
        # print('Error_array shape: ', np.shape(Error_array))
        Time_array_L_1O += results_L_1O[0]
        n_array_L_1O += results_L_1O[1]
        Error_array_L_1O[i] = results_L_1O[2]/error_svd
        print('shape of Time_array_SLS : ', np.shape(Time_array_SLS))
        print('shape of results_SLS[0] : ', np.shape(results_SLS[0]))
        Time_array_SLS += results_SLS[0]
        n_array_SLS += results_SLS[1]
        Error_array_SLS[i] = results_SLS[2]/error_svd
        # print('results_L_1O[2] : ', results_L_1O[2])
        if (np.where(results_L_1O[2] == 0)[0][0]<idxi_L_1O):
            idxi_L_1O = (np.where(results_L_1O[2] == 0))[0][0]
        
        if (np.where(results_SLS[2] == 0)[0][0]<idxi_SLS):
            idxi_SLS = (np.where(results_SLS[2] == 0))[0][0]

        print('idxi : ', idxi_L_1O)
        print('idxi : ', idxi_SLS)
        # indices_L_1O[:idxi] += np.ones(idxi) #ok

    # print(indices_L_1O[:2])
    
    #find the first 0 index in Time_array
    # idx = np.where(Time_array == 0)[0][0]
    # print('idx : ', idx)
    Time_array_L_1O = Time_array_L_1O[:idxi_L_1O]/n_test
    n_array_L_1O = n_array_L_1O[:idxi_L_1O]/n_test
    Error_array_mean_L_1O = Error_array_L_1O[:,:idxi_L_1O].mean(axis = 0)
    yerr_L_1O = Error_array_L_1O[:,:idxi_L_1O].std(axis = 0)

    Time_array_SLS = Time_array_SLS[:idxi_SLS]/n_test
    n_array_SLS = n_array_SLS[:idxi_SLS]/n_test
    Error_array_mean_SLS = Error_array_SLS[:,:idxi_SLS].mean(axis = 0)
    yerr_SLS = Error_array_SLS[:,:idxi_SLS].std(axis = 0)

    print(len(Error_array_mean_L_1O))
    # Time_array_L_1O = np.linspace(1, 100, 100)
    #save Timme_array, n_array, Error_array_mean and Error_array_std in a file .txt
    with open('results/speed3_comparison_L_1O_'+filename+'.txt', 'w') as f:
        f.write('Time_array_L_1O : ')
        f.write(np.array2string(Time_array_L_1O, max_line_width=np.inf))
        f.write('\n')
        f.write('n_array_L_1O : ')
        f.write(np.array2string(n_array_L_1O, max_line_width=np.inf))
        f.write('\n')
        f.write('Error_array_mean_L_1O : ')
        f.write(np.array2string(Error_array_mean_L_1O, max_line_width=np.inf))
        f.write('\n')
        f.write('Error_array_std_L_1O : ')
        f.write(np.array2string(yerr_L_1O, max_line_width=np.inf))
        f.write('\n')
        f.close()
        
    with open('results/speed3_comparison_SLS_'+filename+'.txt', 'w') as f:
        f.write('Time_array_SLS : ')
        f.write(np.array2string(Time_array_SLS, max_line_width=np.inf))
        f.write('\n')
        f.write('n_array_SLS : ')
        f.write(np.array2string(n_array_SLS, max_line_width=np.inf))
        f.write('\n')
        f.write('Error_array_mean_SLS : ')
        f.write(np.array2string(Error_array_mean_SLS, max_line_width=np.inf))
        f.write('\n')
        f.write('Error_array_std_SLS : ')
        f.write(np.array2string(yerr_SLS, max_line_width=np.inf))
        f.write('\n')
        f.close()
    return None 

    
def plot_speed_3(filename_L_1O = '_', filename_SLS = '_', dataset_name = 'noname', save = False,  ylim = False):
    with open('results/'+filename_L_1O+'.txt', 'r') as f:
        lines = f.readlines()
        Times_arr_L_1O = np.array(lines[0].split('[')[1].split(']')[0].split()).astype(float)
        n_arr_L_1O = np.array(lines[1].split('[')[1].split(']')[0].split()).astype(float)
        Error_arr_mean_L_1O = np.array(lines[2].split('[')[1].split(']')[0].split()).astype(float)
        Error_arr_std_L_1O = np.array(lines[3].split('[')[1].split(']')[0].split()).astype(float)
    with open('results/'+filename_SLS+'.txt', 'r') as f:
        lines = f.readlines()
        Times_arr_SLS = np.array(lines[0].split('[')[1].split(']')[0].split()).astype(float)
        n_arr_SLS = np.array(lines[1].split('[')[1].split(']')[0].split()).astype(float)
        Error_arr_mean_SLS = np.array(lines[2].split('[')[1].split(']')[0].split()).astype(float)
        Error_arr_std_SLS = np.array(lines[3].split('[')[1].split(']')[0].split()).astype(float)
    if ylim:
        n_lim = 364
        t_lim = Times_arr_SLS[n_lim]
    # Create the subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=False, sharey=True)
    # fig.suptitle('Execution Time Comparison', fontsize=16)

    # First plot (Time vs. Approximation factor)
    axs1_twin = axs[0].twiny()  # Create a secondary x-axis

    # Plot on the primary x-axis
    axs[0].plot(Times_arr_L_1O, Error_arr_mean_L_1O, label=r'$L_1$ Orthogonal Regularization', color='blue')

    axs[0].set_xlabel(r"L$_1$OR's  time [s]")
    axs[0].set_ylabel('Approximation factor')
    axs[0].set_xlim(0, Times_arr_L_1O[-1])
    axs[0].yaxis.grid(True)


    axs1_twin.plot(Times_arr_SLS, Error_arr_mean_SLS, label = 'Stochatic Landmark Selection',  color='green')
    axs1_twin.set_xlabel(r"SLS's time [s]")
    if ylim:
        axs1_twin.set_xlim(0, t_lim)
    else:
        axs1_twin.set_xlim(0, Times_arr_SLS[-1])
    # Set limits for the primary x-axis

    # Set labels and grid for the primary axis
    # axs[0].grid()

    # Set labels for the secondary x-axis
    # axs1_twin.set_xlim(axs[0].get_xlim())  # Synchronize x-limits with the primary axis

    # Second plot (Iteration vs. Approximation factor)
    axs2_twin = axs[1].twiny()  # Create a secondary x-axis

    # Plot on the primary x-axis
    axs[1].plot(n_arr_L_1O, Error_arr_mean_L_1O, label=r'$L_1$ Orthogonal Regularization', color='blue')
    # axs[1].plot(n_arr_SLS, Error_arr_mean_SLS, label='Landmark Selection', color='green')

    # Plot on the secondary x-axis
    axs2_twin.plot(n_arr_SLS, Error_arr_mean_SLS, label = 'Stochatic Landmark Selection' , color='green')


    # Set limits for the primary x-axis


    # Set labels and grid for the primary axis
    axs[1].set_xlabel(r"L$_1$OR's iterations")
    axs[1].set_xlim(0, n_arr_L_1O[-1])
    axs[1].yaxis.grid(True)

    # Set labels for the secondary x-axis
    axs2_twin.set_xlabel(r"SLS's iterations")
    if ylim:
        axs2_twin.set_xlim(0, n_lim)
    else:
        axs2_twin.set_xlim(0, n_arr_SLS[-1])
    # Add legends to both subplots
    axs[0].legend()
    axs2_twin.legend()

    # Final layout and display
    plt.tight_layout()
    if save:
        plt.savefig('results/speed3/speed3_'+dataset_name+'.eps')
    else:
        plt.show()
