import test as t
import numpy as np
import mnist
from mnist import MNIST
import random
import pandas as pd
import time
import matplotlib.pyplot as plt

#

# import file_txt as ft

#MNIST1K 
mndata = MNIST('samples')

images, labels = mndata.load_training()

#shuflle the images using a seed
np.random.seed(42)
np.random.shuffle(images)

print(np.shape(images))
images = np.array(images)
images_stdfull = (images - images.mean())/images.std()
images_std = np.array(images_stdfull[:100])
print(np.shape(images_std))
# images_std_lst = []
# for i in range(15):
#     np.random.shuffle(images_stdfull)
#     images_std_lst.append(images_stdfull[:100])
# ld = 1e-12
# step = 1e-6


# print(results[1:])
ld = np.geomspace(1e-5, 1e-1, 5) 
step = np.geomspace(1e-4, 1e1, 6)
# t.test_find_optimal_params_L_1O_real_matrix(images_std, Lambdas= ld , steps = step, s=30, save = True, tol = 1e-3, filename='Massart_Mnist', display = False, itermax= 400, n_trials = 10) #OK done

print(ld)
print(step)

# t.std_data('data/arrhythmia.data', file_std='data/arrhythmia_std.data')
# print('done')
data = pd.read_csv('data/arrhythmia_std.data', header=None)

# data = data.drop(index = 0, axis=0)
#replace the Nan values by 0
data = data.fillna(0)
data = data.to_numpy()

np.random.seed(42)
np.random.shuffle(data)
print(np.shape(data))
data_arrithmia = data[:100]


# print(data[0])
# 
# 
# data_arrithmia_list = []
# for i in range(15):
#     np.random.shuffle(data)
#     data_arrithmia_list.append(data[:100])

#     data_arrithmia_list.append(data_arrithmia[i])
print('shuffle done')

######-------------MASSART optimal parameters----------------######## 
##Random matrix
# t.test_find_optimal_params_L_1O(Lambdas= np.geomspace(1e-9, 1e-5, 5), steps = np.geomspace(1e2, 1e6, 5),
#                                     save = True, n_test=100, tol = 1e-3, itermax=2000) #done
## Arrithmia
# t.test_find_optimal_params_L_1O_real_matrix(data_arrithmia, Lambdas= np.geomspace(1e-3, 1e2, 6)  , steps = np.geomspace(1e-5, 1e0, 6), s=30, save = True,
#                                                 tol = 1e-3, filename='Arrithmia1', display = False, itermax= 400, n_trials = 50) #done
## MNIST
# t.test_find_optimal_params_L_1O_real_matrix(images_std, Lambdas= np.geomspace(1e-5, 1e0, 6)  , steps = np.geomspace(1e-4, 1e1, 6)
#                                                 ,s=30, save = True, tol = 1e-3, filename='MNIST1', display = False, itermax= 400, n_trials = 50) #done
# 
# t.L_1O(images_std, 100, 784, 30, 1e1, 1e-4, 400, display = True, tol = 1e-10, save=False) 
######-------------MATHUR stochatic optimal parameters----------------######## 

# t.test_find_optimal_params_SLS(Lambdas= np.geomspace(1e-3, 1e1, 5), steps = np.geomspace(1e-6, 1e-2, 5), 
#                                   stochastic=True, n_test=100, itermax=2000, save = True, tol = 1e-3) # OK done 


ld = np.geomspace(1e-10, 1e1, 6)
print(ld)
step = [1e-2, 1e-1]
ld = np.geomspace(1e-6, 1e-3, 4)
step = np.geomspace(1e-3, 1e0, 4)
# step = [ 0.025, 0.05, 0.1, 0.15,  0.2]#np.geomspace(1e-2, 10.0, 4)
# print(step)
ld = np.geomspace(0.01, 10, 4)
step = np.geomspace(1e-4, 1e-1, 4)
# t.test_find_optimal_params_SLS_real_matrix(data_arrithmia, Lambdas= ld , steps = step, s=30, save = True, 
#                                               stochastic = True, tol = 1e-3, filename='Arrithmia_stoch_1', itermax= 400, 
#                                               X_list = False, n_test = 5) #to run 

# ld = np.geomspace(0.01, 10, 4)
# step = np.geomspace(1e-3, 1.0, 4)
# t.test_find_optimal_params_SLS_real_matrix(images_std, Lambdas= ld , steps = step, s=30, save = True, 
#                                               stochastic = True, tol = 1e-3, filename='MNIST_stoch', itermax= 400, 
#                                               X_list = False, n_test = 4) #ok


#############------------------INfluence of M -----------------############## #done 
# t.influence_of_M_SLS([[1]], 20, 50, 10,ld = 1, step = 1e-2,n_test = 30, filename = 'random', itermax=20)
# t.influence_of_M_SLS(data_arrithmia, 20, 50, 30,ld = 1e-5, step = 1e-1,n_test = 30, filename = 'Arrithmia', itermax=400, real_matrix=True)
# t.influence_of_M_SLS(images_std, 20, 50, 30, ld = 1e0, step = 1e-3,n_test = 3, filename = 'MNIST', itermax=400, real_matrix=True)

# t.plot_influence_of_M_SLS('random', save = True)
# t.plot_influence_of_M_SLS('MNIST')
# t.plot_influence_of_M_SLS('Arrithmia')


###############-----------------Comparison-----------------############## # done 
# timer = time.tim('e()
# t.L_1O_SLS_comparison( n_test=20, filename='test2', save = True, itermax=2000, uniform_only = True) #to run 25/07 43min
# print('time taken : ', time.time() - timer)
# timer = time.time()
# t.L_1O_SLS_comparison_arrithmia(data_arrithmia, n_test=20, filename='arrithmia_test', save = True, itermax=400) #to run 25/07 1h30 
# print('time taken : ', time.time() - timer)
# timer = time.time()
# t.L_1O_SLS_comparison_MNIST(images_std, n_test=20, filename='MNIST2', save = True, itermax=400) #to run 25/07 1h
# print('time taken : ', time.time() - timer)

# t.test_comparison_plot('definitif/comparison_perf_MNIST_mat', save = True)
# t.test_comparison_plot('definitif/comparison_perf_arrithmia_mat', uniform=True, save = True)
# t.test_comparison_plot('definitif/comparison_perf_random_mat', save = False, random = True)

s = np.arange(1,15,1)
print(s/20*100)
##############_______________SPEED______________################


# t.test_speed_1(r = 11)
# t.plot_speed_1(save = True)

# t.test_speed_2()
# t.plot_speed_2(save = True)



# np.random.seed(42)
# Xlist = []
# n_test = 20
# for i in range(n_test):
#     Xlist.append(np.random.rand(20,50))
# ld_L_1O = 1e-8
# step_L_1O = 1e5
# ld_SLS = 1e-1
# step_SLS = 5e-2
# t.test_speed_3(Xlist, 10, 2000 , 3, ld_L_1O, step_L_1O, ld_SLS, step_SLS, filename= 'random')
# t.plot_speed_3(filename_L_1O='speed3_comparison_L_1O_random', filename_SLS='speed3_comparison_SLS_random', dataset_name= 'Random', save = True)

# np.random.seed(42)
# images_std_lst = []
# n_test = 5
# for i in range(n_test):
#     np.random.shuffle(images_stdfull)
#     images_std_lst.append(images_stdfull[:100])
# ld_L_1O = 1e-1
# step_L_1O = 1e-3
# ld_SLS = 1e0
# step_SLS = 1e-3
# t.test_speed_3(images_std_lst, 30, 400,3, ld_L_1O, step_L_1O, ld_SLS, step_SLS, filename= 'MNIST_2's)
# t.plot_speed_3(filename_L_1O='speed3_comparison_L_1O_MNIST_2', filename_SLS='speed3_comparison_SLS_MNIST_2', dataset_name= 'MNIST', save = True)

# np.random.seed(42)
np.random.shuffle(data)
data_arrithmia = data[:100]
n_test = 5
data_arrithmia_list = []
for i in range(n_test):
    np.random.shuffle(data)
    data_arrithmia_list.append(data[:100])

ld_L_1O = 1e-0
step_L_1O = 1e-3
# ld_SLS = 1e-0
# step_SLS = 1e-3
ld_SLS = 1e0
step_SLS = 1e-3

# print(data_arrithmia_list[0])
# t.test_speed_3(data_arrithmia_list, 30, 400, 1, ld_L_1O, step_L_1O, ld_SLS, step_SLS, filename= 'Arrithmia_std')
# t.plot_speed_3(filename_L_1O='speed3_comparison_L_1O_Arrithmia_std', 
#                filename_SLS='speed3_comparison_SLS_Arrithmia_std', dataset_name= "Arrithmia" , save = True, ylim = False)


