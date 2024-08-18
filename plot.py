import numpy as np
import matplotlib.pyplot as plt
from CSSP import *
import file_txt as ft

def entrywise_l1_norm(A):
    return np.sum(np.abs(A))


def matrix_norm(W):
    # xI_n = np.eye(W.shape[0])
    return entrywise_l1_norm(W@W.T -np.identity(W.shape[0]))

# Generate values for W
lim = 1
W_values1 = np.linspace(-lim, lim, 31)
W_values2 = np.linspace(-lim*1.25, lim*1.25, 51)
X, Y = np.meshgrid(W_values1, W_values2)


def grad_d_l1( W, n2, s, stoch, sign_tol = False, tolg = 1e-5):
    gradient = np.zeros((n2,s))
    WW_I_sgn = np.sign(W@W.T - np.identity(n2))
    # print('WW_I_sgn : \n', WW_I_sgn)
    # WW_I_sgn = sign_tolerance(W@W.T - np.identity(n2), 1e-3)
    if stoch:
        j = np.random.randint(0, s)
        i = np.random.randint(0, n2)
        Bij = np.zeros((n2,s))
        Bij[i,j] = 1
        dWWT = Bij@W.T + W@Bij.T
        # print('dWWT : ', dWWT)
        if sign_tol:
            if np.abs(WW_I_sgn[i,j]) < tolg:
                gradient[i,j] = 0
        else:
            gradient[i,j] = np.sum(WW_I_sgn* dWWT)
    else :
        gradient = WW_I_sgn@W + WW_I_sgn.T@W
    return gradient

def CSSP_approximation_svd(X, n1, n2, s): # is correct 
    # (U, Sigma, Wt) = np.linalg.svd(X)
    # Ws = Wt.T[:,:s]
    # print('shape of X', np.shape(X))
    # print('s = ', s)
    (U, Sigma, WsT) = svds(X, k=s)
    Ws = WsT.T
    # print('shape of Ws', np.shape(Ws))
    return (Ws, function_objectif_W(X, Ws))

def Approximation_factor(X, s, indices):
    n1 = np.shape(X)[0]
    n2 = np.shape(X)[1]
    C = X[:,indices]
    error_svd = CSSP_approximation_svd(X, n1, n2, s)[1]
    error_C = function_objectif_C(X, C)
    return error_C/error_svd

def Approximation_factor_W(X, s, W):
    n1 = np.shape(X)[0]
    n2 = np.shape(X)[1]
    error_svd = CSSP_approximation_svd(X, n1, n2, s)[1]
    error_C = function_objectif_W(X, W)
    return error_C/error_svd
################PLOT1################

# lim = 1
# W_values1 = np.linspace(-lim, lim, 31)
# W_values2 = np.linspace(-lim*1.25, lim*1.25, 51)
# X, Y = np.meshgrid(W_values1, W_values2)

# # plt.show()
# # Compute the norm for each pair (W1, W2), while keeping W3 constant
# Z = np.zeros_like(X)
# W3 = 0  # Constant value for W3
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         W = np.array([[X[i, j]], [Y[i, j]]])#, [W3]])
#         Z[i, j] = matrix_norm(W)

# plt.plot(Y[:,0], Z[:,0])
# plt.xlabel(r'$W_{1}$')
# plt.ylabel(r'$\| WW^T - xI_n \|_1$')
# plt.grid()
# # plt.savefig('normW1_W2_fixed.svg')
# plt.show()

################PLOT2################

# # print(Z)
# fig = plt.figure()#figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')

# lim = 1.25
# n = 121
# W_values1 = np.linspace(-lim, lim, n)
# W_values2 = np.linspace(-lim, lim, n)
# X, Y = np.meshgrid(W_values1, W_values2)

# # Compute the norm for each pair (W1, W2), while keeping W3 constant
# Z = np.zeros_like(X)
# Z2 = np.zeros_like(X)
# W3 = 0  # Constant value for W3
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         W = np.array([[X[i, j]], [Y[i, j]]])#, [W3]])
#         Z[i, j] = matrix_norm(W)

# ax.set_xlabel(r'$W_{1}$')
# ax.set_ylabel(r'$W_{2}$')
# ax.set_zlabel(r'$\| WW^T - xI_n \|_1$')
# plt.tight_layout()
# plt.savefig('normW1_3D.eps')
# plt.show()

###########PLOT3################

# W_i = np.array([[-1.25],[1]])
# # print('WWT : ', W_i@W_i.T)
# print(grad_d_l1(W_i, 2, 1, False))
# c = 0 
# W_arr = []
# W_arr.append(W_i)
# niter=10
# while (c < niter-1):
#     c += 1
#     W_arr.append(np.array([W_arr[-1][0] - 0.15*grad_d_l1(W_arr[-1], 2, 1, False)[0], W_arr[-1][1]]))
# # print(W_arr)


# # plt.show()
# # Compute the norm for each pair (W1, W2), while keeping W3 constant
# Z = np.zeros_like(X)
# W3 = 0  # Constant value for W3
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         W = np.array([[X[i, j]], [Y[i, j]]])#, [W3]])
#         Z[i, j] = matrix_norm(W)


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# # First subplot
# ax1.plot(Y[:,0], Z[:,0], label=r'$\| WW^T - I_n \|_1$')
# ax1.plot([W_arr[i][0] for i in range(niter)], [np.sum(np.abs(W_arr[i]@W_arr[i].T - np.identity(2))) for i in range(niter)], 'o--', color='red', label=r'GD of $\| WW^T - I_n \|_1$')
# ax1.set_xlabel(r'$W_{1}$')
# # ax1.set_ylabel(r'$\| WW^T - xI_n \|_1$')
# ax1.grid()
# ax1.legend()

# # Second subplot
# ax2.plot([i for i in range(niter)], [np.sum(np.abs(W_arr[i]@W_arr[i].T - np.identity(2))) for i in range(niter)], 'o--', color='red')
# ax2.set_xlabel('Iteration')
# # ax2.set_ylabel(r'$\| WW^T - xI_n \|_1$')
# ax2.grid()
# plt.tight_layout()
# plt.show()
# # plt.savefig('WW-I2by1_GD.eps')

################PLOT4################
# W = np.random.randn(20, 10)
# print('shape of W : ', np.shape(W))
# n = 1
# W_arr = []
# # W_arr.append(W)
# W_arr1 = []
# alpha = 0.01
# alpha1 = 0.03
# niter=150
# tol = 1e-5
# f_best = matrix_norm(W)
# W1 = W.copy()
# W_arr.append(W)
# W_arr1.append(W1)
# while True and n < 150:
#     W = W - alpha*grad_d_l1(W, 20, 10, False)
#     W1 = W1 - alpha1*grad_d_l1(W1, 20, 10, False)
#     # alpha *= 0.9
#     W_arr.append(W)
#     W_arr1.append(W1)
#     # if matrix_norm(W)<60:
#     #     alpha = 0.001
#     # if matrix_norm(W)<42:
#     #     alpha *= 0.1
#     # print('norm of W : ', matrix_norm(W))
#     if matrix_norm(W) < 1e-5:
#         break
#     n += 1
#     fk = matrix_norm(W)
#     # fk1 = matrix_norm(W1)
#     if fk + tol < f_best:
#         f_best = fk
#         W_best = W
#         n_best = n
#     if (n-n_best>10):
#         W_i = W_best
#         # print(f'critere d arret after {n} iterations')
#         break

# #size of the plot 
# plt.figure(figsize=(10, 5))
# plt.plot([i for i in range(len(W_arr))], [matrix_norm(W_arr[i]) for i in range(len(W_arr))], label=r'$\alpha = 0.01 $')
# plt.plot([i for i in range(len(W_arr1))], [matrix_norm(W_arr1[i]) for i in range(len(W_arr1))], label=r'$\alpha = 0.03$')
# plt.plot([0, len(W_arr)], [10, 10], label='Lower bound', linestyle='--', color='red')
# plt.grid()
# plt.legend()
# plt.xlim(0, len(W_arr))
# plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel(r'$\| WW^T - xI_n \|_1$')
# plt.ylim(10, 40)
# # plt.savefig('GD_WW-I10by5.eps')
# plt.show()



# # ################PLOT5################
# stepsize_1 = 1e-1
# st1 = f'{stepsize_1}'
# ld_1 = 1e-2
# ld1 = f'{ld_1}'

# stepsize_2 = 1e-1
# st2 = f'{stepsize_2}'
# ld_2 = 1e-1
# ld2 = f'{ld_2}'

# stepsize_3 = 1e-2
# st3 = f'{stepsize_3}'
# ld_3 = 1e-1
# ld3 = f'{ld_3}'

# stepsize_4 = 1e-3
# st4 = f'{stepsize_4}'
# ld_4 = 1e0
# ld4 = f'{ld_4}'
# np.random.seed(43)
# X = np.random.randn(20, 10)
# label1 = r'$\alpha = ' + st1 + r'$, $\lambda = ' + ld1 + '$'
# label2 = r'$\alpha = ' + st2 + r'$, $\lambda = ' + ld2 + '$'
# label3 = r'$\alpha = ' + st3 + r'$, $\lambda = ' + ld3 + '$'
# label4 = r'$\alpha = ' + st4 + r'$, $\lambda = ' + ld4 + '$'

# (iter_1, penalization_1, f_1, W_arr_1) = L_1O(X, 20, 10, 5, ld_1, stepsize_1, 350, display = True, tol = 1e-8)#, plot5 = True )
# (iter_2, penalization_2, f_2, W_arr_2) = L_1O(X, 20, 10, 5, ld_2, stepsize_2, 350, display = True, tol = 1e-8)#, plot5 = True )
# (iter_3, penalization_3, f_3, W_arr_3) = L_1O(X, 20, 10, 5, ld_3, stepsize_3, 350, display = True, tol = 1e-8)#, plot5 = True )
# (iter_4, penalization_4, f_4, W_arr_4) = L_1O(X, 20, 10, 5, ld_4, stepsize_4, 350, display = True, tol = 1e-8)#, plot5 = True )
# # plt.plot(iter_1[1:], [matrix_norm(W_arr_1[i]) for i in range(1, len(W_arr_1))], label=r'$\alpha = 0.1$')
# #         #  , label=r'penalization , $\alpha = 0.1$')
# # plt.plot(iter_2[1:], [matrix_norm(W_arr_2[i]) for i in range(1, len(W_arr_2))], label=r'$\alpha = 0.01$')
# #         #  penalization_2[1:], label=r'penalization , $\alpha = 0.01$')


# Approx_1 = []
# Approx_2 = []
# Approx_3 = []
# Approx_4 = []
# # print(W_arr_1)
# for (i,W_arr_)in zip(range(len(W_arr_1)),W_arr_1):
#     Approx_1.append(f_1[i]/CSSP_approximation_svd(X, 20, 10, 5)[1])

# for (i,W_arr_)in zip(range(len(W_arr_2)),W_arr_2):
#     Approx_2.append(f_2[i]/CSSP_approximation_svd(X, 20, 10, 5)[1])

# for (i,W_arr_)in zip(range(len(W_arr_3)),W_arr_3):
#     Approx_3.append(f_3[i]/CSSP_approximation_svd(X, 20, 10, 5)[1])

# for (i,W_arr_)in zip(range(len(W_arr_4)),W_arr_4):
#     Approx_4.append(f_4[i]/CSSP_approximation_svd(X, 20, 10, 5)[1])

# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

# # ax1.plot(iter_1[1:], [matrix_norm(W_arr_1[i]) for i in range(1, len(W_arr_1))], label=r'$\alpha = 0.1$')
# # ax1.plot(iter_2[1:], [matrix_norm(W_arr_2[i]) for i in range(1, len(W_arr_2))], label=r'$\alpha = 0.01$')

# ax1.plot(iter_1[:], penalization_1[:]/5)
# ax1.plot(iter_2[:], penalization_2[:]/5)
# ax1.plot(iter_3[:], penalization_3[:]/5)
# ax1.plot(iter_4[:], penalization_4[:]/5)


# ax1.plot([0, len(penalization_1)], [1, 1], label='Lower bound', linestyle='--', color='red')
# ax1.set_legend()
# # ax1.set_xlabel('Iteration')
# # plt.ylabel(r'$\lambda \times \text{Penalization}(W)$')
# ax1.set_ylabel(r'$\frac{\| WW^T - I_n \|_1}{n-s}$')
# ax1.set_yscale('log')
# # ax1.show()
# # ax1.legend()
# ax1.grid(True)

# # ax2 = ax1.twinx()

# ax2.plot(iter_1, Approx_1, label= label1)
# ax2.plot(iter_2, Approx_2, label= label2)
# ax2.plot(iter_3, Approx_3, label= label3)
# ax2.plot(iter_4, Approx_4, label= label4)

# ax2.plot([0, len(penalization_1)], [1, 1], label='Lower bound', linestyle='--', color='red')
# # ax1.set_grid()
# ax2.legend()

# ax2.set_xlim(0, min(iter_1[-1], iter_2[-1]))
# ax2.set_xlabel('Iteration')
# ax2.set_ylabel(r'$\frac{\| X - XW(XW)^\dagger X \|_F}{\| X - X_s\|_F}$')
# ax2.grid(True)

# # plt.yscale('log')
# fig.tight_layout()
# # plt.savefig('results/definitif/GD_role_of_params.eps')
# plt.show()


#############PLOT params L_1O################

def plot_dict_(results_dict_L_1O, method = '', save=False, filename='', s=10, maxiter=12, iterlim = True, xind = (0,3), yind = (0,3), lb = 1.2, ub = 2.4):
    keys = list(results_dict_L_1O.keys())
    values = list(results_dict_L_1O.values())
    xl = 1
    yl = 1
    # print('this is keys : ', keys)
    # Extract the x and y values from the keys
    x_values = np.array([key[0] for key in keys])
    y_values = np.array([key[1] for key in keys])
    # x_values = np.delete(x_values, x_values.where(x_values.min()))
    # print('x_values : ', x_values)
    # print('y_values : ', y_values)
    error_array = np.zeros((len(set(x_values)), len(set(y_values))))
    iter_array = np.zeros((len(set(x_values)), len(set(y_values))))
    Mask = np.zeros((len(set(x_values)), len(set(y_values))))
    for i, key in enumerate(keys):
        x_index = np.where(np.array(sorted(set(x_values))) == key[0])[0][0]
        y_index = np.where(np.array(sorted(set(y_values))) == key[1])[0][0]
        error_array[x_index, y_index] = values[i][0] 
        iter_array[x_index, y_index] = values[i][1]
        if iterlim:
            Mask[x_index, y_index] = [True if values[i][1]<0.95*maxiter else False][0]
        else:
            Mask[x_index, y_index] = True
            Mask[x_index, y_index] = False if values[i][0] > 5 else True
    # print(type(Mask))
    # print('Mask : ', Mask)
    # print(type(error_array))
    # print('error_array : ', error_array)
    Mask = Mask[yind[0]:yind[1], xind[0]:xind[1]]
    error_array = error_array[yind[0]:yind[1], xind[0]:xind[1]]
    masked = np.ma.masked_where(Mask == 0, error_array)
    fig, ax = plt.subplots()
    cax = ax.imshow(masked, cmap='viridis', origin='lower', vmin=lb, vmax=ub)
    # plt.colorbar(label='Error')
    ax.set_xlabel(r'$ \alpha $', fontsize=14)
    ax.set_ylabel(r'$ \lambda $', fontsize=14)
    # name = f'{method}, m={n1}, n = {n2}, s = {s}'
    # ax.set_title(f'Approximation Factor')
    x_ = [x for x in sorted(set(x_values))]
    y_ = [x for x in sorted(set(y_values))]
    x_ = x_[yind[0]:yind[1]]
    y_ = y_[xind[0]:xind[1]]
    ax.set_xticks(np.arange(len(y_)))
    ax.set_xticklabels([f'{y:.1e}' for y in y_], rotation = 45, fontsize=12)
    ax.set_yticks(np.arange(len(x_)))
    ax.set_yticklabels([f'{x:.1e}' for x in x_], fontsize=12)
    #reduce the marge around the plot 
    plt.margins(0.0)

    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))

    cbar = fig.colorbar(cax)
    plt.tight_layout()
    fig.tight_layout()
    # fig.bbox_inches = 'tight'
    # plt.show()
    if save:
        # plt.show()
        plt.savefig('results/definitif/' +filename +'.eps', format='eps', bbox_inches='tight', pad_inches=0)
        # # plt.savefig('results/' +filename +'.eps')
        plt.close()
    else:   
        plt.show()
    return True 


##########MASSART######

# dict_L_1O_20x_50 = ft.read_dict_best_params_from_file("results/clean/optimal parameters/Optimal params text file/20x50_s_10_L_1O_large__ .txt")
# plot_dict_(dict_L_1O_20x_50, method = 'Massart', save=True, filename='params_L_1O_50x20', s=10, maxiter=2000, xind = (0,5), yind= (0,5), lb = 1.24, ub = 1.42)

# dict_L_1O_arrithmia = ft.read_dict_best_params_from_file("results/clean/optimal parameters/Optimal params text file/100x281_s_30_L_1O_Arrithmia1.txt")
# plot_dict_(dict_L_1O_arrithmia, method = 'Massart', save=True, filename='params_L_1O_arrithmia', s=30, maxiter=400, iterlim = True, xind = (1,6), yind= (0,5), lb = 1.3, ub = 1.93)

# dict_L_1O_MNIST = ft.read_dict_best_params_from_file("results/clean/optimal parameters/Optimal params text file/100x784_s_30_L_1O_MNIST1.txt")
# plot_dict_(dict_L_1O_MNIST, method = 'Massart', save=True, filename='params_L_1O_MNIST', s=30, maxiter=400, iterlim = True, xind=(0,5), yind=(1,6), lb = 1.38, ub = 1.8)

# ##############-----MATHUR  STOCHASTIC -----------################
# dict_SLS_20x50 = ft.read_dict_best_params_from_file("results/clean/optimal parameters/Optimal params text file/20x50_s_10_SLS_stoch_large.txt")
# plot_dict_(dict_SLS_20x50, method = 'Mathur', save=True, filename='params_SLS_20x50_stoch', s=10, maxiter=2000, xind = (0,4), yind= (0,4),  lb = 1.24, ub = 1.42)

# dict_SLS_Arrithmia = ft.read_dict_best_params_from_file("results/clean/optimal parameters/Optimal params text file/100x280_s_30_SLS_stoch_Arrithmia_stoch.txt")
# # plot_dict_(dict_SLS_Arrithmia, method = 'Mathur', save=True, filename='params_SLS_Arrithmia_stoch', s=30, iterlim=True, maxiter=400, xind = (0,4), yind= (0,5), lb = 1.3, ub = 1.93)
# plot_dict_(dict_SLS_Arrithmia, method = 'Mathur', save=True, filename='params_SLS_Arrithmia_stoch_zoom', s=30, iterlim=True, maxiter=400, xind = (0,5), yind= (0,5), lb = 1.3, ub = 1.93)

# dict_SLS_MNIST = ft.read_dict_best_params_from_file("results/clean/optimal parameters/Optimal params text file/100x784_s_30_SLS_stoch_MNIST_stoch_2.txt")
# plot_dict_(dict_SLS_MNIST, method = 'Mathur', save=True, filename='params_SLS_MNIST_stoch', s=30, maxiter=500, iterlim = True, xind = (0,4), yind= (0,5), lb = 1.38, ub = 1.8)






###############-----MATHUR NOT  STOCHASTIC -----------################

# dict_SLS_stoch_20x50 = ft.read_dict_best_params_from_file("results/clean/optimal parameters/Optimal params text file/20x50_s_10_SLS_large_NOT_STOCH.txt")
# plot_dict_(dict_SLS_stoch_20x50, method = 'Mathur_STOCH', save=False, filename='params_SLS_20x50_stoch', s=10, maxiter=2000, xind = (0,4), yind= (0,4), iterlim = True)

# dict_SLS_MNIST_not_stoch = ft.read_dict_best_params_from_file("results/clean/optimal parameters/Optimal params text file/100x784_s_30_SLS_MNIST___not_stoch.txt")
# plot_dict_(dict_SLS_MNIST_not_stoch, method = 'Mathur', save=False, filename='params_SLS_MNIST_not_stoch', s=30, maxiter=500, iterlim = True, xind = (0,7), yind= (0,7))

# dict_SLS_Arrithmia_not_stoch = ft.read_dict_best_params_from_file("results/clean/optimal parameters/Optimal params text file/100x281_s_30_SLS_Arrithmia_not_stoch.txt")
# plot_dict_(dict_SLS_Arrithmia_not_stoch, method = 'Mathur', save=False, filename='params_SLS_Arrithmia_not_stoch', s=30, iterlim=True, maxiter=500, xind = (0,7), yind= (0,7))



####PLOT EXPLANTION MASSART BEST PARAMS 


# np.random.seed(42)
# X1 = np.random.randn(20, 50)

# import mnist
# from mnist import MNIST
# #MNIST1K 
# mndata = MNIST('samples')

# images, labels = mndata.load_training()

# #shuflle the images using a seed
# np.random.shuffle(images)

# print(np.shape(images))
# images = np.array(images)
# images_stdfull = (images - images.mean())/images.std()
# images_std = np.array(images_stdfull[:100])
# print(np.shape(images_std))

# L_1O_results_X1 = L_1O( X1, 20, 50, 10, 1e-7, 1e4, 2000, display = True, tol = 1e-3, filename= 'LOR_Random_evolution_')
# L_1O_results_ = L_1O( images_std, 100, 784, 30, 1e-1, 1e-3, 2000, display = True, tol = 1e-3, filename= 'LOR_MNIST_evolution_') #0.2599807458065957


# read a file that has been created like this 
# with open('results/'+filename+'t.txt', 'w') as f:
#     for i in range(len(iter)):
#         f.write(f'{iter[i]}\t{ld_penality[i]}\t{f_min_without_penalty[i]}\t{grad_r[i]}\t{grad_l[i]}\n')

filebame = 'LOR_MNIST_evolution'
with open('results/LOR_MNIST_evolution_t.txt', 'r') as f:
    lines = f.readlines()
    iter_MNIST = []
    ld_penality_MNIST = []
    f_min_without_penalty_MNIST = []
    grad_r_MNIST = []
    grad_l_MNIST = []
    for line in lines:
        line = line.split('\t')
        iter_MNIST.append(int(line[0]))
        ld_penality_MNIST.append(float(line[1]))
        f_min_without_penalty_MNIST.append(float(line[2]))
        grad_r_MNIST.append(float(line[3]))
        grad_l_MNIST.append(float(line[4]))

with open('results/LOR_Random_evolution_t.txt', 'r') as f:
    lines = f.readlines()
    iter_X1 = []
    ld_penality_X1 = []
    f_min_without_penalty_X1 = []
    grad_r_X1 = []
    grad_l_X1 = []
    for line in lines:
        line = line.split('\t')
        iter_X1.append(int(line[0]))
        ld_penality_X1.append(float(line[1]))
        f_min_without_penalty_X1.append(float(line[2]))
        grad_r_X1.append(float(line[3]))
        grad_l_X1.append(float(line[4]))
                         
#create a framework with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex='col')

axs[0, 0].sharey(axs[0, 1])
# 1, 1 contains f_min_without_penalty_X1 and ld_penality_X1
axs[0, 0].plot(iter_X1, f_min_without_penalty_X1, label=r'$\frac{\|X-XW(XW)^\dagger X\|_F}{\|X - X_s\|_F}$', color = 'blue')
# axs[0, 0].set_yscale('log')
axs[0, 0].set_title('Random')
axs[0, 0].grid(True)  # add grid
# axs[0, 0].set_ylabel(r'$\frac{\|X-XW(XW)^\dagger X\|_F}{\|X - X_s\|_F}$', color = 'blue', fontsize=16)
axs[0, 0].tick_params(axis='y', labelcolor='blue')
# axs[0, 0].set_ylim(1e0, 1e1)


axs00 = axs[0, 0].twinx()

axs00.plot((5,10), (-2, -2), label=r'$\frac{\|X-XW(XW)^\dagger X\|_F}{\|X - X_s\|_F}$', color = 'blue')
axs00.plot(iter_X1, ld_penality_X1, label=r'$\lambda \cdot\frac{\|WW^\top - I_n\|_1}{n-s}$', color = 'red')
axs00.set_yscale('log')
axs00.legend(fontsize=14)
axs00.tick_params(axis='y', labelcolor='red')


# 1, 2 contains grad_r_X1 and grad_l_X1
axs[1, 0].plot(iter_X1, grad_l_X1, label=r'$\alpha\cdot$ mean of $ \nabla_W F^X$ ', color = 'dodgerblue')
axs[1, 0].plot(iter_X1, grad_r_X1, label=r'$\alpha\cdot\lambda\cdot$ mean of $ \nabla_W R_\lambda$', color = 'indianred' )#'dodgerblue')
axs[1, 0].set_yscale('log')
axs[1, 0].grid(True)  # add grid
axs[1, 0].set_xlabel('Iteration')



# 2, 1 contains f_min_without_penalty_MNIST and ld_penality_MNIST
axs[0, 1].plot(iter_MNIST, f_min_without_penalty_MNIST, color = 'blue')
axs[0, 1].set_title('MNIST')
axs[0, 1].grid(True)  # add grid
axs[0, 1].tick_params(axis='y', labelcolor='blue')


axs01 = axs[0, 1].twinx()
axs[0, 1].legend(loc='upper left')
axs01.legend(loc='upper right')
axs00.sharey(axs01)
axs01.plot((5,10), (-2, -2), label=r'$\frac{\|X-XW(XW)^\dagger X\|_F}{\|X - X_s\|_F}$', color = 'blue')
axs01.plot(iter_MNIST, ld_penality_MNIST, label=r'$\lambda \cdot \frac{\|WW^\top - I_n\|_1}{n-s}$', color = 'red')
axs01.set_yscale('log')
axs01.set_ylim(1e-2, 1e2)
# axs01.set_ylabel(r'$\lambda \cdot \frac{\|WW^\top - I_n\|_1}{n-s}$', color = 'red')
axs01.tick_params(axis='y', labelcolor='red')
axs01.legend(fontsize=14)


# 2, 2 contains grad_r_MNIST and grad_l_MNIST
axs[1, 1].plot(iter_MNIST, grad_l_MNIST, label=r'$\alpha\cdot$ mean of $ \nabla_W F^X$ ', color = 'dodgerblue')
axs[1, 1].plot(iter_MNIST, grad_r_MNIST, label=r'$\alpha\cdot\lambda\cdot$ mean of $ \nabla_W R_\lambda$', color = 'indianred')
axs[1, 1].set_yscale('log')
# axs[1, 1].set_title('MNIST')
axs[1, 1].grid(True)  # add grid
axs[1, 1].set_xlabel('Iteration')
# axs[1, 1].set_ylim(1e-4, 1e-1)

# add the legend
# axs[0, 0].legend()
# axs[0, 1].legend()
axs[1, 0].legend(fontsize=12)
axs[1, 1].legend(fontsize=12)

plt.tight_layout()
# plt.show()
plt.savefig('results/definitif/evolution_LOR.eps')