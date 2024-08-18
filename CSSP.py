import numpy as np
from numpy.linalg import norm
from itertools import combinations
import time
import matplotlib.pyplot as plt 
import file_txt as ft
import scipy as sp
from scipy.linalg import lu
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import scipy

def entrywise_l1_norm(A):
    return np.sum(np.abs(A))

# np.set_printoptions(precision=3)

def moving_average(arr, window_size = 5):
    weights = np.repeat(1.0, window_size) / window_size
    print(weights)
    return np.convolve(arr, weights, mode='valid')

def generate_permutation_matrix(n2):
    P = np.eye(n2)
    np.random.shuffle(P)
    # print(np.shape(P))
    return P


def sign_tolerance(matrix, tol=1e-5):
    return np.where(np.abs(matrix) < tol, 0, np.sign(matrix))

def LHS_function_objectif(X ,W ):
    return np.linalg.norm(X - X@W @ np.linalg.pinv(X@W)@X, ord='fro')

def RHS_function_objectif(n2 ,W ):
    return entrywise_l1_norm(W@W.T -np.identity(n2))

def SLS_f_to_min(X, t, delta, lamda, n2):
    T = np.diag(t)
    P_tilde = X@T@np.linalg.pinv(T@X.T@X@T + delta*(np.identity(n2) - np.square(T)))@T@X.T
    return -np.trace(X.T@P_tilde@X) + lamda*np.sum(t)

def SLS_f_to_min_fast(X, t, delta, lamda, n2):
    T = np.diag(t)
    XT = np.dot(X, T)
    TXT = np.dot(T, X.T)
    P_tilde = np.dot(np.dot(XT,np.linalg.pinv(np.dot(TXT,XT) + delta*(np.identity(n2) - np.square(T)))),TXT)
    return -np.trace(np.dot(np.dot(X.T,P_tilde),X)) + lamda*np.sum(t)

def SLS_f_LHS(X, t, delta, lamda, n2):
    T = np.diag(t)
    # print(T@X.T@X@T + delta*(np.identity(n2) - T@T))
    # print('shape of TxTT+delta(I-TT)) ' , np.shape(T@X.T@X@T + delta*(np.identity(n2) - T@T)))
    P_tilde = X@T@np.linalg.pinv(T@X.T@X@T + delta*(np.identity(n2) - T@T))@T@X.T
    return np.linalg.norm(X - P_tilde@X, ord='fro')


## Calcul of d W/ d W_ij
def grad_g_slow( X, W,  n1, n2, s):
    gd = np.zeros((n2,s))
    A = X@W  # dim n1 x s 
    Aplus = np.linalg.pinv(A) 
    # print('shape of A', np.shape(A))
    un = 0 
    for i in range(n2):
        for j in range(s):
            dW = np.zeros((n2,s))
            dW[i,j]=1
            dA = X@dW
            # dAT = dW.T@X.T
            # dAT = dA.T
            # DAplus = (-Aplus@dA@Aplus + Aplus@Aplus.T@dA.T@(np.identity(n1)-A@Aplus) + (np.identity(s)-Aplus@A)@dA.T@Aplus.T@Aplus)
            un = dA@Aplus - A@Aplus@dA@Aplus + Aplus.T@dA.T@(np.identity(n1)-A@Aplus)
            # gd[i,j]= -np.trace(X.T@(dA@Aplus+A@DAplus)@X)
            gd[i,j]= -np.trace(X.T@un@X)
    return gd

def function_objectif_C(X ,C ):
    return np.linalg.norm(X - C @ np.linalg.pinv(C)@X, ord='fro')

def function_objectif_W(X ,W ):
    return np.linalg.norm(X - X@W @ np.linalg.pinv(X@W)@X, ord='fro')

def function_objectif_W_RHS(X ,W, lamda, n2):
    try:
        return np.linalg.norm(X - np.dot(np.dot(np.dot(X,W), np.linalg.pinv(np.dot(X,W))),X), ord='fro') + lamda*entrywise_l1_norm(np.dot(W,W.T) -np.identity(n2)) #np.linalg.norm(W@W.T -np.identity(n2), ord='nuc')
    except np.linalg.LinAlgError:
        ft.store_matrix(X@W, 'X@W_error.txt')
        random_combination = np.random.choice(range(n2), np.shape(X@W)[1], replace=False)
        C = X[:,random_combination]
        return function_objectif_C( X, C,)

def CSSP_approximation_svd(X, n1, n2, s): # is correct 
    # (U, Sigma, Wt) = np.linalg.svd(X)
    # Ws = Wt.T[:,:s]
    # print('shape of X', np.shape(X))
    # print('s = ', s)
    (U, Sigma, WsT) = svds(X, k=s)
    Ws = WsT.T
    # print('shape of Ws', np.shape(Ws))
    return (Ws, function_objectif_W(X, Ws))

def calculate_right_eigenvectors_k_svd(X_,k):
    _,_,V_k = svds(X_, k,  return_singular_vectors='vh')
    # (U, Sigma, V_k_) = np.linalg.svd(X_)
    return V_k

def Approximation_factor(X, s, indices):
    n1 = np.shape(X)[0]
    n2 = np.shape(X)[1]
    C = X[:,indices]
    error_svd = CSSP_approximation_svd(X, n1, n2, s)[1]
    error_C = function_objectif_C(X, C)
    return error_C/error_svd

def Approximation_factor_C(X, s, C):
    n1 = np.shape(X)[0]
    n2 = np.shape(X)[1]
    error_svd = CSSP_approximation_svd(X, n1, n2, s)[1]
    error_C = function_objectif_C(X, C)
    return error_C/error_svd


def find_non_null_columns(matrix):  #ChatGPT
        non_null_columns = np.where(matrix.any(axis=1))[0]
        return non_null_columns

def is_permuation_matrix(x): ##Stackoverflow
    x = np.asanyarray(x)
    return (x.ndim == 2 and x.shape[0] == x.shape[1] and
            (x.sum(axis=0) == 1).all() and 
            (x.sum(axis=1) == 1).all() and
            ((x == 1) | (x == 0)).all())

def round_matrix(matrix, decimals=2): #Chatgpt
    rounded_matrix = []
    for row in matrix:
        rounded_row = []
        for element in row:
            rounded_element = round(element, decimals)
            rounded_row.append(rounded_element)
        rounded_matrix.append(np.array(rounded_row))
    return np.array(rounded_matrix)

def orthogonal_procrustes(A, B): #Chatgpt
    """
    Solve the Orthogonal Procrustes Problem using Singular Value Decomposition (SVD).

    A and B are two matrices to be aligned.
    Returns the orthogonal matrix R that aligns A to B.
    """
    # Compute the optimal rotation matrix R using Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(np.dot(B.T, A))
    R = np.dot(U, Vt)

    # Ensure determinant of R is +1 to maintain proper orientation
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = np.dot(U, Vt)

    return R

def brute_force(X, n1, n2, s):
    combinations_list = list(combinations(range(n2), s))
    min_value = float('inf')
    for indx in combinations_list:
        C = X[:,indx]
        value = function_objectif_C( X, C)
        if value < min_value:
            min_value = value
            min_indx = indx
    return (min_indx, function_objectif_C( X, X[:, min_indx]))

def average_random_CSSP(X, n1, n2, s):
    combinations_list = list(combinations(range(n2), s))
    value = 0
    for indx in combinations_list:
        C = X[:,indx]
        value += function_objectif_C( X, C)
    value /= len(combinations_list)
    return value, function_objectif_C( X, C)

def grad_d_l1(W, n2):
    WW_I_sgn = np.sign(np.dot(W,W.T) - np.identity(n2))
    return np.dot(WW_I_sgn,W) + np.dot(WW_I_sgn.T,W)

def grad_g( X, W, n1, n2, s): #4h de taff
    gd = np.zeros((n2,s))
    A = X@W  # dim n1 x s 
    Aplus = np.linalg.pinv(A) 
    E1 = Aplus@X@X.T
    E2 = E1@A@Aplus
    E3 = (np.identity(n1)-A@Aplus)@X@X.T@Aplus.T
    return - (E1@X).T + (E2@X).T -(E3.T@X).T 

    
def grad_g_fast(X, W, XXT, n1):
    A = np.dot(X,W)  # dim n1 x s 
    Aplus = np.linalg.pinv(A) 
    AAplus = np.dot(A,Aplus)
    E1 = np.dot(Aplus,XXT)
    return - 2*np.dot(np.dot(X.T,XXT),Aplus.T) + 2*np.dot(np.dot(np.dot(X.T, AAplus),XXT), Aplus.T)

def L_1O( X, n1, n2, s, ld, step, maxiter, display = False, tol = 1e-8, stupid_method = False, save = False, sign_tol = False, tolg = 1e-5,
            filename =''):
    error = CSSP_approximation_svd(X, n1, n2, s)[1]
    n = 0
    W_i = np.random.rand(n2,s)
    W_arr = [W_i]
    XXT = np.dot(X, X.T)
    gl = grad_g_fast(X, W_i, XXT, n1)
    gr = grad_d_l1(W_i, n2)
    if display:
        ld_penality = []
        LHS = []
        iter = []
        grad_r = []
        grad_l = []
        grad = []
        f_min_without_penalty = []
        # fstar = [second_part_of_L_1O(X, W_i, n2, s)]
    f_min = [function_objectif_W_RHS(X, W_i, ld, n2)]
    f_best = float('inf')
    n_best = 0
    while (n < maxiter):
        # if (prog and n%100==0):
        #      ld = 1.01*ld
        # if (n%50==0):
        #     print(n)
        try:
            gl = grad_g_fast(X, W_i, XXT, n1)
            gr = grad_d_l1(W_i, n2)
        except np.linalg.LinAlgError:
            return (np.random.choice(range(n2), s, replace=False), function_objectif_C( X, X[:,np.random.choice(range(n2), s, replace=False)]), maxiter)
        W_i -= step*(gl + ld*gr)
        if (display):# and n%10==0):
            ld_penality.append(ld*RHS_function_objectif(n2,W_i)/(n2-s))
            # LHS.append(LHS_function_objectif(X,W_i))
            grad_r.append(step*ld*np.mean(np.abs(gr)))
            grad.append(np.linalg.norm(gl,ord='nuc') + ld*np.linalg.norm(gr,ord = 'nuc'))
            iter.append(n)
            grad_l.append(step*np.mean(np.abs(gl)))
            f_min.append(function_objectif_W_RHS(X, W_i, ld, n2))
            f_min_without_penalty.append(function_objectif_W(X, W_i)/error)
            W_arr.append(W_i)
            # fstar.append(second_part_of_L_1O(X, W_i, n2, s))
        n+=1   
        fk = function_objectif_W_RHS(X, W_i, ld, n2)
        if fk + tol < f_best:
            f_best = fk
            W_best = W_i
            n_best = n
        if (n-n_best>10):
            W_i = W_best
            # print(f'critere d arret after {n} iterations')
            break
    W = W_i
    if display:
        with open('results/'+filename+'t.txt', 'w') as f:
            for i in range(len(iter)):
                f.write(f'{iter[i]}\t{ld_penality[i]}\t{f_min_without_penalty[i]}\t{grad_r[i]}\t{grad_l[i]}\n')
    if display:
        plt.plot(iter[1:], np.array(ld_penality[1:]), label='penalty (with ld)')
        # plt.plot(iter,(1/ld)*np.array(RHS),label =  r'$\| WW^T - I_n \|_1$')
        # plt.plot(iter, grad_l, label= 'ld*step*grad LHS')
        # print(grad_l)
        # print(grad_r)
        # plt.plot(iter, grad_r, label = 'ld*grad RHS')
        # plt.plot(iter[1:], grad_r[1:], label='ld*grad_penalty')
        # plt.plot(iter, grad, label = 'grad')
        # plt.plot(iter, f_min, label = 'f_with_penalty')
        plt.plot(iter[1:], np.array(f_min_without_penalty[1:]), label = 'f_without_penalty')
        plt.ylim(1e0, 1e4)
        # # plt.plot(iter, fstar, label = 'fstar')
        # # plt.plot(iter, err_avg, label = 'err_avg')  # Add err_avg to the plot
        plt.legend()
        plt.grid()
        plt.yscale('log')
        if save:   
            plt.savefig('results/plot_L_1O_.svg')
            # plt.show()
        else:
            plt.show()
    # ---------------------- seconde partie --------------------------------
    try:
        (Pp, Sigmap, Vp) = np.linalg.svd(W)
    except np.linalg.LinAlgError:
        ft.store_matrix(W, 'W_error.txt')
        random_combination = np.random.choice(range(n2), s, replace=False)
        C = X[:,random_combination]
        return (random_combination, function_objectif_C( X, C), n)
    # ft.store_matrix(Pp, 'Ps_error.txt')
    PsIs = Pp@np.eye(n2,s)
    non_null_columns = np.argsort(np.sum(np.abs(PsIs)**2,axis=-1)**(1./2))[-s:] #find_lines_with_largest_abs(PsIs, s)
    C_emp = X[:,non_null_columns]
    return (non_null_columns, function_objectif_C( X, C_emp), n)


def L_1O_part1_speed( X, n1, n2, s, ld, step, maxiter, mod, display = False, tol = 1e-5, smart_start = False, stupid_method = False, save = False):
    stoch = False  #stoch for stochastic gradient descent
    if smart_start:
        W_i = CSSP_approximation_svd(X, n1, n2, s)[0]
    else: 
        permutation_matrix = generate_permutation_matrix(n2)
        W_i = permutation_matrix@np.eye(n2,s)
    n = 0
    W_array = np.zeros((maxiter//mod+1,n2,s))
    Timer_array = np.zeros(maxiter//mod+1)
    n_array = np.zeros(maxiter//mod+1)
    Error_array = np.zeros(maxiter//mod+1)
    W_i = np.random.rand(n2,s)
    XXT = np.dot(X, X.T)
    gl = grad_g_fast(X, W_i, XXT, n1)
    gr = grad_d_l1(W_i, n2)
    fk_best_current = float('inf')
    f_best = float('inf')
    if stupid_method:
        combinations_list = list(combinations(range(20), s))
        value = 0
        for indx in combinations_list[:50]:
            C = X[:,indx]
            value += function_objectif_C( X, C)
        value /= 50
    n_best = 0
    timer = time.time()
    while (n < maxiter):
        if (n%mod==0):
            Timer_array[n//mod] = time.time() - timer
            W_array[n//mod] = W_i
            n_array[n//mod] = n
        try:
            gl = grad_g_fast(X, W_i, XXT, n1)
            gr = grad_d_l1(W_i, n2)
        except np.linalg.LinAlgError:
            return (np.random.choice(range(n2), s, replace=False), function_objectif_C( X, X[:,np.random.choice(range(n2), s, replace=False)]), maxiter)
        W_i -= step*(gl + ld*gr)
        n+=1
        fk = function_objectif_W_RHS(X, W_i, ld, n2)
        if fk + tol < f_best:
            f_best = fk
            W_best = W_i
            n_best = n
        if (n-n_best>10):
            W_i = W_best
            # print(f'critere d arret after {n} iterations')
            break
    for i in range(n//mod):
        timer = time.time()
            # ---------------------- seconde partie --------------------------------
        try:
            (Pp, Sigmap, Vp) = np.linalg.svd(W_array[i])
        except np.linalg.LinAlgError:
            ft.store_matrix(W, 'W_error.txt')
            print('error in svd')
            random_combination = np.random.choice(range(n2), s, replace=False)
            C = X[:,random_combination]
            return (random_combination, function_objectif_C( X, C), n)
        PsIs = Pp@np.eye(n2,s)
        non_null_columns = np.argsort(np.sum(np.abs(PsIs)**2,axis=-1)**(1./2))[-s:] #find_lines_with_largest_abs(PsIs, s)
        # Timer_array[i] += time.time() - timer
        Error_array[i] = function_objectif_C( X, X[:,non_null_columns])
    return (Timer_array, n_array, Error_array)
    

def grad_SLS(X : np.ndarray, t : np.array, delta :float, ld : float): # to check if correct
    K = X.T @ X
    Z = K - delta * np.identity(np.shape(K)[0])
    T = np.diag(t)
    Lt = T @ Z @ T + delta * np.identity(np.shape(K)[0])
    Lt_inv = np.linalg.inv(Lt)
    return 2*np.diag(Lt_inv @ T @ np.square(K) @ (T @ Lt_inv @ T @ Z - np.identity(np.shape(K)[0]))) + ld*np.ones(np.shape(K)[0])


def SLS(X, n1, n2, s, ld, step, maxiter, delta = 10, M = 5, display = False, tol = 1e-5, stochastic = True, info = False):
    """
    Parameters:
        X (numpy.ndarray): The matrix to approximate.
        n1 (int): The number of rows of the matrix.
        n2 (int): The number of columns of the matrix.
        s (int): The number of columns to select.
        ld (float): The regularization parameter.
        step (float): The step size for the gradient descent.
        maxiter (int): The maximum number of iterations.
        delta (float): The delta parameter.
        M (int): The number of random vectors to sample.
        display (bool): Whether to display the plots.
        tol (float): The tolerance for the stopping criterion.
        stochastic (bool): Whether to use stochastic gradient descent.
    """
    tw = lambda w : np.ones(n2)  - np.exp(-w*w) 
    K = np.dot(X.T,X)
    t = np.ones(n2)/2
    # print('this is t', t)
    w = np.sqrt(-np.log(1-t))
    n = 0
    if display:
        f_to_min = []
        f_LHS = []
        values = []
        grad = []
    z_m = np.zeros((M,n2))
    for i in range(M):
        z_m[i] = np.random.choice([-1, 1], size=n2) 
    f_best = float('inf')
    ti = tw(w)
    n_best = 0
    while (n<maxiter):
        z_m = np.zeros((M,n2))
        for i in range(M):
            z_m[i] = np.random.choice([-1, 1], size=n2) 
        phi = np.zeros(n2)
        if stochastic:
            for z in z_m:
                a = np.dot(K,z)
                T = np.diag(ti)
                Z = K - delta*np.identity(n2)
                L = np.dot(np.dot(T,Z),T) + delta*np.identity(n2)
                try:
                    b = np.linalg.solve(L, ti*a)
                except np.linalg.LinAlgError:
                    rand_indx = np.random.choice(range(n2), s, replace=False)
                    print('error in grad SLS')
                    return (rand_indx, function_objectif_C( X, X[:,rand_indx]), 0, ti)
                # b = np.linalg.inv(L)@(ti* a)
                phi += b*np.dot(Z,(ti*b)) - a*b 
            grad_f_t = 2*phi/M + ld*np.ones(n2)
        else:
            try:
                # print('this is ti', ti)
                grad_f_t = grad_SLS(X, ti, delta, ld)
            except np.linalg.LinAlgError:
                rand_indx = np.random.choice(range(n2), s, replace=False)
                print('error in grad SLS')
                return (rand_indx, function_objectif_C( X, X[:,rand_indx]), 0, ti)
        grad_f_t_w = grad_f_t*(2*w*np.exp(-w*w))
        i = np.random.randint(0, n2)
        # print(n)
        # w = w - step*grad_f_t*(2*w*np.exp(-w*w))
        w -= step*grad_f_t_w
        n+=1
        ti = tw(w)
        if (display):
            indices = np.argpartition(ti, -s)[-s:]
            values.append(function_objectif_C( X, X[:,indices]))
            f_to_min.append(SLS_f_to_min(X, ti, delta, ld, n2))
            # f_LHS.append(SLS_f_LHS(X, ti, delta, ld, n2))
            # grad.append(np.linalg.norm(2*phi/M + ld*np.ones(n2)))

        try:
            fk = SLS_f_to_min(X, ti, delta, ld, n2)
        except np.linalg.LinAlgError:
            print('error in SLS_f_to_min')
            return (np.random.choice(range(n2), s, replace=False), function_objectif_C( X, X[:,np.random.choice(range(n2), s, replace=False)]), 0, ti)
        if fk + tol < f_best:
            f_best = fk
            t = ti
            n_best = n
        if (n-n_best>10):
            # print(f'critere d arret after {n} iterations')
            break
        ### end of new critere d'arret
    if display:
        # plt.plot([0,iter], [value, value], label = 'avg choice')
        plt.plot(range(len(f_to_min)), f_to_min, label = 'f to minimize')
        # plt.plot(range(len(f_LHS)), f_LHS, label = 'f LHS')
        plt.plot(range(len(values)), values, label = 'f discrete' )
        # plt.plot(range(len(grad)), grad, label =  'grad')
        plt.legend()
        plt.grid()
        plt.show()
    t = ti
    indices = np.argpartition(t, -s)[-s:]
    if info:
        return (indices,function_objectif_C( X, X[:,indices]), n, ti)
    else:
        return indices

def SLS_speed(X, n1, n2, s, ld, step, maxiter, mod, delta = 10, M = 5, display = False, tol = 1e-5, stochastic = True):
    """
    Parameters:
        X (numpy.ndarray): The matrix to approximate. 
        n1 (int): The number of rows of the matrix.
        n2 (int): The number of columns of the matrix.
        s (int): The number of columns to select.
        ld (float): The regularization parameter.
        step (float): The step size for the gradient descent.
        maxiter (int): The maximum number of iterations.
        delta (float): The delta parameter.
        M (int): The number of random vectors to sample.
        display (bool): Whether to display the plots.
        tol (float): The tolerance for the stopping criterion.
        stochastic (bool): Whether to use stochastic gradient descent.
    """
    t_array = np.zeros((maxiter//mod + 1,n2))
    Timer_array = np.zeros(maxiter//mod+1)
    n_array = np.zeros(maxiter//mod+1)
    tw = lambda w : np.ones(n2)  - np.exp(-w*w) 
    K = X.T@X
    t = np.ones(n2)/2
    w = np.sqrt(-np.log(1-t))
    n = 0
    if display:
        f_to_min = []
        f_LHS = []
        values = []
        grad = []
    z_m = np.zeros((M,n2))
    for i in range(M):
        z_m[i] = np.random.choice([-1, 1], size=n2) 
    # f_best_N = list(np.ones(11)*float('inf'))
    f_best = float('inf')
    # time_old = time.time()
    ti = tw(w)
    timer = time.time()
    while (n<maxiter):
        z_m = np.zeros((M,n2))
        for i in range(M):
            z_m[i] = np.random.choice([-1, 1], size=n2) 
        phi = np.zeros(n2)
        # print('shape of K', np.shape(K))
        # print('shape of z_m', np.shape(z_m))
        if stochastic:
            for z in z_m:
                a = K@z
                T = np.diag(ti)
                Z = K - delta*np.identity(n2)
                L = T@Z@T + delta*np.identity(n2)
                try:
                    b = np.linalg.solve(L, ti*a)
                except np.linalg.LinAlgError:
                    rand_indx = np.random.choice(range(n2), s, replace=False)
                    print('error in grad SLS')
                    return (rand_indx, function_objectif_C( X, X[:,rand_indx]), maxiter, ti)
                # b = np.linalg.inv(L)@(ti* a)
                phi += b*(Z@(ti*b)) - a*b 
            grad_f_t = 2*phi/M + ld*np.ones(n2)
        else:
            try:
                grad_f_t = grad_SLS(X, ti, delta, ld)
            except np.linalg.LinAlgError:
                rand_indx = np.random.choice(range(n2), s, replace=False)
                print('error in grad SLS')
                return (rand_indx, function_objectif_C( X, X[:,rand_indx]), maxiter, ti)
        grad_f_t_w = grad_f_t*(2*w*np.exp(-w*w))
        i = np.random.randint(0, n2)
        # print(n)
        # w = w - step*grad_f_t*(2*w*np.exp(-w*w))
        w -= step*grad_f_t_w
        n+=1
        ti = tw(w)
        if (display):
            indices = np.argpartition(ti, -s)[-s:]
            values.append(function_objectif_C( X, X[:,indices]))
            f_to_min.append(SLS_f_to_min(X, ti, delta, ld, n2))
            # f_LHS.append(SLS_f_LHS(X, ti, delta, ld, n2))
            # grad.append(np.linalg.norm(2*phi/M + ld*np.ones(n2)))
        # if n==30:
        #     break
        if (n%mod==0):
            Timer_array[n//mod] = time.time() - timer
            t_array[n//mod] = ti
            n_array[n//mod] = n
            # print('this is n', n)
        try:
            fk = SLS_f_to_min(X, ti, delta, ld, n2)
            # print('this is fk')
        except np.linalg.LinAlgError:
            print('error in SLS_f_to_min')
            return (np.random.choice(range(n2), s, replace=False), function_objectif_C( X, X[:,np.random.choice(range(n2), s, replace=False)]), 0, ti)
        # fk = SLS_f_to_min(X, ti, delta, ld, n2)
        if fk + tol < f_best:
            f_best = fk
            t = ti
            n_best = n
        if (n-n_best>10):
            # print(f'critere d arret after {n} iterations')
            break
        ### end of new critere d'arret
    if display:
        # plt.plot([0,iter], [value, value], label = 'avg choice')
        plt.plot(range(len(f_to_min)), f_to_min, label = 'f to minimize')
        # plt.plot(range(len(f_LHS)), f_LHS, label = 'f LHS')
        plt.plot(range(len(values)), values, label = 'f discrete' )
        # plt.plot(range(len(grad)), grad, label =  'grad')
        plt.legend()
        plt.grid()
        plt.show()
        
    Error_array = np.zeros(maxiter//mod +1)
    # print(t)
    # print('fin') 
    for i in range(n//mod):
        indices = np.argpartition(t_array[i], -s)[-s:]
        Error_array[i] = function_objectif_C( X, X[:,indices])
    return (Timer_array, n_array, Error_array)





def local_mu_maximum_volume(A):  #ALGO 1 PAN 2000
    m, n = A.shape
    assert m < n and np.linalg.matrix_rank(A) == m, "Matrix A must have full row rank with m < n"
    
    # Initialization
    P, L, U = lu(A)  # LU factorization with partial pivoting
    # print(np.shape(U))
    # print(np.shape(L))
    # print(np.shape(P))
    # print(A)
    Gamma = P.T
    PI = np.identity(n)
    U1 = U[:, :m]
    U2 = U[:, m:]
    # print(Gamma@A@PI)
    # print(L@U)
    
    j = m-1# correct 
    iter = 0
    # print('new U \n ', U)
    # print('new L \n ', L)
    # print('new LU \n ', L@U)
    c = 0
    while j > 0:#and c<10:
        c+=1
        # print('\niteration : ', iter)
        # print('j : ', j)
        iter+=1
        if j < m:
            U1 = U[:, :m]
            U2 = U[:, m:]
            PI1 = PI[:, :m]
            PI2 = PI[:, m:]

            # print('this is U1 0\n', U1)
            # Step 2: Permute the jth and mth columns of U1 
            
            # Get the number of rows and columns
            
            # Permute the second column to the last position #OK
            # print('this is U \n', U)
            Pii = np.identity(m)
            I = np.identity(m)
            In = np.identity(n)
            P = np.roll(Pii, -1, axis=1)
            Pii1 = np.concatenate((I[:, :j-1], P[:,j-1:-1], I[:, j-1:j]) , axis=1)
            # print('this is Pii1 \n', Pii1)

            U1 =  np.dot(U1, Pii1)
            PI1 = np.dot(PI1, Pii1) #here

            U = np.concatenate((U1,U2),axis=1)

            PI = np.concatenate((PI1,PI2),axis=1)


            # print('P@Gamma@A@PI \n', Gamma@A@PI)
            # print('PLP@PU \n', L@U)

            PLP = L
            PU = U
            P1_Tr = np.identity(m)
            P2_Tr = np.identity(m)

            # print('this is U permuted \n', PU)
            for i in range(j-1, m-1):
                # print(i)
                if (np.abs(PU[i,i]) < np.abs(PU[i+1,i]) * 0.01): 
                    # permute the row i,i+1 of PU 
                    # print('row permutation')
                    # print('this is PU \n', PU)
                    P = np.identity(m)
                    P[i,:] = np.identity(m)[i+1,:]
                    P[i+1,:] = np.identity(m)[i,:]
                    PU = P@PU
                    PLP = P@PLP@P
                    # Gamma = P.T@Gamma
                    Gamma = P@Gamma

                f = PLP[i,i+1]
                a = PU[i+1,i]
                b = PU[i,i]
                P = np.identity(m)
                # print('a \n', a)    
                # print('b \n', b)
                t = a/b
                P_tr1 = np.identity(m)
                P_tr1[i,i] = 1/(1+f*t)
                P_tr1[i,i+1] = -f
                P_tr1[i+1,i] = t/(1+f*t)
                P_tr1[i+1,i+1] = 1
                P_tr2 = np.identity(m)
                P_tr2[i,i] = 1
                P_tr2[i+1,i+1] = 1/(1+f*t)
                P_tr2[i+1,i] = -t/(1+f*t)
                P_tr2[i,i+1] = f
                # print('this is P_tr1 \n', P_tr1)
                # print('this is P_tr2 \n', P_tr2)
                # print('P_tr2@U \n', P_tr2@PU)
                PU = P_tr2@PU
                P1_Tr = P1_Tr@P_tr1
                P2_Tr = P_tr2@P2_Tr
            L = PLP@P1_Tr
            # U = P2_Tr@PU
            U = PU

            # print('this is U triangular \n', U)
            # print('this is L \n', L)

            # print('P1@P2 \n', P1@P2)


        ####### UNTIL HERE IT IS CORRECT !!!
        #PIVOTING DONE
        # Step 3: Check the condition
        # print('this is U \n', U)
        u_mm = U[m-1, m-1] # correct 
        u_m_row = U[m-1, m-1:] # correct 
        # print('this is u_m row \n', u_m_row)
        l = np.argmax(np.abs(u_m_row)) + m # correct 
        # print(l)
        # l = 5 # to delete 

        u_ml = U[m-1, l-1]  # correct 
        mu = 1.1 # sur above 1 but how much ??
        if mu*np.abs(u_mm) >= np.abs(u_ml): # correct 
            j = j-1 # correct 
        else : 
            # print("else")
            # Interchange mth and lth columns 
            # print('Interchange ', m,' and ',l,' columns ')
            # print('U \n', U)
            # print('PI \n', PI)
            Pml = np.identity(n)
            # l+=1
            Pml[:,m-1] = np.identity(n)[:,l-1]
            Pml[:,l-1] = np.identity(n)[:,m-1]
            U = np.dot(U, Pml)
            PI = np.dot(PI, Pml)
            # print('U \n', U)
            # print('PI \n', PI)
            j = m - 1 # j = m-1
        # break
    U1 = U[:, :m]
    U2 = U[:, m:]
    # print('PI \n', PI)
    # print('GAPI \n', Gamma@A@PI)
    # print('LU \n', L@U)
    # print('U \n', U)
    # print('L \n', L)    
    # print('Gamma \n', Gamma)
    # print('PI \n', PI)
    # print('API \n', A@PI)
    # print('GTLU \n', Gamma.T@L@U)
    # print('A \n', A)
    # print('GTLUPIT \n', Gamma.T@L@U@PI.T)
    # print('API \n', A@PI)
    # print('GTLU1 \n', Gamma.T@L@U1)#@PI.T)
    return Gamma, L, U1, U2, PI

def Boutsidis_Mahoney_Drineas( X, s, c=1, advanced_leveraging_score = True): # en pause pour l'instant
    """
    Function to select columns based on leverage scores. Then Apply the local mu maximum volume algorithm (from PAN)
    
    Parameters:
    X (numpy.ndarray): m x n matrix.
    s (int): Number of columns to select.
    c (int): factor in front of the rank parameter slog(s)
    advanced_leveraging_score (bool): If True, use the advanced leverage scores.    

    Returns:
    numpy.ndarray: m x c matrix with c columns from X.
    """
    # p = np.zeros(int(s*np.log(s)))
    n1, n2 = np.shape(X)
    V_s = calculate_right_eigenvectors_k_svd(X,s)
    # V_s = Q
    # Hard = True
    p = np.zeros(n2)
    p_simp = np.zeros(n2)
    # print('shape of X', np.shape(X))
    # print('s', s)
    (U, Sigma, VT) = svds(X, s)
    Vs = VT.T
    if (advanced_leveraging_score == True):
        for i in range(n2):
            # print('Vs[i] \n', Vs[i])
            term1 = norm(Vs[i])**2/(2*s)
            term2 = (norm(X[:,i])**2 - norm((X@Vs@Vs.T)[:,i])**2)/(2*(norm(X, ord='fro')**2 - norm(X@Vs@Vs.T, ord='fro')**2))
            p[i] = term1 + term2
        # print('p \n', p)
    else : 
        for i in range(n2):
            p[i] = (np.linalg.norm(Vs[i, :])**2) / s


    # c = c*int(s*np.log(s))
    # c = 2*s
    if c == 1:
        c = 10*s
    # print('p \n', p)
    ##### end of the initial set up 
    ##### Randomized stage
    selected_indices = []
    ind = []
    scalling_array = []
    for i in range(n2):
        # print('c * p[i] \n', c * p[i])
        scaling = min(1, c * p[i])
        scalling_array.append(scaling)
        # a = np.random.choice([1, 0], p=[scaling, 1-scaling])
        if scaling == 1 or scaling > np.random.rand():
            ind.append(i)
            selected_indices.append((i, 1/np.square(scaling)))
    C = X[:, ind]
    # print('scaling \n', scalling_array)
    # print('indices \n', ind)
    current_column = 0
    c_tilde = len(selected_indices)
    print('c_tilde \n', c_tilde)
    S1 = np.zeros((n2, c_tilde))
    for (i, scaling) in selected_indices:
        S1[i, current_column] = 1
        current_column += 1
    # print('S1 \n', S1)
    D1 = np.diag(np.array(selected_indices)[:,1])
    # print('np.shape of D1 \n', np.shape(D1))
    # print('D1 \n', D1)
    # print('Vt \n', Vs.T)
    # print('Vs.T@S1@D1 \n', Vs.T@S1@D1)
    # c = s*np.log(s)
    # c = 10*s
    print(np.shape(Vs.T@S1@D1))
    (m, n) = np.shape(Vs.T@S1@D1)
    if ((m >= n) or (np.linalg.matrix_rank(Vs.T@S1@D1) != m)):
        print(m, n)
        print(np.linalg.matrix_rank(Vs.T@S1@D1))
        print('loop')
        return Boutsidis_Mahoney_Drineas(X, s, c)
    try : 
        assert m < n and np.linalg.matrix_rank(Vs.T@S1@D1) == m, "Matrix A must have full row rank with m < n"
    except AssertionError:
        print('np.shape of D1 \n', np.shape(D1))
        print('np.shape of S1 \n', np.shape(S1))
        print('np.shape of Vs \n', np.shape(Vs))
        print('np.shape of Vs.T@S1@D1 \n', np.shape(Vs.T@S1@D1))
        print('Vs.T@S1@D1 \n', Vs.T@S1@D1)
        print('svd of Vs.T@S1@D1 \n', np.linalg.svd(Vs.T@S1@D1))
    # Second Phase
    Gamma, L, U1, U2, PI = local_mu_maximum_volume(Vs.T@S1@D1)
    VS1D1S2_ = np.linalg.inv(Gamma)@L@U1@PI[:,:s].T
    k = np.where(np.sum(np.abs(VS1D1S2_), axis=0) != 0)[0]
    S2 = np.zeros((c_tilde, len(k)))
    for i,j in zip(k,range(len(k))):
        S2[i, j] = 1
    # print('S2 \n', S2)
    C = X@S1@S2
    s = np.where(S1@S2 == 1)[0]
    return (C, function_objectif_C(X, C))

def Uniform_sampling(A, s):
    """
    Function to select columns based on uniform sampling.
    
    Parameters:
    A (numpy.ndarray): m x n matrix.
    s (int): Number of columns to select.
    
    Returns:
    numpy.ndarray: m x s matrix with s columns from A.
    """
    m, n = A.shape
    indexA = np.random.choice(n, s, replace=False)
    # print('indexA \n', indexA)
    return (A[:, indexA], function_objectif_C(A, A[:, indexA]))