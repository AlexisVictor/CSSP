import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
# import CSSP as cs
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

def create_random_submatrix(A, k):
    """
    Creates a submatrix containing randomly chosen columns of A.

    Parameters:
        A (numpy.ndarray): The input matrix.
        k (int): The number of columns to select.

    Returns:
        numpy.ndarray: The submatrix containing k randomly chosen columns.
    """
    m, n = A.shape
    selected_columns = np.random.choice(n, k, replace=False)
    return A[:, selected_columns]

def create_sparse_column_matrix(n1, n2, s, std_dev):
    Xp = np.random.rand(n1, s)
    X = np.zeros((n1,n2))
    X[:, -s:] = Xp[:,:]
    X += np.random.normal(0, std_dev, size=(n1, n2))
    return X

def create_matrix_file(n, n1, n2):
    """
    Creates a n file containing each one random matrix of dimension n1, n2. The values of the matrix are uniformly distributed on [0,1]
    Parameters:
        n (int): The number of matrix file to generate.
        n1 (int): The number of lines.
        n2 (int): The number of columns 
    Returns:
        True
    """
    for i in range(n):
        file_path = "data/array_data_{}_{}_{}.txt".format(n1, n2, i)
        np.savetxt(file_path, np.random.rand(n1,n2))
    return True

def create_artificial_matrix(n_mat, n1, n2, s, std_dev):
    """
    Creates a n file containing each one random matrix of dimension n1, n2. The values of the matrix are uniformly distributed on [0,1]
    Parameters:
        n (int): The number of matrix file to generate.
        n1 (int): The number of lines.
        n2 (int): The number of columns 
    Returns:
        True
    """
    for i in range(n_mat):
        file_path = "data/array_data_normal_{}_{}_{}_{}_{}.txt".format(std_dev, n1, n2, s, i)
        np.savetxt(file_path, create_sparse_column_matrix(n1, n2, s, std_dev))
    return True


def matrix_from_file(i, n1, n2):
    """
    Retrieves a matrix from the specified file.
    Parameters:
        file_path (str): The path to the file containing the matrix.
    Returns:
        numpy.ndarray: The matrix loaded from the file.
    """
    file_path = "data/array_data_{}_{}_{}.txt".format(n1, n2, i)
    matrix = np.loadtxt(file_path)
    return matrix

def plot_dict_artificial_matrix_from_file(i, n1, n2, s, sigma):
    """
    Retrieves a matrix from the specified file.
    Parameters:
        file_path (str): The path to the file containing the matrix.
    Returns:
        numpy.ndarray: The matrix loaded from the file.
    """
    file_path = "data/array_data_normal_{}_{}_{}_{}_{}.txt".format(sigma, n1, n2, s, i)
    matrix = np.loadtxt(file_path)
    return matrix


def store_matrix(X, file_path):
    """
    Stores a matrix n1 by n2 in a file.
    Parameters:
        X (numpy.ndarray): The matrix to store.
        file_path (str): The path to the file where to store the matrix.
        Returns:
        True"""
    np.savetxt(file_path, X)
    return True

def plot_dict(results_dict_L_1O, n1, n2, method = '', save=False, filename='results_r.png', s=10):
    keys = list(results_dict_L_1O.keys())
    values = list(results_dict_L_1O.values())
    # print('this is keys : ', keys)
    # Extract the x and y values from the keys
    x_values = [key[0] for key in keys]
    y_values = [key[1] for key in keys]

    error_array = np.zeros((len(set(x_values)), len(set(y_values))))
    iter_array = np.zeros((len(set(x_values)), len(set(y_values))))
    Mask = np.zeros((len(set(x_values)), len(set(y_values))))
    for i, key in enumerate(keys):
        x_index = np.where(np.array(sorted(set(x_values))) == key[0])[0][0]
        y_index = np.where(np.array(sorted(set(y_values))) == key[1])[0][0]
        error_array[x_index, y_index] = values[i][0]
        iter_array[x_index, y_index] = values[i][1]
        Mask[x_index, y_index] = values[i][3]
    masked = np.ma.masked_where(Mask == 0, error_array)
    fig, ax = plt.subplots()
    cax = ax.imshow(masked, cmap='viridis', origin='lower')
    # plt.colorbar(label='Error')
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$ \lambda $')
    name = f'{method}, m={n1}, n = {n2}, s = {s}'
    ax.set_title(f'Approximation Factor')
    x_ = [x for x in sorted(set(x_values))]
    y_ = [x for x in sorted(set(y_values))]
    ax.set_xticks(np.arange(len(y_)))
    ax.set_xticklabels([f'{y:.1e}' for y in y_], rotation = 45)
    ax.set_yticks(np.arange(len(x_)))
    ax.set_yticklabels([f'{x:.1e}' for x in x_])

    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))

    cbar = fig.colorbar(cax)
    plt.tight_layout()

    # plt.show()
    if save:
        plt.savefig('results/' +filename+ name +'.svg')
        plt.close()
    else:   
        plt.show()
    return True 

# def save_dict(results_dict : dict, depth : int, filename : str):
#     """ 
#     Store a dict in a text file, the dict contains depth sub dict. 
#     """
#     with open(filename, 'w') as f:
#         f.write('filename:\n')
#         for key, value in results_dict.items():
#             f.write(f'{key}: {value}\n')
#     return None


# print(dict_fac['prof']['SLS'])
# save_dict(dict_fac, 3, 'test.txt')

def store_dict_in_file(dict_, depth, filename):
    """ 
    Store a dict in a text file, the dict contains depth sub dict. 
    Parameters:
        dict (dict): The dictionary to store.
        depth (int): The depth of sub-dictionaries.
        filename (str): The path to the file where to store the dictionary.
    Returns:
        None
    """
    with open(filename, 'w') as f:
        # f.write('filename:\n')
        write_dict_to_file(dict_, f, depth)
    return None

def write_dict_to_file(dict_, f, depth):
    """
    Helper function to recursively write the dictionary to the file.
    Parameters:
        dict (dict): The dictionary to write.
        f (file): The file object to write to.
        depth (int): The current depth of sub-dictionaries.
    Returns:
        None
    """
    for key, value in dict_.items():
        if isinstance(value, dict) and depth > 0:
            f.write(f'{key}:\n')
            write_dict_to_file(value, f, depth - 1)
        else:
            f.write(f'{key}: {value}\n')
    return None

def read_dict_from_file(filename):
    """
    Read a dictionary from a text file.
    Parameters:
        filename (str): The path to the file containing the dictionary.
    Returns:
        dict: The dictionary loaded from the file.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        dict = {}
        current_dict = dict
        current_key = None
        for line in lines:
            line = line.strip()
            if line.endswith(':'):
                current_key = line[:-1]
                current_dict[current_key] = {}
                current_dict = current_dict[current_key]
            else:
                print(line)
                print(line.split(': '))
                key, value = line.split(': ')
                current_dict[key] = value
    return dict




def store_dict_in_file_best_params(dict, filename):
    # Store the results into a file
    with open(filename, 'w') as f:
        f.write('filename:\n')
        for key, value in dict.items():
            f.write(f'Lambda: {key[0]}, Step: {key[1]}, error: {value[0]}, after {value[1]} iterations, used the stop criterion {value[2]}  \n')
        f.write('\n')
        min_error = min(dict.values())
        min_keys = [key for key, value in dict.items() if value == min_error]
        # print(min_keys)
        f.write(f'the parameters with the lowest error are : Lambda: {min_keys[0][0]}, step: {min_keys[0][1]}, error {dict[min_keys[0]][0]}, iteration {dict[min_keys[0]][1]}, the std deviation is {dict[min_keys[0]][2]}\n')
        # f.write(f'the parameters with the lowest error are : Lambda: {min_keys[0][0]}, step: {min_keys[0][1]}, error {dict[min_keys[0][0]][min_keys[0][1]]}\n')
        
    return None

def read_dict_best_params_from_file(filename): # still to check
    with open(filename, 'r') as f:
        data = f.readlines()[1:-2]
        results_dict = {}
        for line in data:
            line = line.split()
            # print(line)
            key = (float(line[1][:-1]), float(line[3][:-1]))
            # print(key)
            value = (float(line[5][:-1]), float(line[7][:-1]), float(line[-1]))
            # print(value)
            results_dict[key] = value
    return results_dict

def read_data_from_file(filename):
    """
    read a matrix from a file and standardize it
    Parameters:
        filename (str): The path to the file containing the matrix.
    Returns:
        numpy.ndarray: The matrix loaded from the file.
    """
    data = pd.read_csv(filename, header=None)
    data.replace('?', np.nan, inplace=True)
    data = data.astype(float)
    data.fillna(data.mean(), inplace=True)
    # standardize data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data