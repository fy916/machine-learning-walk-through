"""
Author: Feiyang Wang
COMP3055 Machine Learning Coursework

This file contains the functions used in main.py
for Task 1
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from utils import save_figs


def applyPCA(train_original_data, test_original_data, components, reconstruct, scaler_train, scaler_test):
    """
        This function applies PCA to the input train and test data, with a specified number of components.
        The function also has a parameter to reconstruct the original data and save the reconstruction results.
        The input train and test data are firstly scaled using StandardScaler.
        The function returns the transformed train & test data, and the fitted PCA model.

        Parameters:
        train_original_data: the training data with shape (n_samples, n_features)
        test_original_data: the test data with shape (n_samples, n_features)
        components: the number of principal components to use. If set to 1, default PCA will be applied
        reconstruct: a boolean value indicating whether to reconstruct the image
        scaler_train: the StandardScaler fit to the training data
        scaler_test: the StandardScaler fit to the test data

        Returns:
        x_train_trans: the transformed training data
        x_test_trans: the transformed test data
        pca: the PCA model fitted from the training data
    """
    print("=======================================================================")
    print("Task 1, PCA Start, Keep "+ str(int(components/3072*100)) +" % Dimensions")

    if components == 1: # use default PCA
        pca = PCA()
    else:               # use set components value
        pca = PCA(n_components=components)
    x_train_trans =  pca.fit_transform(train_original_data) # fit data for the model
    x_test_trans = pca.transform(test_original_data)        # transform the original data
    print(f'Total number of components used after PCA : {pca.n_components_}')
    print('Information kept: ', sum(pca.explained_variance_ratio_)*100, '%')
    print('Noise variance: ', pca.noise_variance_)
    print(f'train_dataset_transformed shape : {x_train_trans.shape}')
    print(f'test_dataset_transformed shape : {x_test_trans.shape}')

    if reconstruct: # reconstruct the image from transformed data vector
        x_train_reconstruct = pca.inverse_transform(x_train_trans)
        x_train_reconstruct = scaler_train.inverse_transform(x_train_reconstruct)

        x_test_reconstruct = pca.inverse_transform(x_test_trans)
        x_test_reconstruct = scaler_test.inverse_transform(x_test_reconstruct)

        plt.figure()    # plot the reconstructed train example image
        for i in range(6):
            plt.subplot(1,6,i+1)
            plt.imshow((np.reshape(x_train_reconstruct[i],(32,32,3))*255).astype(np.uint8))
        file_loc = save_figs("figs/X_reconstruct_train") # save the figure
        plt.savefig(file_loc, bbox_inches='tight', dpi=500)
        plt.show()

        plt.figure()    # plot the reconstructed test example image
        for i in range(6):
            plt.subplot(1,6,i+1)
            plt.imshow((np.reshape(x_test_reconstruct[i],(32,32,3))*255).astype(np.uint8))
        file_loc = save_figs("figs/X_reconstruct_test")
        plt.savefig(file_loc, bbox_inches='tight', dpi=500) # save the figure
        plt.show()
        print("Reconstructed figures are saved in the folder: figs/")

    print("============================== PCA END ================================")
    return x_train_trans, x_test_trans, pca


def run_task1(dimensions, X_train, X_test, reconstruct = True):
    """
    This is function for task 1. It preprocesses the input and return the result of the task 1 data features.
    The input data is first divided by 255 to scale it to the range [0, 1].
    Then it is reshaped to a 2D array with shape (n_samples, n_features).
    Finally, it is scaled using StandardScaler.

    Parameters:
    dimensions: the number of principal components to use for PCA. If set to None, no PCA is applied.
    X_train: the training data, a 3D array with shape (n_samples, 32, 32, 3)
    X_test: the test data, a 3D array with shape (n_samples, 32, 32, 3)
    reconstruct: a boolean value indicating whether to reconstruct the image

    Returns:
    If no PCA is required, directly returns the scaled training data and the scaled test data,
    two 2D array with shape (n_samples, n_features)
    If dimensions is set to a positive integer, PCA is applied.
    returns the transformed training and test data, two 2D array with shape (n_samples, n_components)
    pca: the PCA model fitted from the training data
    """

    # scale the data
    X_train = X_train/255
    X_test = X_test/255

    # reshape to (n_samples, n_features)
    X_train = np.reshape(X_train, (-1,3072))
    X_test = np.reshape(X_test, (-1,3072))

    # scale the data
    scaler_train = StandardScaler()
    scaler_test = StandardScaler()
    X_train = scaler_train.fit_transform(X_train)
    X_test = scaler_test.fit_transform(X_test)


    # return the data and pca model based on requirements
    if dimensions == None:
        return X_train, X_test
    else:
        return applyPCA(X_train, X_test, dimensions, reconstruct, scaler_train, scaler_test)

