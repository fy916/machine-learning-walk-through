"""
Author: Feiyang Wang
COMP3055 Machine Learning Coursework

This is the entry of this coursework solution, all functions are encapsulated and organized in the following files:
task1.py, task2.py, task3_net.py, task3_utils.py, utils.py

Please check if all the above files are in the same folder, and simply run this main.py will generate all the outputs
requested by the coursework issue sheet.

Please also note that for task1 and task2 8000 training data is used, which could lead to slow calculation.
The task 2.1 and 2.3 may take more than 500 seconds to produce the cross validation for one model, tested on Intel i7-10700
if you want to use smaller data, please change the parameters for fetch_dataset function after import

Meanwhile, all the CNNs in task 3 are tested on RTX 3070 8GB without errors,
for smaller VRAM the training may generate errors, please use CPU mode

Hope you enjoy using this program!
"""

from utils import fetch_dataset, Evaluation_Statistic, plot_SVM_RBF_Cross_Validation
from task1 import run_task1
from task2 import linearSVM_cross_validation, linearSVM_test, SVM_RBF_Cross_Validation, SVM_RBF_Test
from task3_utils import prep_data, train_CNN, test_CNN
from task3_net import CNN_1, CNN_2, CNN_3, CNN_4, CNN_5

import numpy as np

# Prepare the dataset
# if you want to use the full dataset for task 1 and task 2, please set if_full_dataset to true
# if if_full_dataset is set to false, the subset of data will be used, which is the parameter for fetch_dataset function
if_full_dataset = False
raw_directory, subset_directory = fetch_dataset('data/', train_size = 8000, val_size = 1000, test_size = 1000)


if if_full_dataset:
    data = np.load(raw_directory, allow_pickle=True)
else:
    data = np.load(subset_directory, allow_pickle=True)

X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']



print("\n\n#######################################################################")
print("################################ TASK 1 ###############################")
print("#######################################################################")
# Task 1:
# Apply 20%, 40%, 60%, 80%, 100% PCA to reduce the original input features

# keep 20% input features, 20% * 3072 = 615 dimensions
x_train_20_flat, x_test_20_flat, pca20 = run_task1(dimensions = 615, X_train = X_train, X_test = X_test, reconstruct = True)
# keep 40% input features, 40% * 3072 = 1230 dimensions
x_train_40_flat, x_test_40_flat, pca40 = run_task1(dimensions = 1230, X_train = X_train, X_test = X_test, reconstruct = True)
# keep 60% input features, 60% * 3072 = 1845 dimensions
x_train_60_flat, x_test_60_flat, pca60 = run_task1(dimensions = 1845, X_train = X_train, X_test = X_test, reconstruct = True)
# keep 20% input features, 80% * 3072 = 2460 dimensions
x_train_80_flat, x_test_80_flat, pca80 = run_task1(dimensions = 2460, X_train = X_train, X_test = X_test, reconstruct = True)
# keep 100% input features, 100% * 3072 = 3072 dimensions
x_train_100_flat, x_test_100_flat, pca100 = run_task1(dimensions = 3072, X_train = X_train, X_test = X_test, reconstruct = True)
# keep all input features, simply pre-process the input data
x_train_original_flat, x_test_original_flat = run_task1(dimensions = None, X_train = X_train, X_test = X_test, reconstruct = False)




print("\n\n#######################################################################")
print("############################### TASK 2.1 ##############################")
print("#######################################################################")
# Task 2.1:
# Apply linear SVM to the training data.
# Do 5-fold cross validation to train and validate models using the input feature vectors from task 1.


# create an evaluator used for data analysis
evaluator = Evaluation_Statistic("PCA")

# apply linear SVM to the training data and perform cross validation for every input features
svm_20, evaluator = linearSVM_cross_validation("PCA 20%", x_train_20_flat, y_train, evaluator)
svm_40, evaluator = linearSVM_cross_validation("PCA 40%", x_train_40_flat, y_train, evaluator)
svm_60, evaluator = linearSVM_cross_validation("PCA 60%", x_train_60_flat, y_train, evaluator)
svm_80, evaluator = linearSVM_cross_validation("PCA 80%", x_train_80_flat, y_train, evaluator)
svm_100, evaluator = linearSVM_cross_validation("PCA 100%", x_train_100_flat, y_train, evaluator)
svm_original, evaluator = linearSVM_cross_validation("Original", x_train_original_flat, y_train, evaluator)

# plot the result of cross validation of different PCA feature dimensions
evaluator.plot_accuracy_for_cross_validation()




print("\n\n#######################################################################")
print("############################### TASK 2.2 ##############################")
print("#######################################################################")
# Task 2.2:
# Use both the train and test data separately to test the Linear SVM models generated in task 2.1
# which are fitted using different feature vectors from task 1.
#
# Compute the precision, recall, f1 values for each class, check for overfitting
# and calculate the avg precision, recall, f1 and overall accuracy for these models in task 2.1.
#
# Finally, Plot figures of all the results.


# create evaluators used for data analysis
evaluator_train = Evaluation_Statistic("PCA","train")
evaluator_test = Evaluation_Statistic("PCA","test")

# evaluate every model based on train and test data seperately, and plot figures for the test result for each input
evaluator_train, evaluator_test = linearSVM_test(
    "PCA 20%", x_train_20_flat, y_train, x_test_20_flat, y_test, svm_20, evaluator_train, evaluator_test)

evaluator_train, evaluator_test = linearSVM_test(
    "PCA 40%", x_train_40_flat, y_train, x_test_40_flat, y_test, svm_40, evaluator_train, evaluator_test)

evaluator_train, evaluator_test = linearSVM_test(
    "PCA 60%", x_train_60_flat, y_train, x_test_60_flat, y_test, svm_60, evaluator_train, evaluator_test)

evaluator_train, evaluator_test = linearSVM_test(
    "PCA 80%", x_train_80_flat, y_train, x_test_80_flat, y_test, svm_80, evaluator_train, evaluator_test)

evaluator_train, evaluator_test = linearSVM_test(
    "PCA 100%", x_train_100_flat, y_train, x_test_100_flat, y_test, svm_100, evaluator_train, evaluator_test)

evaluator_train, evaluator_test = linearSVM_test(
    "Original", x_train_original_flat, y_train, x_test_original_flat, y_test, svm_original, evaluator_train, evaluator_test)

print("======================================================================")
print("Task 2.2.2 Overall Accuracy And Avg Precision, Recall, F1")
print("for each input features are plotted and saved in /figs")

# plot the precision, recall, f1 for each class; also the overall accuracy, avg precision, recall, f1 for each model
evaluator_train.plot_all_precision()
evaluator_train.plot_all_recall()
evaluator_train.plot_all_f1()
evaluator_train.plot_all_accuracy()

evaluator_test.plot_all_precision()
evaluator_test.plot_all_recall()
evaluator_test.plot_all_f1()
evaluator_test.plot_all_accuracy()
print("======================================================================")






print("\n\n#######################################################################")
print("############################### TASK 2.3 ##############################")
print("#######################################################################")
# Task 2.3:
# Apply SVM with RBF kernel with different hyperparameter C values to the training data.
# Do 5-fold cross validation to train and validate models using the input data.


list_SVM_res = [] # the list that stores the scores for cross validating each model with different c value
svc_list = [] # the list that stores the model
C_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 1000, 10000]
# run the training and cross validation one by one
for C in C_values:
    svc, score = SVM_RBF_Cross_Validation(x_train_original_flat, y_train, C)
    list_SVM_res.append(score) # store the scores
    svc_list.append(svc) # store the model
averages = [np.mean(array) for array in list_SVM_res] # calculate the average score for each model
# plot the cross validation average score figure, the reason that does not show the every fold results is because
# there are too much c values which will be hard to read from the graph.
plot_SVM_RBF_Cross_Validation(C_values, averages)





print("\n\n#######################################################################")
print("############################### TASK 2.4 ##############################")
print("#######################################################################")
# Task 2.4:
# Use both the train and test data separately to test SVM with RBF kernel models with different hyperparameter
# C values in task 2.3 which are fitted from the original input from task 1.
#
# Compute the precision, recall, f1 values for each class, check for overfitting
# and calculate the avg precision, recall, f1 and overall accuracy for these models in task 2.3.
#
# Finally, plot figures of all the results.


# create evaluators used for data analysis
evaluator_train = Evaluation_Statistic("C", "train")
evaluator_test = Evaluation_Statistic("C", "test")

# run the test on each model that with different C values
for i in range(len(C_values)):
    evaluator_train, evaluator_test = SVM_RBF_Test(
        x_train_original_flat, y_train, x_test_original_flat, y_test, C_values[i], svc_list[i], evaluator_train, evaluator_test)

print("======================================================================")
print("Task 2.4.2 Overall Accuracy And Avg Precision, Recall, F1")
print("for each C values are plotted and saved in /figs")

# plot the precision, recall, f1 for each class; also the overall accuracy, avg precision, recall, f1 for each model
evaluator_train.plot_all_precision()
evaluator_train.plot_all_recall()
evaluator_train.plot_all_f1()
evaluator_train.plot_all_accuracy()

evaluator_test.plot_all_precision()
evaluator_test.plot_all_recall()
evaluator_test.plot_all_f1()
evaluator_test.plot_all_accuracy()
print("======================================================================")





print("\n\n#######################################################################")
print("############################### TASK 3 ################################")
print("#######################################################################")
# Task 3
# Design and implement CNNs to recognize and classify the images in the dataset

# pre-defined parameters
BATCH_SIZE = 64
num_epochs = 50
lr = 1e-4

# if set to false, 8000 train data, 1000 validate data and 1000 test data will be used, which is consistent with task 1
# this can be modified by changing the parameter in the prep_data function as follows
# if set to true, 40000 training data, 10000 validation data and 10000 test data will be used
full_dataset = True

# create evaluators used for data analysis
if full_dataset:
    evaluator_train = Evaluation_Statistic("CNN_Full", "train")
    evaluator_test = Evaluation_Statistic("CNN_Full", "test")
else:
    evaluator_train = Evaluation_Statistic("CNN_Partial", "train")
    evaluator_test = Evaluation_Statistic("CNN_Partial", "test")

train_loader, val_loader, test_loader = prep_data(
    full_dataset, BATCH_SIZE, train_size = 8000, val_size = 1000, test_size = 1000)

# if you want to test the training, validation, and test dataset for the model after every epoch of training,
# set it to True. It will slow down the training but is clear for testing purposes.
test_after_every_epoch = True


# train and test the CNN
CNN_type = "1" # use CNN 1
file_loc = train_CNN(CNN_1(), num_epochs, lr, train_loader, val_loader, test_loader, full_dataset, CNN_type,
                     test_after_epoch = test_after_every_epoch)
evaluator_train, evaluator_test = test_CNN(CNN_1(), file_loc, evaluator_train, evaluator_test, "CNN_"+CNN_type,
                                           train_loader, test_loader)

CNN_type = "2" # use CNN 2
file_loc = train_CNN(CNN_2(), num_epochs, lr, train_loader, val_loader, test_loader, full_dataset, CNN_type,
                     test_after_epoch = test_after_every_epoch)
evaluator_train, evaluator_test = test_CNN(CNN_2(), file_loc, evaluator_train, evaluator_test, "CNN_"+CNN_type,
                                           train_loader, test_loader)

CNN_type = "3" # use CNN 3
file_loc = train_CNN(CNN_3(), num_epochs, lr, train_loader, val_loader, test_loader, full_dataset, CNN_type,
                     test_after_epoch = test_after_every_epoch)
evaluator_train, evaluator_test = test_CNN(CNN_3(), file_loc, evaluator_train, evaluator_test, "CNN_"+CNN_type,
                                           train_loader, test_loader)

CNN_type = "4" # use CNN 4
file_loc = train_CNN(CNN_4(), num_epochs, lr, train_loader, val_loader, test_loader, full_dataset, CNN_type,
                     test_after_epoch = test_after_every_epoch)
evaluator_train, evaluator_test = test_CNN(CNN_4(), file_loc, evaluator_train, evaluator_test, "CNN_"+CNN_type,
                                           train_loader, test_loader)

CNN_type = "5" # use CNN 5
file_loc = train_CNN(CNN_5(), num_epochs, lr, train_loader, val_loader, test_loader, full_dataset, CNN_type,
                     test_after_epoch = test_after_every_epoch)
evaluator_train, evaluator_test = test_CNN(CNN_5(), file_loc, evaluator_train, evaluator_test, "CNN_"+CNN_type,
                                           train_loader, test_loader)


# plot the precision, recall, f1 for each class; also the overall accuracy, avg precision, recall, f1 for each model
evaluator_train.plot_all_precision()
evaluator_train.plot_all_recall()
evaluator_train.plot_all_f1()
evaluator_train.plot_all_accuracy()

evaluator_test.plot_all_precision()
evaluator_test.plot_all_recall()
evaluator_test.plot_all_f1()
evaluator_test.plot_all_accuracy()