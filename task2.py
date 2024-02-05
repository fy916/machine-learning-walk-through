"""
Author: Feiyang Wang
COMP3055 Machine Learning Coursework

This file contains the functions used in main.py
for Task 2
"""
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time


def linearSVM_cross_validation(tag, training_data, training_label, evaluator):
    """
    This function is for the task 2.1.
    It applies linear SVM to the training data.
    It also does 5-fold cross validation to train and validate models using the input feature vectors from task 1.

    Parameters:
    tag (str): a string representing the name of the data
    training_data: a 2D array containing the training data
    training_label: a 1D array containing the labels corresponding to the training data
    evaluator: imported from utils.py, an encapsulated class to record the training label and predicted label,
    and generate the final analysis from the result.

    Returns:
    evaluator: the updated evaluator for result analysis.
    """
    print("======================================================================")
    print("TASK 2.1.1, Apply linear SVM using "+tag+" data")

    # perform scale on the data
    scaler = StandardScaler()
    scaler.fit(training_data)
    training_data = scaler.transform(training_data)

    start_time = time.perf_counter()
    # create a linear SVM model
    svm = SVC(kernel='linear')
    # apply linear SVM to the training data
    svm.fit(training_data, training_label)
    # note the processing time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f'Done! Time elapsed for training linear svm: {elapsed_time:.6f} seconds')

    print("----------------------------------------------------------------------")
    print("TASK 2.1.2, 5-Fold Cross validating linear SVM using "+tag+" data")
    start_time = time.perf_counter()
    # set 5-fold cross validation to train and validate models
    kf = KFold(n_splits=5)
    # train and validate the model, and get the 5-fold cross validation score
    scores = cross_val_score(SVC(kernel='linear'), training_data, training_label, cv=kf, n_jobs=-1)
    # append the scores to the evaluator object for analysis
    evaluator.accuracy_for_cross_validation_append(tag, scores)
    # note the processing time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("Type: ", tag)
    print("Scores:", scores)
    print(f"Cross Validation on training data, Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f'Done! Time elapsed for cross validation: {elapsed_time:.6f} seconds')
    print("======================================================================")
    return svm, evaluator


def linearSVM_test(tag, training_data, training_label, testing_data, testing_label, svm, evaluator_train, evaluator_test):
    """
    This function is for the task 2.2.

    It uses both the train and test data separately to test the Linear SVM models generated in task 2.1
    which are fitted using different feature vectors from task 1.

    It also computes the precision, recall, f1 values for each class, checks for overfitting
    and calculates the avg precision, recall, f1 and overall accuracy for these models in task 2.1.

    Finally, it plots figures of all the results.


    Parameters:
    tag (str): a string representing the name of the data
    training_data: a 2D array containing the train data
    training_label: a 1D array containing the labels corresponding to the train data
    testing_data: a 2D array containing the test data
    testing_label: a 1D array containing the labels corresponding to the test data
    svm: the model used for testing
    evaluator_train, evaluator_test: imported from utils.py, an encapsulated class to record the ground truth
                                     and predicted label for the train data and test data,
                                     and generate the final analysis from the result.

    Returns:
    evaluator_train, evaluator_test: the updated evaluator for result analysis.
    """

    print("======================================================================")
    print("Task 2.2.1 Precision, Recall, F1 For Each Class For "+ tag)

    # perform scale on the test data
    scaler_testing = StandardScaler()
    scaler_testing.fit(testing_data)
    testing_data = scaler_testing.transform(testing_data)

    # predict the result using the trained model
    predicted_train = svm.predict(training_data)
    predicted_test = svm.predict(testing_data)

    # append the predicted result and the ground truth to the evaluator for analysis
    print("For Train Data: ")
    evaluator_train.accuracy_calculate_append(training_label, predicted_train, tag)
    print("----------------------------------------------------------------------")
    print("For Test Data: ")
    evaluator_test.accuracy_calculate_append(testing_label, predicted_test, tag)
    print("======================================================================")
    return evaluator_train, evaluator_test


def SVM_RBF_Cross_Validation(training_data, training_label, cvalue):
    """
    This function is for the task 2.3.
    It applies SVM with RBF kernel with different hyperparameter C values to the training data.
    It also does 5-fold cross validation to train and validate models using the input data.

    Parameters:
    training_data: a 2D array containing the training data
    training_label: a 1D array containing the labels corresponding to the training data
    cvalue: the value of the hyperparameter C for the SVM with RBF kernel model

    Returns:
    svc: the trained model
    scores: the 5-fold cross validation scores
    """
    print("======================================================================")
    print("TASK 2.3.1, Apply SVM with RBF kernel to the train data SVM C Value "+ str(cvalue))

    # perform scale on the data
    scaler = StandardScaler()
    scaler.fit(training_data)
    training_data = scaler.transform(training_data)

    start_time = time.perf_counter()

    # create the SVM model with an RBF kernel and the given C value
    svc = SVC(kernel='rbf', C=cvalue)
    # apply SVM with RBF kernel to the training data
    svc.fit(training_data, training_label)

    # note the processing time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f'Time elapsed for training SVM with RBF kernel: {elapsed_time:.6f} seconds')

    print("----------------------------------------------------------------------")
    print("TASK 2.3.2, 5-Fold Cross validating SVM with RBF kernel SVM C Value "+ str(cvalue))
    start_time = time.perf_counter()

    # set 5-fold cross validation to train and validate models
    kf = KFold(n_splits=5)
    # train and validate the model, and get the 5-fold cross validation score
    scores = cross_val_score(SVC(kernel='rbf', C=cvalue), training_data, training_label, cv=kf, n_jobs=-1)

    # note the processing time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("Scores:", scores)
    print(f"Cross Validation on training data, Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f'Time elapsed: {elapsed_time:.6f} seconds')
    print("======================================================================")

    return svc, scores

def SVM_RBF_Test(training_data, training_label, testing_data, testing_label, cvalue, svc, evaluator_train, evaluator_test):
    """
    This function is for the task 2.4.

    It uses both the train and test data separately to test SVM with RBF kernel models with different hyperparameter
    C values in task 2.3 which are fitted from the original input from task 1.

    It also computes the precision, recall, f1 values for each class, checks for overfitting
    and calculates the avg precision, recall, f1 and overall accuracy for these models in task 2.3.

    Finally, it plots figures of all the results.

    Parameters:
    training_data: a 2D array containing the training data
    training_label: a 1D array containing the labels corresponding to the training data
    testing_data: a 2D array containing the test data
    testing_label: a 1D array containing the labels corresponding to the test data
    cvalue: the C parameter for the SVM model.
    svc: the model used for testing
    evaluator_train, evaluator_test: imported from utils.py, an encapsulated class to record the ground truth
                                     and predicted label for the train data and test data,
                                     and generate the final analysis from the result.

    Returns:
    evaluator_train, evaluator_test: the updated evaluator for result analysis.
    """
    print("======================================================================")
    print("Task 2.4.1 Precision, Recall, F1 For Each Class For SVM C Value "+ str(cvalue))

    # perform scale on the test data
    scaler_testing = StandardScaler()
    scaler_testing.fit(testing_data)
    testing_data = scaler_testing.transform(testing_data)

    # predict the result using the trained model
    predicted_train = svc.predict(training_data)
    predicted_test = svc.predict(testing_data)

    # append the predicted result and the ground truth to the evaluator for analysis
    print("For Train Data: ")
    evaluator_train.accuracy_calculate_append(training_label, predicted_train, cvalue)
    print("----------------------------------------------------------------------")
    print("For Test Data: ")
    evaluator_test.accuracy_calculate_append(testing_label, predicted_test, cvalue)
    print("======================================================================")
    return evaluator_train, evaluator_test