"""
Author: Feiyang Wang
COMP3055 Machine Learning Coursework

This file contains some utilities for the task 1 and task 2,
which includes the data processing and plotting
"""

from matplotlib import ticker
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
import numpy as np
import torchvision as torchvision
import os


def fetch_dataset(directory, train_size=8000, val_size=1000, test_size=1000):
    # This function is used to fetch and save datasets of CIFAR10 images and labels
    def get_data_targets(dataset):
        # this function is used to get the image and label in the torch dataset
        images = []
        labels = []
        for (image, label) in dataset:
            images.append(np.array(image))
            labels.append(np.array(label))
        return images, labels

    # set a seed for reproducibility
    torch.manual_seed(2021)

    # fetch the original train and test datasets
    original_train_dataset = torchvision.datasets.CIFAR10(directory, train=True, download=True)
    original_test_dataset = torchvision.datasets.CIFAR10(directory, train=False, download=True)

    # for the normal split, the training dataset is 40000 and validating dataset is 10000
    train_normal, val_normal = random_split(original_train_dataset, [40000, 10000])

    # get the data
    X_train, y_train = get_data_targets(train_normal)
    X_val, y_val = get_data_targets(val_normal)
    X_test = original_test_dataset.data
    y_test = original_test_dataset.targets

    full_directory = directory + 'CIFAR10_Raw.npz'
    # store the data in the directory
    np.savez(
        full_directory,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    # calculate the data split if use smaller dataset
    dataset_size = []
    dataset_size.append(train_size)
    dataset_size.append(val_size)
    dataset_size.append(test_size)
    dataset_size.append(50000 - train_size - val_size - test_size)

    # for the small split, the training dataset is 8000, validating dataset is 1000 and testing dataset is 1000
    # the test dataset is splited from the original training set as well.
    train_small, val_small, test_small, remaining = random_split(original_train_dataset, dataset_size)

    # get the data
    X_train, y_train = get_data_targets(train_small)
    X_val, y_val = get_data_targets(val_small)
    X_test, y_test = get_data_targets(test_small)

    partial_directory = directory + 'CIFAR10_Reduced.npz'
    # store the data in the directory
    np.savez(
        partial_directory,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    print("Raw Training Data Saved in " + full_directory)
    print("Reduced Training Data Saved in " + partial_directory)
    return full_directory, partial_directory


def save_figs(filename):
    # This function is used to generate a unique file name and save the figure, it will not overwrite the existing file,
    # but adding _1, _2, _3... to avoid conflicts
    file_name = filename
    file_extension = ".png"
    # get the folder path
    folder_path = os.path.dirname(filename)
    # create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # use a counter to count the existing duplicate files
    i = 1
    while os.path.exists(file_name + "_" + str(i) + file_extension):
        i += 1
    # return the non-duplicate file name
    return (file_name + "_" + str(i) + file_extension)


def plot_SVM_RBF_Cross_Validation(new_numbers, averages):
    # This function is used to plot the results of cross validation for SVM with an RBF kernel
    # get the x and y values for the plot
    x = new_numbers
    y = averages
    # find the index of the maximum value in the figure
    max_y_index = np.argmax(y)
    max_x = x[max_y_index]
    max_y = y[max_y_index]

    fig, ax = plt.subplots()
    ax.set_xscale('log')  # use a log scale x axis
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10, subs=[1.0]))
    ax.plot(x, y)  # plot the graph
    ax.scatter(max_x, max_y, color='red')  # find the max point and mark it using red
    ax.annotate(f'(x={max_x:.2f}, y={max_y:.2f})', xy=(max_x, max_y), xytext=(max_x + 1, max_y),
                arrowprops=dict(facecolor='black', arrowstyle='->'))  # annotate the max point
    # add labels and tittles
    plt.xlabel('C Value')
    plt.ylabel('Accuracy')
    title = "Average Accuracy of 5-Fold Cross Validation of Different C Value"
    plt.title(title)
    # save the figures
    file_loc = save_figs("figs/" + title)
    plt.savefig(file_loc, bbox_inches='tight', dpi=500)
    plt.show()


class Evaluation_Statistic():
    # this class is an evaluator which integrates the data analysis functions along with plotting functions
    def __init__(self, evaluation_type, data_type=None):
        self.type = evaluation_type  # the type can be "PCA", "C", or "CNN_Full" or "CNN_Partial"
        self.data_type = data_type  # the data applied to the model, can be "train" or "test"
        # store the accuracy, precision, recall, f1 values
        self.all_accuracy = []
        self.all_precision = []
        self.all_recall = []
        self.all_f1 = []
        # store the vector name, which can be different PCA name, or C values, or CNN names
        self.vec_name = []
        # labels for the CIFAR10 dataset
        self.class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def plot_for_each_feature_vector(self, precision, recall, f1, title):
        # This function plots a bar graph for the input showing its precision, recall and f1 value for each class
        bar_width = 0.2  # set bar width
        x_pos = range(len(self.class_labels))  # locate drawing anchor for precision bars
        plt.bar(x_pos, precision, width=bar_width, label='Precision')  # plot the bar
        x_pos = [x + bar_width for x in x_pos]  # locate drawing anchor for recall bars
        plt.bar(x_pos, recall, width=bar_width, label='Recall')  # plot the bar
        x_pos = [x + bar_width for x in x_pos]  # locate drawing anchor for F1 bars
        plt.bar(x_pos, f1, width=bar_width, label='F1')  # plot the bar
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # define legend placed next to the table

        plt.xticks(x_pos, self.class_labels, rotation=45)  # rotate the x labels for easy read
        plt.xlabel('Class')  # x y label
        plt.ylabel('Score')
        # set the title according to the data type
        if self.data_type == "train":
            title += " Predicting Train Data"
        else:
            title += " Predicting Test Data"
        plt.title(title)
        # save figures
        file_loc = save_figs("figs/" + title)
        plt.savefig(file_loc, bbox_inches='tight', dpi=500)
        plt.show()

    def plot_all(self, list_to_plot, title):
        # This function plots a bar graph for all models comparing their precision or recall or f1 value for each class
        # according to the caller requirements

        # create a dataframe with dict representation
        df = pd.DataFrame({self.vec_name[i]: list_to_plot[i] for i in range(len(self.vec_name))},
                          index=self.class_labels)
        # print the dataframe
        print(title)
        print(df, "\n")
        ax = df.plot.bar(rot=0, width=0.8, figsize=(10, 5))  # set figure size
        # set the title according to the data type

        if self.data_type == "train":
            title += " Predicting Train Data"
        else:
            title += " Predicting Test Data"

        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # define legend placed next to the table
        # save figures
        file_loc = save_figs("figs/" + title)
        plt.savefig(file_loc, bbox_inches='tight', dpi=500)
        plt.show()

    def plot_all_precision(self):
        # this function calls plot_all function to generate the figure showing
        # precisions for all models for each class
        if self.type == "PCA":
            self.plot_all(self.all_precision, "Precision for Each Class vs Feature Dimension")
        elif self.type == "C":
            self.plot_all(self.all_precision, "Precision for Each Class vs Different C Values")
        elif self.type == "CNN_Full":
            self.plot_all(self.all_recall, "Precision for Each Class vs Different CNN Versions Trained on Full Dataset")
        elif self.type == "CNN_Partial":
            self.plot_all(self.all_recall,
                          "Precision for Each Class vs Different CNN Versions Trained on Partial Dataset")

    def plot_all_recall(self):
        # this function calls plot_all function to generate the figure showing
        # recalls for all models for each class
        if self.type == "PCA":
            self.plot_all(self.all_recall, "Recall for Each Class vs Feature Dimension")
        elif self.type == "C":
            self.plot_all(self.all_recall, "Recall for Each Class vs Different C Values")
        elif self.type == "CNN_Full":
            self.plot_all(self.all_recall, "Recall for Each Class vs Different CNN Versions Trained on Full Dataset")
        elif self.type == "CNN_Partial":
            self.plot_all(self.all_recall, "Recall for Each Class vs Different CNN Versions Trained on Partial Dataset")

    def plot_all_f1(self):
        # this function calls plot_all function to generate the figure showing
        # f1 values for all models for each class
        if self.type == "PCA":
            self.plot_all(self.all_f1, "F1 for Each Class vs Feature Dimension")
        elif self.type == "C":
            self.plot_all(self.all_f1, "F1 for Each Class vs Different C Values")
        elif self.type == "CNN_Full":
            self.plot_all(self.all_recall, "F1 for Each Class vs Different CNN Versions Trained on Full Dataset")
        elif self.type == "CNN_Partial":
            self.plot_all(self.all_recall, "F1 for Each Class vs Different CNN Versions Trained on Partial Dataset")

    def plot_all_accuracy(self):
        # this function shows the average precision, recall and f1 values along with the overall accuracy
        # for all models for each class

        # stores the name, which can be PCA n%, or C values, or CNN name
        labels = self.vec_name
        precision_average = [sum(x) / len(x) for x in self.all_precision]  # get the average precision
        recall_average = [sum(x) / len(x) for x in self.all_recall]  # get the average recall
        f1_average = [sum(x) / len(x) for x in self.all_f1]  # get the average f1
        accuracy = self.all_accuracy  # get the overall accuracy

        # generate a header of the data
        header = [["avg_precision", "avg_recall", "avg_f1", "accuracy"]]

        # generate the data for average precision, recall and f1 values along with the overall accuracy
        data = [[precision_average[i], recall_average[i], f1_average[i], accuracy[i]] for i in
                range(len(precision_average))]
        # convert to dataframe
        df = pd.DataFrame(data, columns=header, index=labels)

        # Create a figure and a subplot
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))
        # plot the precision average and accuracy list
        plt.plot(precision_average, label='Precision')
        plt.plot(recall_average, label='Recall')
        plt.plot(f1_average, label='F1')
        plt.plot(accuracy, label='Accuracy')
        plt.xticks(range(len(labels)), labels)

        # add labels and title according to type
        if self.type == "PCA":
            plt.xlabel('PCA Feature Dimension')
            title = "Accuracy, Precision, Recall and F1 of Linear SVM vs Different Feature Dimensions"
        elif self.type == "C":
            plt.xlabel('C Value')
            title = "Accuracy, Precision, Recall and F1 of SVM with RBF kernel with Different C Values"
        elif self.type == "CNN_Full":
            plt.xlabel('CNN Versions Trained on Full Dataset')
            title = "Accuracy, Precision, Recall and F1 of Different CNN Versions Trained on Full Dataset"
        elif self.type == "CNN_Partial":
            plt.xlabel('CNN Version Trained on Partial Dataset')
            title = "Accuracy, Precision, Recall and F1 of Different CNN Versions Trained on Partial Dataset"

        # set the title according to the data type
        if self.data_type == "train":
            title += " Predicting Train Data"
            print("----------------------------------------------------------------------")
        else:
            title += " Predicting Test Data"
            print("----------------------------------------------------------------------")
        plt.title(title)  # plot title
        plt.legend()  # plot legend
        print(title)
        print(df, "\n")
        print("----------------------------------------------------------------------")
        # save figures
        file_loc = save_figs("figs/" + title)
        plt.savefig(file_loc, bbox_inches='tight', dpi=500)
        plt.show()

    def accuracy_calculate_append(self, ground_truth, predicted, title):
        # this function appends the ground truth and the predicted results to the class variables
        # and also plot the graph showing the precision, recall, and f1 score for each class
        accuracy = accuracy_score(ground_truth, predicted) # generate accuracy score
        precision = precision_score(ground_truth, predicted, average=None) # generate precision for each class
        recall = recall_score(ground_truth, predicted, average=None) # generate recall for each class
        f1 = f1_score(ground_truth, predicted, average=None) # generate f1 score for each class

        # plot the figure according to the type
        if self.type == "CNN_Full":
            self.plot_for_each_feature_vector(precision, recall, f1,
                                              "Precision, Recall, and F1 for Each Class of " + str(
                                                  title) + " Trained on Full Dataset")
        elif self.type == "CNN_Partial":
            self.plot_for_each_feature_vector(precision, recall, f1,
                                              "Precision, Recall, and F1 for Each Class of " + str(
                                                  title) + " Trained on Partial Dataset")
        else:
            self.plot_for_each_feature_vector(precision, recall, f1,
                                              "Precision, Recall, and F1 for Each Class of " + str(title))

        # Create a dataframe to display the scores
        scores_df = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1': f1}, index=range(10))
        print(scores_df)
        print("Accuracy: ", accuracy)

        # append the data to class variables
        self.all_accuracy.append(accuracy)
        self.all_precision.append(precision)
        self.all_recall.append(recall)
        self.all_f1.append(f1)
        self.vec_name.append(title)

        return accuracy, precision, recall, f1

    def accuracy_for_cross_validation_append(self, label, scores):
        # append the accuracy score of the cross validation result
        self.all_accuracy.append(scores)
        self.vec_name.append(label)

    def plot_accuracy_for_cross_validation(self):
        # plot the graph showing the accuracy of each fold in the cross-validation
        accuracy_list = self.all_accuracy
        vec_name = self.vec_name
        labels = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]

        # plot every fold accuracy for each model
        for i, accuracy in enumerate(accuracy_list):
            plt.plot(accuracy, label=f'Accuracy for input {vec_name[i]}')
            # add the labels to the x-axis
            plt.xticks(range(len(labels)), labels)

        # plot the x y label
        plt.xlabel('Cross Validation Folds')
        plt.ylabel('Accuracy')

        # add title according to type
        if self.type == "PCA":
            title = "Cross Validation of Different PCA Feature Dimensions"
        elif self.type == "C":
            title = "Cross Validation of Different C Values"

        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # add legend
        # save the figures
        file_loc = save_figs("figs/" + title)
        plt.savefig(file_loc, bbox_inches='tight', dpi=500)
        plt.show()
