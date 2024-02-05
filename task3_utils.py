"""
Author: Feiyang Wang
COMP3055 Machine Learning Coursework

This file contains the tools for preparing data, training and testing the CNN network
for Task 3
"""

from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch
import os
import time


def prep_data(full_dataset, BATCH_SIZE, train_size = 8000, val_size = 1000, test_size = 1000):
    # define the transforms for the training and test datasets
    tranform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tranform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # set a seed for reproducibility
    torch.manual_seed(2021)

    # fetch the original train and test datasets
    train = torchvision.datasets.CIFAR10("data/", train=True, download=True, transform=tranform_train)
    test = torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=tranform_test)

    # calculate the data split if use smaller dataset
    dataset_size = []
    dataset_size.append(train_size)
    dataset_size.append(val_size)
    dataset_size.append(test_size)
    dataset_size.append(50000-train_size-val_size-test_size)

    # if full_dataset is set to false, the subset of data will be used
    if not full_dataset:
        train, val, test, remaining = random_split(train, dataset_size)
    else:
        train, val = random_split(train, [40000, 10000])

    #  train, val and test datasets to the dataloader
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=False)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


def train_CNN(model, num_epochs, lr, train_loader, val_loader, test_loader, full_dataset, CNN_type, test_after_epoch):
    # this function trains the CNN

    print("#######################################################################")
    print("----------------------------- START -----------------------------------")
    # set the model saving directory
    if full_dataset:
        print("Start Training CNN " + CNN_type + " Using Full Dataset! ")
        mark = "_full"
    else:
        print("Start Training CNN " + CNN_type + " Using Partial Dataset! ")
        mark = "_partial"

    file_name = "models/version_" + CNN_type + mark + ".pt"
    file_loss_name = "records/total_loss_" + CNN_type + mark + ".txt"
    file_val_acc_name = "records/total_val_accuracy_" + CNN_type + mark + ".txt"
    file_test_acc_name = "records/total_test_accuracy_" + CNN_type + mark + ".txt"
    file_train_acc_name = "records/total_train_accuracy_" + CNN_type + mark + ".txt"

    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('records'):
        os.makedirs('records')

    # create the log file
    open(file_loss_name, "w")
    open(file_val_acc_name, "w")
    open(file_test_acc_name, "w")
    open(file_train_acc_name, "w")

    # check if the GPU is available, if not use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # clean VRAM cache
    torch.cuda.empty_cache()

    # set the model to train model
    model.train()
    # move model to device
    model.to(device)

    # use adam optimizer, which is popular and stable
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train every epoch
    for epoch in range(num_epochs):
        # note the time
        start_time = time.perf_counter()
        loss_var = 0

        # go through the dataloader by batch
        for idx, (images, labels) in enumerate(train_loader):
            # move training data to device
            images = images.to(device=device)
            labels = labels.to(device=device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            scores = model(images)
            loss = criterion(scores, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            # update loss
            loss_var += loss.item()
            if (idx + 1) % 50 == 0:
                # print every 50 batch
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}] || Step [{idx + 1}/{len(train_loader)}] || Loss:{loss_var / len(train_loader)}')
        print(f"Loss at epoch {epoch + 1} || {loss_var / len(train_loader)}")
        overall_loss = loss_var / len(train_loader)

        # please note, the test of train, validation, and test dataset will be performed after every epoch for
        # testing purposes, and it will not influence the training process.
        # But it will slow the training down, you can set test_after_epoch to False if you do not want to test.
        if test_after_epoch:
            # set the model to evaluation mode
            model.eval()
            with torch.no_grad():
                correct = 0
                samples = 0
                # test the training dataset accuracy for the model after training this epoch, to check overfitting
                for idx, (images, labels) in enumerate(train_loader):
                    # move data to device
                    images = images.to(device=device)
                    labels = labels.to(device=device)
                    # predict using the model
                    outputs = model(images)
                    _, preds = outputs.max(1)
                    # compare for correctness
                    correct += (preds == labels).sum()
                    samples += preds.size(0)
                print(f"Train accuracy is {float(correct) / float(samples) * 100:.2f}% || Correctness is: {correct} out of {samples} samples")
                overall_train_accuracy = float(correct) / float(samples) * 100

                correct = 0
                samples = 0
                # test the validation dataset accuracy for the model after training this epoch, to check overfitting
                for idx, (images, labels) in enumerate(val_loader):
                    # move data to device
                    images = images.to(device=device)
                    labels = labels.to(device=device)
                    # predict using the model
                    outputs = model(images)
                    _, preds = outputs.max(1)
                    # compare for correctness
                    correct += (preds == labels).sum()
                    samples += preds.size(0)
                print(f"Val accuracy is {float(correct) / float(samples) * 100:.2f}% || Correctness is: {correct} out of {samples} samples")
                overall_val_accuracy = float(correct) / float(samples) * 100

                correct = 0
                samples = 0
                # test the testing dataset accuracy for the model after training this epoch, to check performance
                for idx, (images, labels) in enumerate(test_loader):
                    # move data to device
                    images = images.to(device=device)
                    labels = labels.to(device=device)
                    # predict using the model
                    outputs = model(images)
                    _, preds = outputs.max(1)
                    # compare for correctness
                    correct += (preds == labels).sum()
                    samples += preds.size(0)
                print(f"Test accuracy is {float(correct) / float(samples) * 100:.2f}% || Correctness is: {correct} out of {samples} samples\n")
                overall_test_accuracy = float(correct) / float(samples) * 100

                # save the model based on the given location
                torch.save(model.state_dict(), file_name)
                print("Model saved in path " + file_name)

                # log the training info
                with open(file_loss_name, 'a') as f:
                    f.write(f"{overall_loss}\n")

                with open(file_val_acc_name, 'a') as f:
                    f.write(f"{overall_val_accuracy}\n")

                with open(file_test_acc_name, 'a') as f:
                    f.write(f"{overall_test_accuracy}\n")

                with open(file_train_acc_name, 'a') as f:
                    f.write(f"{overall_train_accuracy}\n")

        # note the time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f'Time elapsed: {elapsed_time:.6f} seconds')
        epoch += 1

    print("Finished Training! Model is saved in path " + file_name)
    print("------------------------------ END ------------------------------------")
    return file_name


def test_CNN(cnn, file_loc, evaluator_train, evaluator_test, type, train_loader, test_loader):
    # this function tests the CNN performance
    print("#######################################################################")
    print("----------------------------- START -----------------------------------")
    print(" ----------------------- Testing " + type + " --------------------------------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    model = cnn
    # if the model is not passed directly, load from file
    if not file_loc == None:
        model.load_state_dict(torch.load(file_loc))  # loads the trained model
    # set the model to evaluation mode
    model.eval()
    # move model to device
    model.to(device)

    with torch.no_grad():
        correct = 0
        samples = 0
        predicted = []
        y_labels = []
        # test the accuracy of the train data to check the overfitting
        for idx, (images, labels) in enumerate(train_loader):
            # move training data to device
            images = images.to(device=device)
            labels = labels.to(device=device)
            # predict using the model
            outputs = model(images)
            _, preds = outputs.max(1)
            predicted.append(preds)
            y_labels.append(labels)
            # compare for correctness
            correct += (preds == labels).sum()
            samples += preds.size(0)

        # get the predicted results and ground truth
        predicted = torch.cat(predicted, dim=0)
        y_labels = torch.cat(y_labels, dim=0)
        # append the predicted result and the ground truth to the evaluator for analysis
        evaluator_train.accuracy_calculate_append(y_labels.cpu(), predicted.cpu(), type)

        print( f"Accuracy for train data is {float(correct) / float(samples) * 100:.2f}% || "
            f"Correctness is: {correct} out of {samples} samples")


        correct = 0
        samples = 0
        predicted = []
        y_labels = []
        # test the accuracy of the test data to check the performance
        for idx, (images, labels) in enumerate(test_loader):
            # move training data to device
            images = images.to(device=device)
            labels = labels.to(device=device)
            # predict using the model
            outputs = model(images)
            _, preds = outputs.max(1)
            predicted.append(preds)
            y_labels.append(labels)
            # compare for correctness
            correct += (preds == labels).sum()
            samples += preds.size(0)

        # get the predicted results and ground truth
        predicted = torch.cat(predicted, dim=0)
        y_labels = torch.cat(y_labels, dim=0)
        # append the predicted result and the ground truth to the evaluator for analysis
        evaluator_test.accuracy_calculate_append(y_labels.cpu(), predicted.cpu(), type)

        print(f"Accuracy for test data is {float(correct) / float(samples) * 100:.2f}% || "
              f"Correctness is: {correct} out of {samples} samples")
        print("Test result saved in /figs")
        print("------------------------------ END ------------------------------------")

        return evaluator_train, evaluator_test
