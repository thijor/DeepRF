#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)

Script to test DeepRF on empirical data using a CPU and passing individual voxels through the model to
make it most comparable to CFpRF.
"""

import os
import numpy as np
from scipy.stats import pearsonr
import time

import stimulus as stim
import data_synthetic as ds
import data_empirical as de
import deeprf

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

def run(root="/project/2420084.01", subject=0, cat_dim="time", n_layers=50, n_dropout=1000):

    # Load stimuli
    stimuli = stim.BensonStimulus(os.path.join(root, "resources"))
    stimuli.resample_stimulus(0.5)

    # Initialize data generators
    train_dataset = de.BensonDataset(stimuli, root, subject, "train", cat_dim)
    valid_dataset = de.BensonDataset(stimuli, root, subject, "test", cat_dim)
    n_voxels = len(train_dataset)
    train_generator = ds.SyntheticDataGenerator(stimuli, n_voxels, (1, 30, 30), "train", cat_dim)
    valid_generator = ds.SyntheticDataGenerator(stimuli, n_voxels, (1, 40, 40), "test", cat_dim)

    # Set up model
    if cat_dim == "time":
        num_in_channels = 1
    else:
        num_in_channels = 6
    if n_layers == 50:
        model = deeprf.get_resnet50(num_in_channels=num_in_channels)
    elif n_layers == 34:
        model = deeprf.get_resnet34(num_in_channels=num_in_channels)
    elif n_layers == 18:
        model = deeprf.get_resnet18(num_in_channels=num_in_channels)
    else:
        raise Exception("Unknown n_layers:", n_layers)

    # Set up tester
    tester = deeprf.Tester(model, path=os.path.join(root, "derivatives"), suffix=cat_dim)
    tester.load_model()

    # Set bounds
    h_bound = (-2, 2)
    s_bound = (1/stimuli.pixperdeg, stimuli.radius_deg)
    x_bound = (-stimuli.radius_deg, stimuli.radius_deg)
    y_bound = (-stimuli.radius_deg, stimuli.radius_deg)

    # Pre-allocate memory for results
    y_train = np.zeros((n_voxels, 4), dtype="float32") # predicted parameters train (and used for valid)
    r_train = np.zeros((n_voxels,), dtype="float32") # squared Pearson's correlation between predicted train data and observed train data
    r_valid = np.zeros((n_voxels,), dtype="float32") # squared Pearson's correlation between predicted valid data and observed valid data
    ctime = np.zeros((n_voxels,), dtype="float32") # time it took to train the model

    y_train_dropout = np.zeros((n_voxels, 4, n_dropout), dtype="float32") # predicted parameters train (and used for valid)
    ctime_dropout = np.zeros((n_voxels,), dtype="float32") # time it took to train the model

    # Loop individual voxels
    for i in range(n_voxels):

        # Generate train data
        x = train_dataset.get_item(i)

        # Apply the model
        start_time = time.time()
        y_train[i, :] = tester.test_model(x)
        ctime[i] = time.time() - start_time

        # Apply with dropout for uncertainty estimation
        if n_dropout > 0:
            start_time = time.time()
            for j in range(n_dropout):
                y_train_dropout[i, :, j] = tester.test_model(x, test_time_dropout=True)
            ctime_dropout[i] = time.time() - start_time

        # Post-process outputs
        y_train[i, 0] = np.min([h_bound[1], np.max([h_bound[0], y_train[i, 0]])])
        y_train[i, 1] = np.min([s_bound[1], np.max([s_bound[0], y_train[i, 1]])])
        y_train[i, 2] = np.min([x_bound[1], np.max([x_bound[0], y_train[i, 2]])])
        y_train[i, 3] = np.min([y_bound[1], np.max([y_bound[0], y_train[i, 3]])])

        # Generate prediction of train data
        y = train_generator.generate_prediction(*y_train[i, :])

        # Compute explained variance of train data (might be overfitted)
        try:
            r_train[i] = pearsonr(x.flatten(), y.flatten())[0]**2
        except:
            print("warning: r could not be estimated with parameters: ", y_train[i, :])
            r_train[i] = 0

        # Generate valid data
        x = valid_dataset.get_item(i)

        # Generate prediction of the valid data using trained parameters
        y = valid_generator.generate_prediction(*y_train[i, :])

        # Compute explained variance of valid data
        try:
            r_valid[i] = pearsonr(x.flatten(), y.flatten())[0]**2
        except:
            print("warning: r could not be estimated with parameters: ", y_train[i, :])
            r_valid[i] = 0

        # Print some statistics
        print("subject: {} ({:d})  voxel {:5d}/{} ctime: {:6.3f} ({:6.3f}) + {:6.2f} ({:6.2f}) train: {:.2f} ({:.2f}) valid: {:.2f} ({:.2f})".format(
        	train_dataset.subject, subject, 1+i, n_voxels, ctime[i], np.mean(ctime[:1+i]),
            ctime_dropout[i], np.mean(ctime_dropout[:1+i]), r_train[i], np.mean(r_train[:1+i]), r_valid[i], np.mean(r_valid[:1+i])))

    # Make output folder if it does not already exist
    if not os.path.exists(os.path.join(root, "derivatives", model.name+"_"+cat_dim)):
        os.makedirs(os.path.join(root, "derivatives", model.name+"_"+cat_dim))

    # Save results
    np.savez(os.path.join(root, "derivatives", model.name+"_"+cat_dim, "results_empirical_{}.npz".format(train_dataset.subject)), 
        y_train=y_train, r_train=r_train, r_valid=r_valid, ctime=ctime, y_train_dropout=y_train_dropout, ctime_dropout=ctime_dropout)

if __name__ == "__main__":
    root = "/project/2420084.01"
    #root = "/Users/jordythielen/2420084.01"
    n_subjects = 29
    for i_subject in range(n_subjects):
        run(root=root, subject=i_subject, cat_dim="chan", n_layers=50, n_dropout=0)
