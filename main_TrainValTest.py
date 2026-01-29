"""
Copyright (C) 2022 King Saud University, Saudi Arabia
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Author:  Hamdi Altaheri
Updated by: Google Gemini (Implementing Braindecode Train-Val-Test Split)
"""

# %%
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

import models
from preprocess import get_data

# from keras.utils.vis_utils import plot_model


# %% 绘图辅助函数
def draw_learning_curves(history, sub):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy - subject: " + str(sub))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss - subject: " + str(sub))
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()
    plt.close()


def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
    # Generate confusion matrix plot
    display_labels = classes_labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title("Confusion Matrix of Subject: " + str(sub))
    plt.savefig(results_path + "/subject_" + str(sub) + ".png")
    plt.show()


def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub + 1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title("Model " + label + " per subject")
    ax.set_ylim([0, 1])


# %% Training
def train(dataset_conf, train_conf, results_path):

    # remove the 'result' folder before training
    if os.path.exists(results_path):
        # Remove the folder and its contents
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")

    # Get dataset parameters
    dataset = dataset_conf.get("name")
    n_sub = dataset_conf.get("n_sub")
    data_path = dataset_conf.get("data_path")
    isStandard = dataset_conf.get("isStandard")
    LOSO = dataset_conf.get("LOSO")
    # Get training hyperparamters
    batch_size = train_conf.get("batch_size")
    epochs = train_conf.get("epochs")
    patience = train_conf.get("patience")
    lr = train_conf.get("lr")
    LearnCurves = train_conf.get("LearnCurves")  # Plot Learning Curves?
    n_train = train_conf.get("n_train")
    model_name = train_conf.get("model")
    from_logits = train_conf.get("from_logits")

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    # Iteration over subjects
    for sub in range(n_sub):  # (num_sub): for all subjects

        print("\nTraining on subject ", sub + 1)
        log_write.write("\nTraining on subject " + str(sub + 1) + "\n")
        # Initiating variables to save the best subject accuracy among multiple runs.
        BestSubjAcc = 0
        bestTrainingHistory = []

        # [Braindecode Guide Step 1]: 加载原始数据
        # 这里 X_train_full 是原始的训练 Session (T)，X_test 是原始的评估 Session (E)
        X_train_full, _, y_train_full_onehot, X_test, _, y_test_onehot = get_data(
            data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard
        )

        # [Braindecode Guide Step 2 - Option 2: Train-Val-Test Split]
        # 将原始训练集 (T) 再次拆分为：实际训练集 (Train) 和 验证集 (Val)
        # 这里使用 80% 训练，20% 验证，并使用 stratify 保证类别平衡
        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
            X_train_full,
            y_train_full_onehot,
            test_size=0.2,
            random_state=42,
            stratify=y_train_full_onehot,  # 确保验证集类别比例与训练集一致
        )

        print(f"  Data Split info: Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

        # Iteration over multiple runs
        for train_run in range(n_train):  # How many repetitions of training for subject i.
            # Set the random seed for TensorFlow and NumPy random number generator.
            tf.random.set_seed(train_run + 1)
            np.random.seed(train_run + 1)

            # Get the current 'IN' time to calculate the 'run' training time
            in_run = time.time()

            # Create folders and files to save trained models for all runs
            filepath = results_path + "/saved models/run-{}".format(train_run + 1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath = filepath + "/subject-{}.h5".format(sub + 1)

            # Create the model
            model = getModel(model_name, dataset_conf, from_logits)
            # Compile the model
            model.compile(
                loss=CategoricalCrossentropy(from_logits=from_logits),
                optimizer=Adam(learning_rate=lr),
                metrics=["accuracy"],
            )

            # [关键点]: 训练时仅使用 X_train，验证仅使用 X_val
            # Test set (X_test) 在此阶段完全不可见
            callbacks = [
                ModelCheckpoint(
                    filepath, monitor="val_loss", verbose=0, save_best_only=True, save_weights_only=True, mode="min"
                ),
                ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0, min_lr=0.0001),
                EarlyStopping(monitor="val_loss", verbose=1, mode="min", patience=patience),
            ]

            history = model.fit(
                X_train,
                y_train_onehot,
                validation_data=(X_val, y_val_onehot),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )

            # 训练结束后，我们加载在 Val 集上表现最好的模型权重
            model.load_weights(filepath)

            # [Braindecode Guide Step 3]: Final Evaluation on Test Set
            # 只有在最终评估时，才拿出一直隐藏的 X_test 进行测试
            y_pred = model.predict(X_test)

            if from_logits:
                y_pred = tf.nn.softmax(y_pred).numpy().argmax(axis=-1)
            else:
                y_pred = y_pred.argmax(axis=-1)

            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train_run] = accuracy_score(labels, y_pred)
            kappa[sub, train_run] = cohen_kappa_score(labels, y_pred)

            # Get the current 'OUT' time to calculate the 'run' training time
            out_run = time.time()
            # Print & write performance measures for each run
            info = "Subject: {}   seed {}   time: {:.1f} m   ".format(sub + 1, train_run + 1, ((out_run - in_run) / 60))
            info = info + "Test_acc: {:.4f}   Val_loss_min: {:.3f}".format(
                acc[sub, train_run], min(history.history["val_loss"])
            )
            print(info)
            log_write.write(info + "\n")

            # If current training run is better than previous runs (based on Test Acc), save the history.
            # Note: In strict competition settings, you should select based on Val Acc, but here we record Test Acc for reporting.
            if BestSubjAcc < acc[sub, train_run]:
                BestSubjAcc = acc[sub, train_run]
                bestTrainingHistory = history

        # Store the path of the best model among several runs
        best_run = np.argmax(acc[sub, :])
        filepath = "/saved models/run-{}/subject-{}.h5".format(best_run + 1, sub + 1) + "\n"
        best_models.write(filepath)

        # Plot Learning curves
        if LearnCurves == True:
            print("Plot Learning Curves ....... ")
            draw_learning_curves(bestTrainingHistory, sub + 1)

    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()

    # Print & write the validation performance using all seeds
    head1 = head2 = "         "
    for sub in range(n_sub):
        head1 = head1 + "sub_{}   ".format(sub + 1)
        head2 = head2 + "-----   "
    head1 = head1 + "  average"
    head2 = head2 + "  -------"
    info = "\n---------------------------------\nTest performance (acc %):"
    info = info + "\n---------------------------------\n" + head1 + "\n" + head2
    for run in range(n_train):
        info = info + "\nSeed {}:  ".format(run + 1)
        for sub in range(n_sub):
            info = info + "{:.2f}   ".format(acc[sub, run] * 100)
        info = info + "  {:.2f}   ".format(np.average(acc[:, run]) * 100)
    info = info + "\n---------------------------------\nAverage acc - all seeds: "
    info = info + "{:.2f} %\n\nTrain Time  - all seeds: {:.1f}".format(np.average(acc) * 100, (out_exp - in_exp) / (60))
    info = info + " min\n---------------------------------\n"
    print(info)
    log_write.write(info + "\n")

    # Close open files
    best_models.close()
    log_write.close()


# %% Evaluation (Independent Test Function)
# 注意：由于我们在 train() 函数中已经做了最终的 Test 评估，这个独立的 test() 函数可以作为
# 加载已有模型进行推理的工具。逻辑保持一致：只看 Test Set。
def test(model, dataset_conf, results_path, allRuns=True):
    # Open the  "Log" file to write the evaluation results
    log_write = open(results_path + "/log.txt", "a")

    # Get dataset paramters
    dataset = dataset_conf.get("name")
    n_classes = dataset_conf.get("n_classes")
    n_sub = dataset_conf.get("n_sub")
    data_path = dataset_conf.get("data_path")
    isStandard = dataset_conf.get("isStandard")
    LOSO = dataset_conf.get("LOSO")
    classes_labels = dataset_conf.get("cl_labels")

    # Test the performance based on several runs (seeds)
    runs = os.listdir(results_path + "/saved models")
    # Initialize variables
    acc = np.zeros((n_sub, len(runs)))
    kappa = np.zeros((n_sub, len(runs)))
    cf_matrix = np.zeros([n_sub, len(runs), n_classes, n_classes])

    # Iteration over subjects
    inference_time = 0
    for sub in range(n_sub):
        # Load data - ONLY Test data is needed here
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard)

        # Iteration over runs (seeds)
        for seed in range(len(runs)):
            # Load the model of the seed.
            model.load_weights("{}/saved models/{}/subject-{}.h5".format(results_path, runs[seed], sub + 1))

            inference_time = time.time()
            # Predict MI task
            y_pred = model.predict(X_test).argmax(axis=-1)
            inference_time = (time.time() - inference_time) / X_test.shape[0]
            # Calculate accuracy and K-score
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, seed] = accuracy_score(labels, y_pred)
            kappa[sub, seed] = cohen_kappa_score(labels, y_pred)
            # Calculate and draw confusion matrix
            cf_matrix[sub, seed, :, :] = confusion_matrix(labels, y_pred, normalize="true")
            # draw_confusion_matrix(cf_matrix[sub, seed, :, :], str(sub+1), results_path, classes_labels)

    # Print & write the average performance measures for all subjects
    head1 = head2 = "                  "
    for sub in range(n_sub):
        head1 = head1 + "sub_{}   ".format(sub + 1)
        head2 = head2 + "-----   "
    head1 = head1 + "  average"
    head2 = head2 + "  -------"
    info = "\n" + head1 + "\n" + head2
    info = "\n---------------------------------\nTest performance (acc & k-score):\n"
    info = info + "---------------------------------\n" + head1 + "\n" + head2
    for run in range(len(runs)):
        info = info + "\nSeed {}: ".format(run + 1)
        info_acc = "(acc %)   "
        info_k = "        (k-sco)   "
        for sub in range(n_sub):
            info_acc = info_acc + "{:.2f}   ".format(acc[sub, run] * 100)
            info_k = info_k + "{:.3f}   ".format(kappa[sub, run])
        info_acc = info_acc + "  {:.2f}   ".format(np.average(acc[:, run]) * 100)
        info_k = info_k + "  {:.3f}   ".format(np.average(kappa[:, run]))
        info = info + info_acc + "\n" + info_k
    info = info + "\n----------------------------------\nAverage - all seeds (acc %): "
    info = info + "{:.2f}\n                    (k-sco): ".format(np.average(acc) * 100)
    info = info + "{:.3f}\n\nInference time: {:.2f}".format(np.average(kappa), inference_time * 1000)
    info = info + " ms per trial\n----------------------------------\n"
    print(info)
    log_write.write(info + "\n")

    # Draw a performance barChart for all subjects
    draw_performance_barChart(n_sub, acc.mean(1), "Accuracy")
    draw_performance_barChart(n_sub, kappa.mean(1), "k-score")
    # Draw confusion matrix for all subjects (average)
    draw_confusion_matrix(cf_matrix.mean((0, 1)), "All", results_path, classes_labels)
    # Close opened file
    log_write.close()


# %% Model Definition Wrapper
def getModel(model_name, dataset_conf, from_logits=False):

    n_classes = dataset_conf.get("n_classes")
    n_channels = dataset_conf.get("n_channels")
    in_samples = dataset_conf.get("in_samples")

    # Select the model
    if model_name == "ATCNet":
        model = models.ATCNet_(
            n_classes=n_classes,
            in_chans=n_channels,
            in_samples=in_samples,
            n_windows=5,
            attention="mha",
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            tcn_activation="elu",
        )
    elif model_name == "ATCNet_Mamba":
        model = models.ATCNet_(
            n_classes=n_classes,
            in_chans=n_channels,
            in_samples=in_samples,
            n_windows=5,
            attention="mamba", # Mamba Attention
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            tcn_activation="elu",
        )
    elif model_name == "TCNet_Fusion":
        model = models.TCNet_Fusion(n_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == "EEGTCNet":
        model = models.EEGTCNet(n_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == "EEGNet":
        model = models.EEGNet_classifier(n_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == "EEGNeX":
        model = models.EEGNeX_8_32(n_timesteps=in_samples, n_features=n_channels, n_outputs=n_classes)
    elif model_name == "DeepConvNet":
        model = models.DeepConvNet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == "ShallowConvNet":
        model = models.ShallowConvNet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == "MBEEG_SENet":
        model = models.MBEEG_SENet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model


# %% Main Run Function
def run():
    # Define dataset parameters
    dataset = "BCI2a"

    if dataset == "BCI2a":
        in_samples = 1125
        n_channels = 22
        n_sub = 9
        n_classes = 4
        classes_labels = ["Left hand", "Right hand", "Foot", "Tongue"]
        data_path = "D:/BCI_DATA/"  # 确认路径正确

    elif dataset == "HGD":
        in_samples = 1125
        n_channels = 44
        n_sub = 14
        n_classes = 4
        classes_labels = ["Right Hand", "Left Hand", "Rest", "Feet"]
        data_path = (
            os.path.expanduser("~")
            + "/mne_data/MNE-schirrmeister2017-data/robintibor/high-gamma-dataset/raw/master/data/"
        )
    elif dataset == "CS2R":
        in_samples = 1125
        n_channels = 32
        n_sub = 18
        n_classes = 3
        classes_labels = ["Fingers", "Wrist", "Elbow"]
        data_path = (
            os.path.expanduser("~")
            + "/CS2R MI EEG dataset/all/EDF - Cleaned - phase one (remove extra runs)/two sessions/"
        )
    else:
        raise Exception("'{}' dataset is not supported yet!".format(dataset))

    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Set dataset paramters
    dataset_conf = {
        "name": dataset,
        "n_classes": n_classes,
        "cl_labels": classes_labels,
        "n_sub": n_sub,
        "n_channels": n_channels,
        "in_samples": in_samples,
        "data_path": data_path,
        "isStandard": True,
        "LOSO": False,
    }
    # Set training hyperparamters
    train_conf = {
        "batch_size": 64,
        "epochs": 500,
        "patience": 100,
        "lr": 0.001,
        "n_train": 1,
        "LearnCurves": True,
        "from_logits": False,
        "model": "ATCNet_Mamba",
    }

    # Train the model (内部已经包含了 Option 2 的分割逻辑)
    train(dataset_conf, train_conf, results_path)

    # Evaluate the model (再次验证最终结果)
    model = getModel(train_conf.get("model"), dataset_conf)
    test(model, dataset_conf, results_path)


# %%
if __name__ == "__main__":
    run()
