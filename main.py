
import time
import sys
import itertools
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from preprocess_data import prepare_data
from preprocess_data import prepare_data_foc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import to_categorical
from collections import defaultdict
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, roc_auc_score, roc_curve
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.multiclass import type_of_target


    
def create_model(train_x, train_y):
    num_classes = train_y.shape[1]

    model = Sequential()

    # Define the input layer explicitly
    model.add(Input(shape=(train_x.shape[1],)))

    # Add the rest of the layers
    model.add(Dense(15, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(45, activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def multilayer_perceptron(k=10):
    """
    This function performs multiclass classification with a multilayer perceptron.
    """
    train_x, test_x, train_y, test_y = prepare_data()

    # Convert labels to categorical (one-hot encoding)
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    num_classes = train_y.shape[1]

    # Setup k-fold cross-validation
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracy_scores = []
    times = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_proba = []
    
    # Loop over each fold
    for train_idx, val_idx in skfold.split(train_x, np.argmax(train_y, axis=1)):  # Adjusted for one-hot encoded labels
        # Use .iloc for robust position-based indexing
        train_fold_x, val_fold_x = train_x.iloc[train_idx], train_x.iloc[val_idx]
        train_fold_y, val_fold_y = train_y[train_idx], train_y[val_idx]

        # Create and compile the model
        model = create_model(train_fold_x, train_fold_y)
        
        # Train model
        start = time.time()
        history = model.fit(train_fold_x, train_fold_y, epochs=50, batch_size=30, validation_data=(val_fold_x, val_fold_y), verbose=0)
        end = time.time()
        
        # Evaluate model on the validation set
        y_pred_proba = model.predict(val_fold_x)
        all_y_pred_proba.extend(y_pred_proba)  # Collect probabilities
        y_true = np.argmax(val_fold_y, axis=1)
        all_y_true.extend(y_true)

        accuracy_scores.append(np.mean(np.argmax(y_pred_proba, axis=1) == y_true) * 100)
        times.append(end - start)

    # all_y_true = np.array(all_y_true)
    # all_y_pred_proba = np.array(all_y_pred_proba)

    if len(np.unique(all_y_true)) > 1:
        roc_auc = roc_auc_score(to_categorical(all_y_true, num_classes), all_y_pred_proba, multi_class='ovr')
        print(f"Mean ROC AUC: {roc_auc:.2f}")
    
    # print(classification_report(all_y_true, np.argmax(all_y_pred_proba, axis=1)))
    # plt.figure()
    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.title('Learning Curve')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    print("### MLP with K-fold Cross-validation ###\n")
    print(f"{k}-Fold Cross-validation Accuracies: {accuracy_scores}")
    print(f"Mean Accuracy = {np.mean(accuracy_scores):.2f}%")
    print(f"Mean Training Time = {np.mean(times):.2f} seconds")

    return np.mean(times),np.mean(accuracy_scores),roc_auc



def xgboost(k=10):
    """
    This function performs classification with XGBoost, handling cases with unseen classes in test data
    """
    train_x, test_x, train_y, test_y = prepare_data()
    if isinstance(train_y, pd.Series):
        train_y = train_y.values.reshape((train_y.shape[0], ))
    if isinstance(test_y, pd.Series):
        test_y = test_y.values.reshape((test_y.shape[0], )) 
    # Convert train_x to numpy array if it's a DataFrame
    if isinstance(train_x, pd.DataFrame):
        train_x = train_x.values

    clf = XGBClassifier(n_estimators=20)
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies = []
    start = time.time()
    for train_index, test_index in skfold.split(train_x, train_y):
        X_train_fold, X_test_fold = train_x[train_index], train_x[test_index]
        y_train_fold, y_test_fold = train_y[train_index], train_y[test_index]

        try:
            clf.fit(X_train_fold, y_train_fold)
            predictions = clf.predict(X_test_fold)
            fold_accuracy = accuracy_score(y_test_fold, predictions)
            accuracies.append(fold_accuracy)
        except ValueError as e:
            # print(f"Skipping fold due to error: {e}")
            continue

    end = time.time()
    if accuracies:
        mean_accuracy = 100 * np.mean(accuracies)
    else:
        mean_accuracy = float('nan')  # No valid fold was processed
    total_time = end - start

    print("### XGB with K-fold Cross-validation ###\n")
    print(f"{k}-Fold Cross-validation Accuracy: {accuracies}")
    print(f"Mean Accuracy = {mean_accuracy:.2f}%")
    print(f"Training and Validation lasted {total_time:.2f} seconds")

    return total_time, mean_accuracy

def logistic_regression(k=10):
    """
    This function performs classification with logistic regression using k-fold cross-validation
    """
    train_x, test_x, train_y, test_y = prepare_data()
    if isinstance(train_y, pd.Series):
        train_y = train_y.values.reshape((train_y.shape[0], ))
    if isinstance(test_y, pd.Series):
        test_y = test_y.values.reshape((test_y.shape[0], )) 
    clf = LogisticRegression(random_state=0, solver='liblinear', max_iter=300, multi_class='ovr')

    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    start = time.time()
    cv_results = cross_val_score(clf, train_x, train_y, cv=skfold, scoring='accuracy')
    end = time.time()

    time_ = end - start
    mean_accuracy = 100 * cv_results.mean()
    print("### Logistic Regression with K-fold Cross-validation ###\n")
    print(f"{k}-Fold Cross-validation Accuracy: {cv_results}")
    print("Mean Accuracy = %.2f" % mean_accuracy)
    print("Training and Validation lasted %.2f seconds" % time_)

    return time_, mean_accuracy


def random_forest(k=10):
    """
    This function performs classification with random forest.
    """
    # output=[]
    # for params in range(1,34):
    train_x, test_x, train_y, test_y = prepare_data()
    if isinstance(train_y, pd.Series):
        train_y = train_y.values.reshape((train_y.shape[0], ))
    if isinstance(test_y, pd.Series):
        test_y = test_y.values.reshape((test_y.shape[0], ))        
    clf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=1)

    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    start = time.time()
    cv_results = cross_val_score(clf, train_x, train_y, cv=skfold, scoring='accuracy')
    end = time.time()

    time_ = end - start
    mean_accuracy = 100 * cv_results.mean()
    print("### Random Forest with K-fold Cross-validation ###\n")
    print(f"{k}-Fold Cross-validation Accuracy: {cv_results}")
    print("Mean Accuracy = %.2f" % mean_accuracy)
    print("Training and Validation lasted %.2f seconds" % time_)

    return time_, mean_accuracy


def k_neighbors(k=10):
    """
    This function performs classification with k-neighbors algorithm using k-fold cross-validation
    """
    train_x, test_x, train_y, test_y = prepare_data()
    if isinstance(train_y, pd.Series):
        train_y = train_y.values.reshape((train_y.shape[0], ))
    if isinstance(test_y, pd.Series):
        test_y = test_y.values.reshape((test_y.shape[0], )) 

    clf = KNeighborsClassifier(n_neighbors=3)
    
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    start = time.time()
    cv_results = cross_val_score(clf, train_x, train_y, cv=skfold, scoring='accuracy')
    end = time.time()

    time_ = end - start
    mean_accuracy = 100 * cv_results.mean()
    print("### K-Nearest Neighbors with K-fold Cross-validation ###\n")
    print(f"{k}-Fold Cross-validation Accuracy: {cv_results}")
    print("Mean Accuracy = %.2f" % mean_accuracy)
    print("Training and Validation lasted %.2f seconds" % time_)
    clf.fit(train_x, train_y)

    # Evaluate the model on the test set
    test_accuracy = clf.score(test_x, test_y)

    # Print or return the test accuracy
    print("Test Accuracy = %.2f" % (100 * test_accuracy))

    return time_, mean_accuracy


def plot_confusion_matrix(cm, target_names, title, cmap=None, normalize=False):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def classify(k=10):
    x_train, x_test, y_train, y_test = prepare_data(iter=42)

    # Initialize the Random Forest Classifier
    # n_estimators is the number of trees in the forest, adjust as needed
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(x_train, y_train.ravel())
    # Fit the model on the training data

    # Predict the labels of the test set
    y_pred = rf_classifier.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Detailed classification report
    print(classification_report(y_test, y_pred))

def gradient_boosting(k=10):
    """
    This function performs classification with Gradient Boosting using k-fold cross-validation
    """
    train_x, test_x, train_y, test_y = prepare_data()
    unique_values = np.unique(test_y)
    print(unique_values)
    print(len(train_x),len(train_y),len(test_x),len(test_y))

    # Ensure train_y is a one-dimensional numpy array
    train_y = np.array(train_y).reshape(-1)  # reshape(-1) ensures it is one-dimensional

    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=75)
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    start = time.time()
    cv_results = cross_val_score(clf, train_x, train_y, cv=skfold, scoring='accuracy')
    end = time.time()

    time_ = end - start
    mean_accuracy = 100 * cv_results.mean()
    print("### Gradient Boosting with K-fold Cross-validation ###\n")
    print(f"{k}-Fold Cross-validation Accuracy: {cv_results}")
    print("Mean Accuracy = %.2f" % mean_accuracy)
    print("Training and Validation lasted %.2f seconds" % time_)
    clf.fit(train_x, train_y)

    # Evaluate the model on the test set
    test_accuracy = clf.score(test_x, test_y)
    y_pred = clf.predict(test_x)

    # Print or return the test accuracy
    print("Test Accuracy = %.2f" % (100 * test_accuracy))

    return time_, mean_accuracy

    

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def gradient_boosting1(k=10):
    """
    This function performs classification with Gradient Boosting using k-fold cross-validation
    """
    train_x, test_x, train_y, test_y = prepare_data()

    print(len(train_x), len(train_y), len(test_x), len(test_y))
    
    # Reshape train_y and test_y for multiclass classification
    lb = LabelBinarizer()
    train_y = lb.fit_transform(train_y.reshape(-1, 1))
    test_y_bin = lb.transform(test_y.reshape(-1, 1))

    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=75)

    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    start = time.time()
    cv_results = cross_val_score(clf, train_x, train_y.argmax(axis=1), cv=skfold, scoring='accuracy')
    end = time.time()

    time_ = end - start
    mean_accuracy = 100 * cv_results.mean()
    print("### Gradient Boosting with K-fold Cross-validation ###\n")
    print(f"{k}-Fold Cross-validation Accuracy: {cv_results}")
    print("Mean Accuracy = %.2f" % mean_accuracy)
    print("Training and Validation lasted %.2f seconds" % time_)
    clf.fit(train_x, train_y.argmax(axis=1))

    # Evaluate the model on the test set
    test_accuracy = clf.score(test_x, test_y_bin.argmax(axis=1))
    y_pred = clf.predict(test_x)
    print(test_y)
    print(y_pred)

    # Print test accuracy
    print("Test Accuracy = %.2f" % (100 * test_accuracy))

    # Compute confusion matrix
    cm = confusion_matrix(test_y, lb.inverse_transform(y_pred.reshape(-1, 1)))
    np.set_printoptions(precision=2)

    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=lb.classes_, title='Confusion Matrix')
    plt.show()

    return time_, mean_accuracy

def support_vector_machine(k=10):
    """
    This function performs classification with support vector machine
    """
    train_x, test_x, train_y, test_y = prepare_data()
    if isinstance(train_y, pd.Series):
        train_y = train_y.values.reshape((train_y.shape[0], ))
    if isinstance(test_y, pd.Series):
        test_y = test_y.values.reshape((test_y.shape[0], )) 

    classifier = SVC(kernel="rbf", C=10, gamma=1)

    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    start = time.time()
    cv_results = cross_val_score(classifier, train_x, train_y, cv=skfold, scoring='accuracy')
    end = time.time()

    time_ = end - start
    mean_accuracy = 100 * cv_results.mean()
    print("### Support Vector Machine with K-fold Cross-validation ###\n")
    print(f"{k}-Fold Cross-validation Accuracy: {cv_results}")
    print("Mean Accuracy = %.2f" % mean_accuracy)
    print("Training and Validation lasted %.2f seconds" % time_)

    return time_, mean_accuracy

def kmeans(num_iterations=20, n_clusters=2, n_init=10, max_iter=300, tol=1e-4):
    """
    This function performs clustering with kmeans and evaluates clustering performance.
    """
    clustering_scores = []
    train_accuracy = []
    test_accuracy = []
    start = time.time()
    for i in range(num_iterations):
        train_x, test_x, train_y, test_y = prepare_data(i)  # Pass iteration as random state

        # Run KMeans with different random states
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, 
                        max_iter=max_iter, tol=tol, random_state=None)
        kmeans.fit(train_x)
        train_cluster_labels = kmeans.predict(train_x)
        test_cluster_labels = kmeans.predict(test_x)
        cluster_to_class_map = {}
        for cluster_label in range(n_clusters):
            # Find indices of data points belonging to the current cluster
            train_indices = train_cluster_labels == cluster_label
            # Get the class labels for data points in this cluster
            class_labels_in_cluster = train_y[train_indices]
            # Find the most common class label in this cluster
            most_common_class_label = class_labels_in_cluster.mode()[0]
            # Map the cluster label to the most common class label
            cluster_to_class_map[cluster_label] = most_common_class_label
        # Map cluster labels to class labels for both training and test data
        
        train_predicted_labels = [cluster_to_class_map[label] for label in train_cluster_labels]
        test_predicted_labels = [cluster_to_class_map[label] for label in test_cluster_labels]

        # Calculate accuracy scores
        train_accuracy.append(accuracy_score(train_y, train_predicted_labels))
        test_accuracy.append(accuracy_score(test_y, test_predicted_labels))
    
        y_pred_train = kmeans.labels_
        y_pred_test = kmeans.predict(test_x)

        # Evaluate clustering metrics
        train_silhouette = silhouette_score(train_x, y_pred_train)
        test_silhouette = silhouette_score(test_x, y_pred_test)
        calinski_harabasz = calinski_harabasz_score(train_x, y_pred_train)
        davies_bouldin = davies_bouldin_score(train_x, y_pred_train)
        adjusted_rand = adjusted_rand_score(test_y, y_pred_test)
        normalized_mutual_info = normalized_mutual_info_score(test_y, y_pred_test)
        fowlkes_mallows = fowlkes_mallows_score(test_y, y_pred_test)
        
        clustering_scores.append([
            i+1, train_silhouette, test_silhouette, calinski_harabasz, davies_bouldin,
            adjusted_rand, normalized_mutual_info, fowlkes_mallows
        ])

    df = pd.DataFrame(clustering_scores, columns=[
        'Iteration', 'Train Silhouette', 'Test Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin',
        'Adjusted Rand Index', 'Normalized Mutual Info', 'Fowlkes-Mallows Index'
    ])
    
    # Save the DataFrame to a CSV file
    df.to_csv('clustering_scores.csv', index=False)
    print("Successfully executed and results saved.")
    end = time.time()
    time_=(end-start)/20
    mean_accuracy = np.mean(test_accuracy)*100
    print("### Kmeans ###\n")
    print("Mean Accuracy = %.2f" % mean_accuracy)
    print("Training and Validation lasted %.2f seconds" % time_)
    return time_,mean_accuracy

# main
if __name__ == '__main__':
    # Define the user's preferred method
    if sys.argv[1] == 'svm':
        svm_time, svm_accuracy = support_vector_machine()
    elif sys.argv[1] == 'random_forest':
        rf_time, rf_accuracy = random_forest()
    elif sys.argv[1] == 'mlp':
        mlp_time, mlp_accuracy, roc_auc = multilayer_perceptron()
    elif sys.argv[1] == 'gradient_boosting':
        grad_time, grad_accuracy = gradient_boosting()
    elif sys.argv[1] == 'k_neighbors':
        k_time, k_accuracy = k_neighbors()
    elif sys.argv[1] == 'logistic_regression':
        log_time, log_accuracy = logistic_regression()
    elif sys.argv[1] == 'xgboost':
        xg_time, xg_accuracy = xgboost()
    elif sys.argv[1] == 'kmeans':
        kmeans_time, kmeans_accuracy = kmeans()
    elif sys.argv[1] == 'classify':
        classify()    
    elif sys.argv[1] == 'comparative':
        svm_time, svm_accuracy = support_vector_machine()
        rf_time, rf_accuracy = random_forest()
        mlp_time, mlp_accuracy, roc_auc = multilayer_perceptron()
        grad_time, grad_accuracy = gradient_boosting()
        k_time, k_accuracy = k_neighbors()
        log_time, log_accuracy = logistic_regression()
        xg_time, xg_accuracy = xgboost()
        kmeans_time, kmeans_accuracy = kmeans()

        accuracy = [svm_accuracy, rf_accuracy, mlp_accuracy, grad_accuracy, k_accuracy, log_accuracy, xg_accuracy, kmeans_accuracy]
        time_ = [svm_time, rf_time, mlp_time, grad_time, k_time, log_time, xg_time,kmeans_time]

        data = {"SVM-acc": accuracy[0], "RF-acc": accuracy[1], "MLP-acc": accuracy[2], "GB-acc": accuracy[3], "K-acc": accuracy[4], "log-acc": accuracy[5], "xg-acc": accuracy[6],"kmeans-acc": accuracy[7]}

        # Creating a DataFrame from the dictionary
        df = pd.DataFrame([data])

        # Printing the DataFrame
        print(df)

        plt.ylim(0, 100)
        plt.xlabel("accuracy ")
        plt.title("Comparison of permormance")
        l1, l2, l3, l4, l5, l6, l7, l8 = plt.bar(["SVM-acc", "RF-acc", "MLP-acc",
                                                  "GB-acc", "K-acc", "log-acc",
                                                  "xg-acc","kmeans-acc"],
                                                 accuracy)
        
        plt.xticks(rotation=45)

        l1.set_facecolor('r')
        l2.set_facecolor('r')
        l3.set_facecolor('r')
        l4.set_facecolor('r')
        l5.set_facecolor('r')
        l6.set_facecolor('r')
        l7.set_facecolor('r')
        l8.set_facecolor('r')
        
        plt.show()
        plt.close('all')
        plt.show()        
        
        
    else:
        print("None algorithm was given from input")
        exit