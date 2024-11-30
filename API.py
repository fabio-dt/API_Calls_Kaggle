import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE

import sys

FILENAME = "dynamic_api_call_sequence_per_malware_100_0_306.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
NR_FEATURES = 10
EPOCHS = 10

runLogisticRegression = False
runKNN = False
runRF = False
runDL = False
runRNN = False
runLSTM = True
runBiLSTM = False
runRFE = False

Confusion_Matrix = False
Loss_Plots = False
Feature_Importance = False


def main():
    df = readData(FILENAME)
    print("\n\nDataset info:")
    print(df.info())
    
    for i in range(1, NR_FEATURES+1):
        print(f"***Features number: {i}***")
        features, labels = extractFeaturesLabels(df, i) 
    
        run(features, labels)


def preprocessing(features, labels):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    #Scale the features
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)
    
    return (features_train, features_test, labels_train, labels_test)


def run(features, labels):
    # Under/Oversampling
    if len(sys.argv) > 1:
        argument = sys.argv[1]        
        if argument == "randomO":
            features, labels = under_and_oversampling("randomO", features, labels)            
        if argument == "randomU":
            features, labels = under_and_oversampling("randomU", features, labels)

    # Preprocessing
    features_train, features_test, labels_train, labels_test = preprocessing(features, labels)


    print("\n")
    print("features_train shape:\t", features_train.shape)
    print("labels_train shape:\t", labels_train.shape)
    print("features_test shape:\t", features_test.shape)
    print("labels_test shape:\t", labels_test.shape)


    
    if (runLogisticRegression):

        # Initialize and train the Logistic Regression model
        logreg = LogisticRegression(solver='liblinear', random_state=42) # You can adjust parameters as needed
        logreg.fit(features_train, labels_train)

        # Make predictions on the test set
        labels_pred = logreg.predict(features_test)

        # Evaluate the model
        print(confusion_matrix(labels_test, labels_pred))
        print(classification_report(labels_test, labels_pred))
        print("Accuracy:", accuracy_score(labels_test, labels_pred))
        
        # Visualize the confusion matrix 
        if (Confusion_Matrix):
            cm = confusion_matrix(labels_test, labels_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()
  
    if (runKNN):
        knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

        # Train the KNN classifier
        knn.fit(features_train, labels_train)

        # Make predictions on the test set
        labels_pred = knn.predict(features_test)

        # Evaluate the model
        print("Confusion Matrix:\n", confusion_matrix(labels_test, labels_pred))
        print("\nClassification Report:\n", classification_report(labels_test, labels_pred))

        # Visualize the confusion matrix 
        if (Confusion_Matrix):
            cm = confusion_matrix(labels_test, labels_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()
    
    if (runRF):
        # Initialize and train the Random Forest Classifier
        rf_classifier = RandomForestClassifier(random_state=42)  # You can adjust parameters
        rf_classifier.fit(features_train, labels_train)

        # Make predictions
        labels_pred = rf_classifier.predict(features_test)

        # Evaluate the model
        print(confusion_matrix(labels_test, labels_pred))
        print(classification_report(labels_test, labels_pred))
        print("Accuracy:", accuracy_score(labels_test, labels_pred))
        
        # Confusion Matrix
        if (Confusion_Matrix):
            cm = confusion_matrix(labels_test, labels_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()

        # Feature Importance
        if (Feature_Importance):
            importances = rf_classifier.feature_importances_
            feature_names = features.columns  # Assuming 'features' DataFrame has column names
            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

            # Sort features by importance
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            print(feature_importance_df)

            # Visualize feature importance
            plt.figure(figsize=(15, 10))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
            plt.title('Feature Importance (Random Forest)')
            plt.show()
    
    if (runDL):
        # Define the model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(features_train.shape[1],)),
            keras.layers.Dropout(0.2),  # Add dropout for regularization
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid') # Output layer with sigmoid for binary classification
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', # Use binary_crossentropy for binary classification
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(features_train, labels_train, epochs=EPOCHS, batch_size=32, validation_split=0.1) #Added validation split
        
        ## Confusion Matrix
        if (Confusion_Matrix):
            # Predict probabilities on the test set
            y_pred_prob = model.predict(features_test)
            # Convert probabilities to class labels (0 or 1)
            y_pred = (y_pred_prob > 0.5).astype(int) # Adjust threshold if needed
            y_pred = y_pred.flatten()
            # Compute the confusion matrix
            cm = confusion_matrix(labels_test, y_pred)
            # Plot the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix (Deep Learning Model)')
            plt.show()

        # Evaluate the model
        loss, accuracy = model.evaluate(features_test, labels_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
        # Predict probabilities on the test set
        y_pred_prob = model.predict(features_test)
        # Convert probabilities to class labels (0 or 1) using a threshold of 0.5
        y_pred = (y_pred_prob > 0.5).astype(int)
        # Evaluate the model using classification_report
        print(classification_report(labels_test, y_pred))
        
        
        
        # Plot training & validation loss values
        if (Loss_Plots):
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss']) 
            plt.plot(history.history['val_loss']) 
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
        
    if (runRNN):
        # Reshape the data for RNN input
        features_train = np.reshape(features_train, (features_train.shape[0], 1, features_train.shape[1]))
        features_test = np.reshape(features_test, (features_test.shape[0], 1, features_test.shape[1]))


        # Define the RNN model
        model = keras.Sequential([
            keras.layers.SimpleRNN(64, activation='relu', input_shape=(features_train.shape[1], features_train.shape[2])),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(features_train, labels_train, epochs=EPOCHS, batch_size=32, validation_split=0.1)

        # Evaluate the model and print metrics (same as before)
        loss, accuracy = model.evaluate(features_test, labels_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
        # Predict probabilities on the test set
        y_pred_prob = model.predict(features_test)
        # Convert probabilities to class labels (0 or 1) using a threshold of 0.5
        y_pred = (y_pred_prob > 0.5).astype(int)
        # Evaluate the model using classification_report
        print(classification_report(labels_test, y_pred))
        
        ## Confusion Matrix
        if (Confusion_Matrix):
            # Predict probabilities on the test set
            y_pred_prob = model.predict(features_test)
            # Convert probabilities to class labels (0 or 1)
            y_pred = (y_pred_prob > 0.5).astype(int) # Adjust threshold if needed
            y_pred = y_pred.flatten()
            # Compute the confusion matrix
            cm = confusion_matrix(labels_test, y_pred)
            # Plot the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix (Deep Learning Model)')
            plt.show()
        
        # Plot training & validation loss values
        if (Loss_Plots):
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss']) 
            plt.plot(history.history['val_loss']) 
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
    
    if (runLSTM):
        # Reshape the data for LSTM input - adding a timestep dimension
        features_train = features_train.reshape(features_train.shape[0], 1, features_train.shape[1])
        features_test = features_test.reshape(features_test.shape[0], 1, features_test.shape[1])

        # Define the LSTM model
        model = keras.Sequential([
            keras.layers.LSTM(64, activation='relu', input_shape=(features_train.shape[1], features_train.shape[2])),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(features_train, labels_train, epochs=EPOCHS, batch_size=32, validation_split=0.1)

        # Evaluate the model and print metrics
        loss, accuracy = model.evaluate(features_test, labels_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
        # Predict probabilities on the test set
        y_pred_prob = model.predict(features_test)
        # Convert probabilities to class labels (0 or 1) using a threshold of 0.5
        y_pred = (y_pred_prob > 0.5).astype(int)
        # Evaluate the model using classification_report
        print(classification_report(labels_test, y_pred))
        
        ## Confusion Matrix
        if (Confusion_Matrix):
            # Predict probabilities on the test set
            y_pred_prob = model.predict(features_test)
            # Convert probabilities to class labels (0 or 1)
            y_pred = (y_pred_prob > 0.5).astype(int) # Adjust threshold if needed
            y_pred = y_pred.flatten()
            # Compute the confusion matrix
            cm = confusion_matrix(labels_test, y_pred)
            # Plot the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix (Deep Learning Model)')
            plt.show()
        
        # Plot training & validation loss values
        if (Loss_Plots):
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss']) 
            plt.plot(history.history['val_loss']) 
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
        
    if (runBiLSTM):
        # Reshape the data for LSTM input - adding a timestep dimension
        features_train = features_train.reshape(features_train.shape[0], 1, features_train.shape[1])
        features_test = features_test.reshape(features_test.shape[0], 1, features_test.shape[1])

        # Define the BiLSTM model
        model = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(64, activation='relu', return_sequences=True), input_shape=(features_train.shape[1], features_train.shape[2])),
            keras.layers.Bidirectional(keras.layers.LSTM(32, activation='relu')), #Added another bidirectional layer
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(features_train, labels_train, epochs=EPOCHS, batch_size=32, validation_split=0.1)

        # Evaluate the model and print metrics
        loss, accuracy = model.evaluate(features_test, labels_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
        # Predict probabilities on the test set
        y_pred_prob = model.predict(features_test)
        # Convert probabilities to class labels (0 or 1) using a threshold of 0.5
        y_pred = (y_pred_prob > 0.5).astype(int)
        # Evaluate the model using classification_report
        print(classification_report(labels_test, y_pred))
        
        ## Confusion Matrix
        if (Confusion_Matrix):
            # Predict probabilities on the test set
            y_pred_prob = model.predict(features_test)
            # Convert probabilities to class labels (0 or 1)
            y_pred = (y_pred_prob > 0.5).astype(int) # Adjust threshold if needed
            y_pred = y_pred.flatten()
            # Compute the confusion matrix
            cm = confusion_matrix(labels_test, y_pred)
            # Plot the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix (Deep Learning Model)')
            plt.show()
        
        # Plot training & validation loss values
        if (Loss_Plots):
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss']) 
            plt.plot(history.history['val_loss']) 
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
    
    if (runRFE):
        # Assuming features_train, labels_train, features_test, labels_test are already defined from previous code

        # Initialize the SVM classifier
        svc = SVC(kernel="linear")

        # Initialize the Recursive Feature Elimination (RFE) object
        num_features = features_train.shape[1]
        rfecv = RFE(estimator=svc, n_features_to_select=1, step=1, verbose = 1)

        # Fit RFE to the training data
        rfecv.fit(features_train, labels_train)

        # Get the accuracy scores for each number of features
        accuracy_scores = []
        for i in range(1, num_features + 1):
          # Select the top i features
          selected_features_train = features_train.iloc[:, rfecv.support_[:i]]
          selected_features_test = features_test.iloc[:, rfecv.support_[:i]]

          # Train an SVM model with the selected features
          svm = SVC(kernel='linear')
          svm.fit(selected_features_train, labels_train)

          # Evaluate the model and store the accuracy score
          accuracy = svm.score(selected_features_test, labels_test)
          accuracy_scores.append(accuracy)

        # Create a plot of accuracy vs number of features
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_features + 1), accuracy_scores, marker='o')
        plt.xlabel("Number of Features")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Number of Features (Recursive Feature Elimination)")
        plt.grid(True)
        plt.show()

   
def readData(filename):
    try:
        df = pd.read_csv(filename)
        print("Malware API Dataset loaded successfully!")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found. Please make sure the file {FILENAME} exists in the current directory or provide the correct path.")
    except Exception as e:
        print(f"An error occurred: {e}")   
 
 
def extractFeaturesLabels(df, nr_features=100): 
    nr_features = 100 - nr_features + 1 # This code is hardcoded with the dataset format
    # The first columns is the malwsare hash (ID), we remove it
    #features = df.iloc[:, 1:-1] # Full 100 API calls
    features = df.iloc[:, 1:-nr_features] # Reduced number of API calls 
    labels = df.iloc[:, -1] 
    return (features, labels)
    
    
def under_and_oversampling(sampling_method, features, labels):
    if sampling_method == "randomO":
        print("Applying Random Oversampling...")
        ros = RandomOverSampler(random_state=42)
        features, labels = ros.fit_resample(features, labels)
 
    elif sampling_method == "randomU":
        print("Applying Random Undersampling...")
        rus = RandomUnderSampler(random_state=42)
        features, labels = rus.fit_resample(features, labels)

    else:
        print("***The command line option has not been recognized!***")
        exit(1)
            
    print("Total number of labels for each class:")
    print(labels.value_counts())
    
    return (features, labels)



if __name__ == "__main__":
    main()







