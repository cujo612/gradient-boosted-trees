from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv("extracted_features.csv")
X = data.drop(columns=["signal_type"])
y = data["signal_type"]

# Function to compute and plot accuracy for each signal type across different SNR levels
def compute_and_plot_accuracy_by_snr(X, y, clf, depth, estimators, rate):
    # Get unique signal types and SNR levels
    signal_types = sorted(y.unique())
    snr_levels = sorted(data["snr"].unique())
    
    # Perform multiple runs to get robust results
    num_runs = 1
    all_accuracies = np.zeros((num_runs, len(signal_types), len(snr_levels)))
    overall_accuracies = np.zeros((num_runs, len(snr_levels)))
    
    for run in range(num_runs):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + run)
        
        # Train the classifier
        clf.fit(X_train, y_train)
        
        # Compute accuracy for each signal type and SNR level
        for j, snr in enumerate(snr_levels):
            # Filter test data for specific SNR
            mask = X_test['snr'] == snr
            
            if mask.sum() > 0:
                X_subset = X_test[mask]
                y_subset = y_test[mask]
                
                # Predict and compute overall accuracy for this SNR
                y_pred = clf.predict(X_subset)
                overall_accuracies[run, j] = accuracy_score(y_subset, y_pred)
                
                # Compute accuracy for each signal type at this SNR
                for i, signal_type in enumerate(signal_types):
                    type_mask = (y_subset == signal_type)
                    if type_mask.sum() > 0:
                        X_type_subset = X_subset[type_mask]
                        y_type_subset = y_subset[type_mask]
                        
                        y_type_pred = clf.predict(X_type_subset)
                        all_accuracies[run, i, j] = accuracy_score(y_type_subset, y_type_pred)
    
    # Average across runs
    accuracy_matrix = np.mean(all_accuracies, axis=0)
    overall_accuracy_by_snr = np.mean(overall_accuracies, axis=0)
    
    # Create figure for multiple plots
    # plt.figure(figsize=(16, 10))
    
    # Subplot 1: Recognition Accuracy by Signal Type
    # plt.subplot(2, 1, 1)
    # Plot individual signal type accuracies
    for i, signal_type in enumerate(signal_types):
        plt.plot(snr_levels, accuracy_matrix[i], marker='o', linestyle='--', label=f'{signal_type}')
    
    plt.title(f'Recognition Accuracy by Signal Type and SNR\n(Depth={depth}, Estimators={estimators}, Rate={rate})')
    plt.xlabel('Signal-to-Noise Ratio (SNR)')
    plt.ylabel('Recognition Accuracy')
    plt.ylim(0, 1)
    plt.legend(title='Modulation Types', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'model_accuracy_by_signal_type_snr_depth_{depth}_estimators_{estimators}_rate_{rate}.png')
    plt.close()
    
    # Subplot 2: Overall Model Accuracy by SNR
    # plt.subplot(2, 1, 2)
    plt.plot(snr_levels, overall_accuracy_by_snr, marker='s', color='red', linewidth=2, label='Overall Model Accuracy')
    plt.title(f'Overall Model Accuracy across Different SNR Levels (Depth={depth}, Estimators={estimators}, Rate={rate})')
    plt.xlabel('Signal-to-Noise Ratio (SNR)')
    plt.ylabel('Model Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # plt.tight_layout()
    plt.savefig(f'overall_model_accuracy_by_snr_depth_{depth}_estimators_{estimators}_rate_{rate}.png')
    plt.close()

# Configurations to test
configs = [(1,300,0.1),(2,500,0.1),(5,300,0.1),(5,300,0.2),(9,500,0.1),(10,300,0.1)]
# configs = [(5,300,0.1)]


# Iterate through configurations
for config in configs:
    depth = config[0]
    estimators = config[1]
    rate = config[2]
    
    # Initialize the classifier
    clf = GradientBoostingClassifier(max_depth=depth, n_estimators=estimators, learning_rate=rate)
    
    # Measure training time
    before = datetime.datetime.now()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    after = datetime.datetime.now()
    time_difference = after - before
    seconds_delta = time_difference.total_seconds()
    
    # Make predictions and evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy of depth {depth}, estimators {estimators}, learning rate {rate}: {accuracy * 100:.2f}%, time to train {seconds_delta} seconds")
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualize the confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues')
    plt.xticks(rotation=45)
    plt.title(f"Confusion Matrix (Depth={depth}, Estimators={estimators}, Rate={rate})")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{depth}_depth_{estimators}_estimators_{rate}_rate.png")
    plt.close()
    
    # Compute and plot recognition accuracy by SNR
    compute_and_plot_accuracy_by_snr(X, y, clf, depth, estimators, rate)
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=sorted(y.unique())))