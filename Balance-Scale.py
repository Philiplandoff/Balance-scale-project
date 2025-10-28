import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Global variables
# X = features
# y = target
dataset = None
model = None
X = None
y = None

# Function definitions
def load_dataset():
    global dataset, X, y
    filename = "Project/data/balance-scale.data"

    if not os.path.exists(filename):
        print("Dataset file not found! Make sure 'balance-scale.data' is in the same folder.")
        return

    # According to UCI: Class, Left-Weight, Left-Distance, Right-Weight, Right-Distance
    col_names = ['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']
    dataset = pd.read_csv(filename, header=None, names=col_names)

    print("\nDataset loaded successfully!")
    print("\nFirst 10 rows:")
    print(dataset.head(10))
    print("\nBasic Statistics:")
    print(dataset.describe())

    # Separate features and target
    X = dataset[['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']]
    y = dataset['Class']

# Train model
def train_model():
    global model, X, y

    if dataset is None:
        print("Please load the dataset first.")
        return

    print("\nChoose a model to train:")
    print("1. Decision Tree")
    print("2. K-Nearest Neighbors (KNN)")
    choice = input("Enter your choice (1 or 2): ")

    # Split dataset for initial training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    if choice == "1":
        model = DecisionTreeClassifier(random_state=10)
        model_name = "Decision Tree"
    elif choice == "2":
        model = KNeighborsClassifier(n_neighbors=5)
        model_name = "K-Nearest Neighbors"
    else:
        print("Invalid choice.")
        return

    # Train
    model.fit(X_train, y_train)
    print(f"\n{model_name} model trained successfully!")

    # Evaluation phase
    print("\nWould you like to evaluate using a separate file? (yes/no)")
    use_file = input("Your choice: ").lower()

    if use_file == 'yes':
        eval_file = input("Enter the path to the evaluation file: ")

        if not os.path.exists(eval_file):
            print("File not found. Evaluation aborted.")
            return

        # Load external evaluation dataset
        col_names = ['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']
        eval_data = pd.read_csv(eval_file, header=None, names=col_names)
        X_eval = eval_data[['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']]
        y_eval = eval_data['Class']
    else:
        # Use internal partitioning if no external file
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=10)
        model.fit(X_train, y_train)

    # Perform evaluation
    y_pred = model.predict(X_eval)

    accuracy = accuracy_score(y_eval, y_pred)
    report = classification_report(y_eval, y_pred)
    matrix = confusion_matrix(y_eval, y_pred)

    print("\n=== Model Evaluation ===")
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", matrix)

    # Ask to save results
    save = input("\nDo you want to save the results to a file? (yes/no): ").lower()
    if save == 'yes':
        filename = input("Enter filename: ")
        with open(filename, "w") as f:
            f.write(f"Model: {model_name}\n\n")
            f.write(f"Accuracy: {accuracy}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(matrix))
        print(f"Results saved to {filename}")

# Simulate prediction
def simulate_prediction():
    
    global model

    if model is None:
        print("Please train a model first.")
        return

    print("\nEnter the values for a new sample:")
    lw = float(input("Left-Weight: "))
    ld = float(input("Left-Distance: "))
    rw = float(input("Right-Weight: "))
    rd = float(input("Right-Distance: "))

    new_data = pd.DataFrame([[lw, ld, rw, rd]], columns=['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'])
    prediction = model.predict(new_data)[0]

    print(f"\nPredicted class: {prediction}")

# Main menu
def main_menu():
    while True:
        print("\n===========================")
        print(" Balance Scale Menu ")
        print("===========================")
        print("1. Load dataset")
        print("2. Train model")
        print("3. Simulate prediction")
        print("4. Exit")
        print("===========================")

        choice = input("Enter your choice: ")

        if choice == "1":
            load_dataset()
        elif choice == "2":
            train_model()
        elif choice == "3":
            simulate_prediction()
        elif choice == "4":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

# Run program
if __name__ == "__main__":
    main_menu()