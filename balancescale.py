import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os


# GLOBAL VARIABLES
dataset = None
model = None
X = None
y = None


# FUNCTION DEFINITIONS
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


def train_model():
    """Train a classification model (Decision Tree or KNN)"""
    global model, X, y

    if dataset is None:
        print("⚠️ Please load the dataset first.")
        return

    print("\nChoose a model to train:")
    print("1. Decision Tree")
    print("2. K-Nearest Neighbors (KNN)")
    choice = input("Enter your choice (1 or 2): ")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if choice == "1":
        model = DecisionTreeClassifier(random_state=42)
        model_name = "Decision Tree"
    elif choice == "2":
        model = KNeighborsClassifier(n_neighbors=5)
        model_name = "K-Nearest Neighbors"
    else:
        print("Invalid choice.")
        return

    # Train
    model.fit(X_train, y_train)
    print(f"\n✅ {model_name} model trained successfully!")

    # Evaluate immediately
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Ask to save results
    save = input("\nDo you want to save the results to a file? (y/n): ").lower()
    if save == 'y':
        filename = input("Enter filename (e.g., results.txt): ")
        with open(filename, "w") as f:
            f.write(f"Model: {model_name}\n\n")
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))
        print(f"💾 Results saved to {filename}")


def simulate_prediction():
    """Simulate prediction for a new unseen example"""
    global model

    if model is None:
        print("⚠️ Please train a model first.")
        return

    print("\nEnter the values for a new sample:")
    lw = float(input("Left-Weight: "))
    ld = float(input("Left-Distance: "))
    rw = float(input("Right-Weight: "))
    rd = float(input("Right-Distance: "))

    new_data = [[lw, ld, rw, rd]]
    prediction = model.predict(new_data)[0]

    print(f"\nPredicted class: {prediction}")


def main_menu():
    """Main program menu"""
    while True:
        print("\n===============================")
        print(" Balance Scale Classifier Menu ")
        print("===============================")
        print("1. Load dataset")
        print("2. Train model")
        print("3. Simulate prediction")
        print("4. Exit")
        print("===============================")

        choice = input("Enter your choice: ")

        if choice == "1":
            load_dataset()
        elif choice == "2":
            train_model()
        elif choice == "3":
            simulate_prediction()
        elif choice == "4":
            print("👋 Exiting program. Goodbye!")
            break
        else:
            print("❌ Invalid choice, please try again.")


# ==============================
# RUN PROGRAM
# ==============================
if __name__ == "__main__":
    main_menu()
