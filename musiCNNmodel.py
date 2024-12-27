import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

def train_musicnn():
    # Load the dataset
    file_path = './music_features.csv'  # Relative path adjusted to the current working directory
    music_features_df = pd.read_csv(file_path)

    # Drop irrelevant columns
    music_features_df = music_features_df.drop(columns=["file_name"])

    # Encode target variable
    label_encoder = LabelEncoder()
    music_features_df["genre"] = label_encoder.fit_transform(music_features_df["genre"])

    # Split features and target
    X = music_features_df.drop(columns=["genre"])
    y = music_features_df["genre"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB()
    }

    # Train and evaluate classifiers
    results = []
    total_duration=0
    for name, clf in tqdm(classifiers.items(), desc="Training Models", unit="model"):
        start_time = time.time()  # Record start time
        clf.fit(X_train, y_train)  # Train on training set
        end_time = time.time()  # Record end time

        y_pred = clf.predict(X_test)  # Predict on test set

        # Evaluate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        duration = end_time - start_time

        results.append({
            "Classifier": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Duration (s)": duration
        })
        total_duration+=duration
    # Display the results as a DataFrame
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="Accuracy", ascending=False, inplace=True)

    # Print and return the results DataFrame
    return results_df,total_duration

# results_df,total_duration = train_musicnn()
# print(results_df)
