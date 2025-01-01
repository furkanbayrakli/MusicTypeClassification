import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm
import random


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

    # Initialize classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (Poly Kernel)": SVC(kernel='poly', degree=3, probability=True),
        "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True),
        "SVM (Linear Kernel)": SVC(kernel='linear', probability=True),
        "XGBoost": XGBClassifier(n_estimators=50, eval_metric='mlogloss'),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "LightGBM": LGBMClassifier(n_estimators=50, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=50, learning_rate=0.1, depth=6, verbose=0),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB()
    }

    # Train and evaluate classifiers for multiple rounds
    results = []
    total_duration=0
    for round_num in range(1, 4):  # Three rounds of training
        seed = random.randint(30, 200)  # Different seed for each round
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

        for name, clf in tqdm(classifiers.items(), desc=f"Training Models - Round {round_num}", unit="model"):
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
                "Round": round_num,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Duration (s)": duration
            })
            total_duration += duration
    # Display the results as a DataFrame
    results_df = pd.DataFrame(results)
    results_df.sort_values(by=["Round", "Accuracy"], ascending=[True, False], inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # Print and return the results DataFrame
    return results_df,total_duration

# results_df, total_duration = train_musicnn()
# results_df2, total_duration2 = train_musicnn()
# results_df3, total_duration3 = train_musicnn()

# print(results_df)
# print(results_df2)
# print(results_df3)
