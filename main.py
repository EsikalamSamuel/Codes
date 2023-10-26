import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler


def print_unique_count_of_values(col: str) -> None:
    print(f"Printing unique values for {col}")
    value_counts = df[col].value_counts()
    for value, count in value_counts.items():
        print(f"Value: {value}, Count: {count}")


def replace_value(df: pd.DataFrame, col: str, val: str, val_to_replace: str) -> None:
    df.replace({col: val}, {col: val_to_replace}, inplace=True)


def create_folders():
    folder_names = ["before", "after"]
    for folder_name in folder_names:
        path = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
            print(f"Folder '{folder_name}' created successfully at {path}")
        else:
            print(f"Folder '{folder_name}' already exists at {path}")


def plot(df: pd.DataFrame, dir: str):
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        if pd.api.types.is_numeric_dtype(df[column]):
            plt.hist(df[column], bins=10, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
        else:
            value_counts = df[column].value_counts()
            value_counts.plot(kind='bar', color='lightblue')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')

        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f"{dir}/{column}.png")
    plt.close()


def examine_variable_dependency(df: pd.DataFrame):
    for col in df:
        # Create a contingency table
        contingency_table = pd.crosstab(df[col], df['embauche'])

        # Chi-squared test
        chi2, p, _, _ = chi2_contingency(contingency_table)

        print(f"Chi2 value: {chi2}")
        print(f"P-value: {p}")
        if p < 0.05:
            print(f"There is a significant association between {col} and 'embauche'.")
        else:
            print(f"There is no significant association between {col} and 'embauche'.")


def in_balanced(df: pd.DataFrame):
    class_counts = df['embauche'].value_counts()
    print(class_counts)

    label_encoder = LabelEncoder()
    categorical_columns = ['cheveux', 'sexe', 'diplome', 'specialite']
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    X = df[['cheveux', 'sexe', 'diplome', 'specialite']]
    y = df['embauche']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", cv_scores.mean())


def address_in_balance(df: pd.DataFrame):
    label_encoder = LabelEncoder()

    categorical_columns = ['cheveux', 'sexe', 'diplome', 'specialite']
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    X = df[['cheveux', 'sexe', 'diplome', 'specialite']]
    y = df['embauche']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", cv_scores.mean())


def analysis(path_to_file: str):
    df = pd.read_csv(path_to_file)
    create_folders()
    df.apply(lambda col: print_unique_count_of_values(col.name))
    plot(df, "before")

    # do replacement
    replace_value(df, "cheveux", "?", "roux")
    replace_value(df, "dispo", "?", "oui")
    replace_value(df, "salaire", "?", "29996")
    replace_value(df, "note", "?", "37.95")
    replace_value(df, "sexe", "?", "F")
    replace_value(df, "diplome", "?", "bac")
    replace_value(df, "specialite", "?", "geologie")

    df.apply(lambda col: print_unique_count_of_values(col.name))
    plot(df, "after")

    examine_variable_dependency(df)

    in_balanced(df)
    address_in_balance(df)

    df.to_csv("processed.csv")


df = pd.read_csv("dataset.csv")

analysis("dataset.csv")
