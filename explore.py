import warnings
import itertools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind, f_oneway

# Ignore all warnings
warnings.simplefilter("ignore")


def explore_2(df, df2, df3, features, target_variable):

    plotted_features = set()  # Keep track of plotted features

    # 1. Explore the Interaction between Independent Variables and the Target Variable
    
    for f1, f2 in itertools.combinations(features, 2):
        # Visualizations (Scatter Plot)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=target_variable, y=f1, data=df, errorbar=None)
        plt.title(f"Bar Plot of {target_variable} vs. {f1}")
        plt.show()
        
        # Mark the feature as plotted
        plotted_features.add(f1)

        # Statistical Testing (Pearson's Correlation)
        correlation_coefficient, p_value = pearsonr(df[f1], df[target_variable])
        print(f"Pearson's Correlation Coefficient between {f1} and {target_variable}: {correlation_coefficient}, P-Value: {p_value}")

    # 2. Explore Clustering with Different Feature Combinations
    
    for combination in itertools.combinations(features, 2):
        f1, f2 = combination
        
        # Feature Selection
        selected_features = df[[f1, f2]]
        selected_features2 = df2[[f1, f2]]
        selected_features3 = df3[[f1, f2]]
        
        # Standardize Features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(selected_features)
        scaled_features2 = scaler.fit_transform(selected_features2)
        scaled_features3 = scaler.fit_transform(selected_features3)
    
        # Cluster Using K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_features)
        df2['cluster'] = kmeans.fit_predict(scaled_features2)
        df3['cluster'] = kmeans.fit_predict(scaled_features3)

        
        # 3. Assess Cluster Utility
        
        # Visualize Clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=f1, y=f2, hue='cluster', data=df, palette='viridis')
        plt.title(f"Cluster Visualization of {f1} and {f2}")
        plt.show()
        
        # Statistical Testing (ANOVA)
        f_stat, p_value = f_oneway(df[df['cluster'] == 0][target_variable], df[df['cluster'] == 1][target_variable], df[df['cluster'] == 2][target_variable])
        print(f"F-Statistic for clusters based on {f1} and {f2}: {f_stat}, P-Value: {p_value}")

        # 4. Compare Clusters to the Target Variable
        
        # Visualize cluster distribution with target variable (Box Plot)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y=target_variable, data=df)
        plt.title(f"Box Plot of Cluster vs. {target_variable}")
        plt.show()
        
        # Statistical Testing (ANOVA or Kruskal-Wallis)
        cluster_groups = [df[df['cluster'] == i][target_variable] for i in range(3)]
        f_stat, p_value = f_oneway(*cluster_groups)
        print(f"F-Statistic for cluster vs. {target_variable}: {f_stat}, P-Value: {p_value}")



def explore(df, df2, df3, f1, f2, target_variable):

    # 1a. Explore the Interaction between Independent Variables and the Target Variable

    # Visualizations (Scatter Plot)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=target_variable, y=f1, data=df, errorbar=None)
    plt.title(f"Bar Plot of {target_variable} vs. {f1}")
    plt.show()

    # Statistical Testing (Pearson's Correlation)
    correlation_coefficient, p_value = pearsonr(df[f1], df[target_variable])
    print(f"Pearson's Correlation Coefficient between {f1} and {target_variable}: {correlation_coefficient}, P-Value: {p_value}")

    # 1b. Explore the Interaction between Independent Variables and the Target Variable

    # Visualizations (Scatter Plot)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=target_variable, y=f2, data=df, errorbar=None)
    plt.title(f"Bar Plot of {target_variable} vs. {f2}")
    plt.show()

    # Statistical Testing (Pearson's Correlation)
    correlation_coefficient, p_value = pearsonr(df[f2], df[target_variable])
    print(f"Pearson's Correlation Coefficient between {f2} and {target_variable}: {correlation_coefficient}, P-Value: {p_value}")
    
    # 2. Explore Clustering with the Specified Feature Combination

    # Feature Selection
    selected_features = df[[f1, f2]]
    selected_features2 = df2[[f1, f2]]
    selected_features3 = df3[[f1, f2]]
    
    # Standardize Features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(selected_features)
    scaled_features2 = scaler.fit_transform(selected_features2)
    scaled_features3 = scaler.fit_transform(selected_features3)

    # Cluster Using K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    df2['cluster'] = kmeans.fit_predict(scaled_features2)
    df3['cluster'] = kmeans.fit_predict(scaled_features3)

    # 3. Assess Cluster Utility

    # Visualize Clusters
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=f1, y=f2, hue='cluster', data=df, palette='viridis')
    plt.title(f"Cluster Visualization of {f1} and {f2}")
    plt.show()

    # Statistical Testing (ANOVA)
    f_stat, p_value = f_oneway(df[df['cluster'] == 0][target_variable], df[df['cluster'] == 1][target_variable], df[df['cluster'] == 2][target_variable])
    print(f"F-Statistic for clusters based on {f1} and {f2}: {f_stat}, P-Value: {p_value}")

    # 4. Compare Clusters to the Target Variable

    # Visualize cluster distribution with target variable (Box Plot)
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='cluster', y=target_variable, data=df)
    plt.title(f"Box Plot of Cluster vs. {target_variable}")
    plt.show()

    # Statistical Testing (ANOVA or Kruskal-Wallis)
    cluster_groups = [df[df['cluster'] == i][target_variable] for i in range(3)]
    f_stat, p_value = f_oneway(*cluster_groups)
    print(f"F-Statistic for cluster vs. {target_variable}: {f_stat}, P-Value: {p_value}")


def explore2(df, df2, df3, f1, f2, target_variable):

    # 1a. Explore the Interaction between Independent Variables and the Target Variable

    # Visualizations (Scatter Plot)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=target_variable, y=f1, data=df, errorbar=None)
    plt.title(f"Bar Plot of {target_variable} vs. {f1}")
    plt.show()

    # Statistical Testing (Pearson's Correlation)
    correlation_coefficient, p_value = pearsonr(df[f1], df[target_variable])
    print(f"Pearson's Correlation Coefficient between {f1} and {target_variable}: {correlation_coefficient}, P-Value: {p_value}")

    # 1b. Explore the Interaction between Independent Variables and the Target Variable

    # Visualizations (Scatter Plot)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=target_variable, y=f2, data=df, errorbar=None)
    plt.title(f"Bar Plot of {target_variable} vs. {f2}")
    plt.ylim(0.98, 1.00)
    plt.show()

    # Statistical Testing (Pearson's Correlation)
    correlation_coefficient, p_value = pearsonr(df[f2], df[target_variable])
    print(f"Pearson's Correlation Coefficient between {f2} and {target_variable}: {correlation_coefficient}, P-Value: {p_value}")
    
    # 2. Explore Clustering with the Specified Feature Combination

    # Feature Selection
    selected_features = df[[f1, f2]]
    selected_features2 = df2[[f1, f2]]
    selected_features3 = df3[[f1, f2]]
        
    # Standardize Features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(selected_features)
    scaled_features2 = scaler.fit_transform(selected_features2)
    scaled_features3 = scaler.fit_transform(selected_features3)
    
    # Cluster Using K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    df2['cluster'] = kmeans.fit_predict(scaled_features2)
    df3['cluster'] = kmeans.fit_predict(scaled_features3)

    # 3. Assess Cluster Utility

    # Visualize Clusters
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=f1, y=f2, hue='cluster', data=df, palette='viridis')
    plt.title(f"Cluster Visualization of {f1} and {f2}")
    plt.show()

    # Statistical Testing (ANOVA)
    f_stat, p_value = f_oneway(df[df['cluster'] == 0][target_variable], df[df['cluster'] == 1][target_variable], df[df['cluster'] == 2][target_variable])
    print(f"F-Statistic for clusters based on {f1} and {f2}: {f_stat}, P-Value: {p_value}")

    # 4. Compare Clusters to the Target Variable

    # Visualize cluster distribution with target variable (Box Plot)
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='cluster', y=target_variable, data=df)
    plt.title(f"Box Plot of Cluster vs. {target_variable}")
    plt.show()

    # Statistical Testing (ANOVA or Kruskal-Wallis)
    cluster_groups = [df[df['cluster'] == i][target_variable] for i in range(3)]
    f_stat, p_value = f_oneway(*cluster_groups)
    print(f"F-Statistic for cluster vs. {target_variable}: {f_stat}, P-Value: {p_value}")



def create_baselines(y_train):
    """
    Create a DataFrame 'baselines' with columns 'y_actual,' 'y_mean,' and 'y_median.'

    Parameters:
    y_train (pd.Series or array-like): Actual target values from the training dataset.

    Returns:
    pd.DataFrame: DataFrame containing 'y_actual,' 'y_mean,' and 'y_median' columns.
    """
    baselines = pd.DataFrame({
        'y_actual': y_train,
        'y_mean': y_train.mean(),
        'y_median': y_train.median()
    })
    
    return baselines


def eval_model(y_actual, y_hat):
    """
    Calculate and return the root mean squared error (RMSE) between actual and predicted values.

    Parameters:
    y_actual (array-like): The actual target values.
    y_hat (array-like): The predicted target values.

    Returns:
    float: The calculated root mean squared error (RMSE) rounded to two decimal places.
    """
    return round(sqrt(mean_squared_error(y_actual, y_hat)), 2)


def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train a machine learning model and evaluate its performance on training and validation data.

    Parameters:
    model: The machine learning model to be trained.
    X_train (array-like): Training features.
    y_train (array-like): Training target values.
    X_val (array-like): Validation features.
    y_val (array-like): Validation target values.

    Returns:
    model: Trained machine learning model.
    """
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    
    train_rmse = eval_model(y_train, train_preds)
    
    val_preds = model.predict(X_val)
    
    val_rmse = eval_model(y_val, val_preds)
    
    print(f'The train RMSE is {round(train_rmse, 2)}.')
    print(f'The validate RMSE is {round(val_rmse, 2)}.')


def polynomial_feature_expansion(X_train, X_val, X_test, degree=2):
    """
    Perform polynomial feature expansion for training and validation datasets.

    Parameters:
    X_train (pd.DataFrame): Training features.
    X_val (pd.DataFrame): Validation features.
    X_test (pd.DataFrame): Test features.
    degree (int): Degree of polynomial features to be created (default is 2).

    Returns:
    pd.DataFrame: Training features with polynomial expansion.
    pd.DataFrame: Validation features with polynomial expansion.
    pd.DataFrame: Test features with polynomial expansion.
    """
    # Create an instance of PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)

    # Transform the training features into polynomial features
    X_train_s = poly.fit_transform(X_train)

    # Transform the validation features into polynomial features
    X_val_s = poly.fit_transform(X_val)

    # Transform the test features into polynomial features
    X_test_s = poly.fit_transform(X_test)

    return X_train_s, X_val_s, X_test_s, poly