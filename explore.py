import warnings
import itertools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind, f_oneway

# Ignore all warnings
warnings.simplefilter("ignore")


def explore_2(df, features, target_variable):

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
        
        # Standardize Features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(selected_features)
        
        # Cluster Using K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_features)
        
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



def explore(df, f1, f2, target_variable):

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

    # Standardize Features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(selected_features)

    # Cluster Using K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)

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


def explore2(df, f1, f2, target_variable):

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

    # Standardize Features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(selected_features)

    # Cluster Using K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)

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