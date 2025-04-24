"""
Clustering utilities for text data sampling and analysis.

This module provides efficient implementations of dimensionality reduction,
clustering, and stratified sampling techniques specialized for large text datasets.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import optuna
from matplotlib import pyplot as plt



def fast_dim_reduction(X, method='pca', n_components=50):
    """
    Perform fast dimensionality reduction on feature matrix.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix to reduce
    method : str
        Method to use ('pca' or 'tsne')
    n_components : int
        Number of dimensions to reduce to
        
    Returns:
    --------
    array-like
        Reduced feature matrix
    """
    if method == 'pca':
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X)
    elif method == 'tsne':
        tsne = TSNE(n_components=n_components, random_state=42, n_iter=300, perplexity=30)
        X_reduced = tsne.fit_transform(X)
    else:
        raise ValueError('Unknown method')
    return X_reduced

def birch_clustering(X, n_clusters=10, threshold=0.5, branching_factor=50):
    """
    Apply Birch clustering algorithm to feature matrix.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix to cluster
    n_clusters : int
        Number of clusters to form
    threshold : float
        Threshold for Birch clustering
    branching_factor : int
        Branching factor for Birch tree
        
    Returns:
    --------
    array-like
        Cluster labels for each sample
    """
    birch = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    labels = birch.fit_predict(X)
    return labels

def fast_cluster_metric(X, labels, metric='silhouette', expected_clusters=None):
    """
    Calculate cluster quality metrics efficiently using sampling for large datasets.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    labels : array-like
        Cluster labels for each sample
    metric : str
        Metric to calculate ('silhouette', 'davies_bouldin', or 'calinski_harabasz')
    expected_clusters : int or None
        The expected number of clusters; will return a penalty score if mismatch
        
    Returns:
    --------
    float
        Calculated metric score
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    # Check if we have at least 2 clusters for valid metrics
    if num_clusters < 2:
        return -999999.0  # Invalid clustering - severe penalty
    
    # Check if number of clusters matches expected value
    if expected_clusters is not None and num_clusters != expected_clusters:
        return -999999.0  # Mismatch in number of clusters - severe penalty
        
    # For large datasets, use sampling
    if len(X) > 10000:
        idx = np.random.choice(len(X), 10000, replace=False)
        X_sample = X[idx]
        labels_sample = np.array(labels)[idx]
    else:
        X_sample = X
        labels_sample = labels
    
    # Calculate selected metric
    if metric == 'silhouette':
        return silhouette_score(X_sample, labels_sample)
    elif metric == 'davies_bouldin':
        # Negative because lower is better and Optuna is set to maximize by default
        return -davies_bouldin_score(X_sample, labels_sample)  
    elif metric == 'calinski_harabasz':
        return calinski_harabasz_score(X_sample, labels_sample)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def objective(trial, X, metric='davies_bouldin'):
    """
    Optuna objective function for Birch parameter tuning.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    X : array-like
        Feature matrix
    metric : str
        Metric to optimize
        
    Returns:
    --------
    float
        Optimization score
    """
    # Define parameter search space
    n_clusters = trial.suggest_int('n_clusters', 5, 500, 10)
    threshold = trial.suggest_float('threshold', 0.05, 2.5, log=True)
    branching_factor = trial.suggest_int('branching_factor', 20, 100)
    
    # Apply clustering with suggested parameters
    labels = birch_clustering(
        X, 
        n_clusters=n_clusters, 
        threshold=threshold, 
        branching_factor=branching_factor
    )
    
    # Evaluate clustering quality - pass expected cluster count for validation
    score = fast_cluster_metric(X, labels, metric=metric, expected_clusters=n_clusters)
    return score

def optuna_birch_tuning(X, n_trials=20, metric='silhouette', timeout=None, show_progress=True):
    """
    Tune Birch clustering parameters using Optuna.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    n_trials : int
        Number of optimization trials
    metric : str
        Metric to optimize ('silhouette', 'davies_bouldin', or 'calinski_harabasz')
    timeout : int or None
        Time limit in seconds for optimization
    show_progress : bool
        Whether to show optimization progress
        
    Returns:
    --------
    dict
        Best parameters found
    optuna.study.Study
        The completed study object
    """
    # Create study with appropriate direction
    study = optuna.create_study(direction='maximize')
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X, metric), 
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress
    )
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_trial.value:.4f}")
    print(f"Best parameters: {study.best_params}")
    return study.best_params, study

def plot_optimization_history(study):
    """
    Plot the optimization history from an Optuna study.
    
    Parameters:
    -----------
    study : optuna.study.Study
        Completed Optuna study
        
    Returns:
    --------
    None
    """
    optuna.visualization.matplotlib.plot_optimization_history(study) 
    optuna.visualization.matplotlib.plot_param_importances(study) 
    return 

def batch_process(X, batch_size=10000, dimred_method='pca', n_components=50, n_trials=20):
    """
    Process large datasets in batches for clustering.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    batch_size : int
        Size of each batch
    dimred_method : str
        Dimensionality reduction method
    n_components : int
        Number of components for dimensionality reduction
    n_trials : int
        Number of Optuna trials
        
    Returns:
    --------
    array-like
        Cluster labels for each sample
    """
    all_labels = []
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size]
        X_reduced = fast_dim_reduction(X_batch, method=dimred_method, n_components=n_components)
        best_params, _ = optuna_birch_tuning(X_reduced, n_trials=n_trials)
        labels = birch_clustering(X_reduced, **best_params)
        all_labels.extend(labels)
    return np.array(all_labels)

def stratified_cluster_sampling(texts, labels, sample_size, min_per_cluster=1):
    """
    Sample text data based on cluster assignments to ensure representation.
    
    Parameters:
    -----------
    texts : list
        List of text documents
    labels : array-like
        Cluster labels for each document
    sample_size : int
        Desired sample size
    min_per_cluster : int
        Minimum samples to take from each cluster
        
    Returns:
    --------
    list
        Indices of sampled documents
    """
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    # Calculate base samples per cluster (ensuring minimum)
    base_per_cluster = max(min_per_cluster, sample_size // n_clusters)
    
    # Calculate proportional allocation for remaining samples
    remaining = sample_size - (base_per_cluster * n_clusters)
    if remaining < 0:
        # If we can't give min_per_cluster to each, adjust base_per_cluster
        base_per_cluster = sample_size // n_clusters
        remaining = sample_size - (base_per_cluster * n_clusters)
    
    # Count documents per cluster
    cluster_counts = {cluster: np.sum(labels == cluster) for cluster in unique_clusters}
    
    # Calculate proportional allocation
    total_docs = len(texts)
    proportions = {cluster: count/total_docs for cluster, count in cluster_counts.items()}
    
    # Allocate remaining samples proportionally
    extra_samples = {cluster: int(remaining * prop) for cluster, prop in proportions.items()}
    
    # Ensure we use exactly sample_size by adjusting the largest cluster if needed
    allocated = sum(base_per_cluster + extra_samples[c] for c in unique_clusters)
    if allocated < sample_size:
        # Find largest cluster and add the difference
        largest_cluster = max(cluster_counts, key=cluster_counts.get)
        extra_samples[largest_cluster] += sample_size - allocated
    
    # Sample from each cluster
    sampled_indices = []
    for cluster in unique_clusters:
        cluster_indices = np.where(labels == cluster)[0]
        cluster_sample_size = base_per_cluster + extra_samples[cluster]
        
        # Ensure we don't try to sample more than available
        cluster_sample_size = min(cluster_sample_size, len(cluster_indices))
        
        if len(cluster_indices) > 0:
            # Sample from this cluster
            sampled_from_cluster = np.random.choice(
                cluster_indices, 
                size=cluster_sample_size, 
                replace=False
            )
            sampled_indices.extend(sampled_from_cluster)
    
    return sampled_indices

def sample_text_data(texts, feature_matrix, sample_size, 
                     batch_size=10000, 
                     dimred_method='pca', 
                     n_components=50, 
                     n_trials=20,
                     min_per_cluster=1):
    """
    Complete pipeline for sampling text data using fast clustering.
    
    Parameters:
    -----------
    texts : list
        List of text documents
    feature_matrix : 2D array or sparse matrix
        Feature representation of texts (e.g., TF-IDF, embeddings)
    sample_size : int
        Desired sample size
    batch_size : int
        Size of batches for processing large datasets
    dimred_method : str
        Dimensionality reduction method ('pca' or 'tsne')
    n_components : int
        Number of components for dimensionality reduction
    n_trials : int
        Number of Optuna trials for hyperparameter tuning
    min_per_cluster : int
        Minimum samples to take from each cluster
        
    Returns:
    --------
    list
        Sampled text documents
    list
        Indices of sampled documents
    """
    # For very large datasets, use batching
    if len(texts) > batch_size:
        labels = batch_process(
            feature_matrix, 
            batch_size=batch_size, 
            dimred_method=dimred_method, 
            n_components=n_components, 
            n_trials=n_trials
        )
    else:
        # For smaller datasets, process all at once
        X_reduced = fast_dim_reduction(
            feature_matrix, 
            method=dimred_method, 
            n_components=n_components
        )
        best_params, _ = optuna_birch_tuning(X_reduced, n_trials=n_trials)
        labels = birch_clustering(X_reduced, **best_params)
        
        # Verify we got the expected number of clusters
        n_clusters = best_params.get('n_clusters')
        actual_clusters = len(np.unique(labels))
        if actual_clusters != n_clusters:
            print(f"Warning: Expected {n_clusters} clusters but got {actual_clusters}")
    
    # Sample based on clusters
    sampled_indices = stratified_cluster_sampling(
        texts, 
        labels, 
        sample_size, 
        min_per_cluster=min_per_cluster
    )
    
    # Get the actual sampled texts
    sampled_texts = [texts[i] for i in sampled_indices]
    
    return sampled_texts, sampled_indices