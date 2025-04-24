# Embedding-based Stratified Sampling

A personal approach for stratified sampling of text data using embedding-based clustering techniques.

## Overview

This module provides utilities for stratified sampling from large text datasets using clustering of embeddings. By grouping semantically similar texts into clusters before sampling, we ensure proper representation across the entire semantic distribution.

## How It Works

1. **Embedding Preparation**: Text data is converted to embeddings (handled outside this pipeline)
2. **Dimensionality Reduction**: PCA or t-SNE is applied to make high-dimensional embeddings more manageable 
3. **Adaptive Clustering**: The BIRCH clustering algorithm identifies semantic clusters within the data
4. **Hyperparameter Optimization**: Optuna automatically tunes parameters with validation of cluster counts
5. **Stratified Sampling**: Samples are drawn proportionally from all semantic clusters
6. **Scalable Processing**: Large datasets are handled through efficient batch processing

## Why Stratified Sampling?

Random sampling often fails to represent minority classes or rare examples in your data. This stratified approach:

- Guarantees representation from all semantic clusters
- Provides better coverage of edge cases and outliers
- Preserves the distribution of concepts while reducing sample size
- Creates more balanced and representative samples

## Key Features

- **Cluster Count Validation**: Ensures the clustering algorithm produces the expected number of clusters
- **Efficient Implementation**: Optimized for large datasets with sampling for metrics calculation
- **Multiple Quality Metrics**: Support for silhouette, Davies-Bouldin, and Calinski-Harabasz indices
- **Visualization Tools**: Functions to plot optimization history and parameter importance
- **Flexible Dimensionality Reduction**: Choose between PCA or t-SNE based on your needs

## Usage Examples

See the accompanying Jupyter notebook for detailed examples and usage patterns. 