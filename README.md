# Clustering Tools work Room

## Overview

This project is designed for copying, studying, and using pre-made algorithms for clustering. The notebook `main.ipynb` covers various clustering techniques, data generation, visualization, and evaluation metrics.

## Clustering Algorithms Comparison

| Algorithm      | Best Use Case                              | Pros                                         | Cons                                        | Complexity |
|----------------|--------------------------------------------|----------------------------------------------|---------------------------------------------|------------|
| **K-Means**    | Large datasets with spherical clusters     | Fast and simple, Scalable, Easy to understand | Requires k value upfront, Sensitive to outliers, Only finds spherical clusters | Medium     |
| **DBSCAN**     | Irregular shapes and noisy data            | No need to specify clusters, Handles noise, Finds arbitrary shapes | Sensitive to parameters, Struggles with varying densities | Medium     |
| **Hierarchical** | Small-medium datasets needing cluster relationships | Creates visual hierarchy, No k needed, Good for visualization | Computationally expensive, Memory intensive for large datasets | High       |
| **Mean Shift** | Non-spherical clusters with unknown count  | Automatically finds clusters, No cluster assumption, Robust to outliers | Slower than K-means, Computationally expensive | High       |

### Best Practices for Algorithm Selection

1. **K-Means**:
   - Spherical clusters
   - Large datasets
   - Known number of clusters

2. **DBSCAN**:
   - Noisy data
   - Irregular shapes
   - Unknown number of clusters

3. **Hierarchical**:
   - Dendrogram visualization
   - Small to medium-sized datasets
   - Important cluster relationships

4. **Mean Shift**:
   - Irregular shapes
   - Unknown number of clusters
   - Computation time is not a concern

## Functions

- `adjusted_r2_score(y_true, y_pred, n_features)`: Calculates the adjusted R² score.
- `create_dimension_reduction_plots(X, y, method='pca', title_prefix='')`: Creates 2D visualizations of data reduced by PCA, t-SNE, or UMAP.
- `log_plot(X, y, method, title_prefix, artifact_name)`: Creates and logs a 2D visualization of data reduced by the specified method.
- `ch_scorer(estimator, X)`: Calinski-Harabasz scorer for clustering algorithms.
- `silhouette_scorer(estimator, X)`: Silhouette scorer for clustering algorithms.
- `plot_clustering_results(df, feature_columns, cluster_column='cluster', figsize=(20, 15))`: Creates comprehensive visualizations for pre-clustered data.

## Data Creation

### Regression Data

- Generates regression data and plots the results using `DataGenerator`.

## Visuals

- 2D and 3D scatter plots for clustering results.
- Feature correlation heatmap.
- Cluster profiles with standardized feature means.

## Dependencies

- Python libraries: `os`, `pandas`, `numpy`, `scipy`, `seaborn`, `matplotlib`, `wandb`, `mlflow`, `sklearn`, `umap-learn`

## Running the Notebook

1. Clone the repository:
   ```sh
   git clone https://github.com/Robertoarce/ML_Work_Room.git
   cd ML_Work_Room
   ```
