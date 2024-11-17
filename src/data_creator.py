import pandas as pd
import numpy as np

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons, make_circles, make_regression

class DataGenerator:
  """
  A class that allows you to create different data sets for learning and using models.
  """
  def __init__(self ):
    self.seed = np.random.seed(42)
    self.data = None
    self.type = None 

  # CLUSTERING

  def generate_cluster(self, dataset_type='blobs', n_samples=300, noise=0.1, random_state=42, centers=4):
      """
      Generate synthetic data for clustering practice.
      
      Parameters:
      dataset_type: str, options: 'blobs', 'moons', 'circles'
      n_samples: int, number of samples to generate
      noise: float, noise level in the data
      random_state: int, random seed for reproducibility
      
      Returns:
      pandas.DataFrame with the generated data
      """
      
      if dataset_type == 'blobs':
          # Generate isotropic Gaussian blobs
          X, y = make_blobs(
              n_samples=n_samples,
              centers=centers,
              n_features=2,
              cluster_std=noise*3,
              random_state=random_state
          )
          
      elif dataset_type == 'moons':
          # Generate two interleaving half circles
          X, y = make_moons(
              n_samples=n_samples,
              noise=noise,
              random_state=random_state
          )
          
      elif dataset_type == 'circles':
          # Generate concentric circles
          X, y = make_circles(
              n_samples=n_samples,
              noise=noise,
              factor=0.5,
              random_state=random_state
          )
      
      # Create DataFrame
      df = pd.DataFrame(X, columns=['feature1', 'feature2'])
      df['true_cluster'] = y
      
      # Add some additional features for more complex clustering scenarios
      df['feature3'] = np.sin(df['feature1']) + np.random.normal(0, noise, size=n_samples)
      df['feature4'] = np.cos(df['feature2']) + np.random.normal(0, noise, size=n_samples)
      
      # Add categorical feature
      df['category'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
      
      self.data = df
      self.type = 'cluster'
      return df

  # Function to visualize the dataset
  def plot_clusters(self, title = 'Clustering'):
      df = self.data
      plt.figure(figsize=(8, 6))
      plt.scatter(df['feature1'], df['feature2'], c=df['true_cluster'], cmap='viridis')
      plt.title(title)
      plt.xlabel('Feature 1')
      plt.ylabel('Feature 2')
      plt.colorbar(label='True Cluster')
      plt.show()

# REGRESSION

  def generate_regression(self, data_type='linear', n_samples=1000, noise=0.1, random_state=42):
    """
    Generate synthetic data for regression practice with different relationships.
    
    Parameters:
    data_type: str, options: 'linear', 'polynomial', 'sinusoidal', 'complex'
    n_samples: int, number of samples to generate
    noise: float, noise level in the data
    random_state: int, random seed for reproducibility
    
    Returns:
    pandas.DataFrame with the generated data
    """
    np.random.seed(random_state)
    
    # Base features
    X = np.linspace(0, 10, n_samples)
    
    if data_type == 'linear':
        # Simple linear relationship with multiple features
        X_expanded, y = make_regression(
            n_samples=n_samples,
            n_features=3,
            n_informative=2,
            noise=noise * 30,
            random_state=random_state
        )
        
        df = pd.DataFrame(X_expanded, columns=['feature1', 'feature2', 'feature3'])
        df['target'] = y
        
    elif data_type == 'polynomial':
        # Polynomial relationship
        y = 2 + 3*X + 2*X**2 - 0.1*X**3
        y += np.random.normal(0, noise * np.std(y), n_samples)
        
        df = pd.DataFrame({
            'feature1': X,
            'feature2': X**2,
            'feature3': X**3,
            'target': y
        })
        
    elif data_type == 'sinusoidal':
        # Sinusoidal relationship
        y = 3 * np.sin(X) + 2 * np.cos(X/2)
        y += np.random.normal(0, noise * np.std(y), n_samples)
        
        df = pd.DataFrame({
            'feature1': X,
            'feature2': np.sin(X/2),
            'feature3': np.cos(X/3),
            'target': y
        })
        
    elif data_type == 'complex':
        # Complex relationship with interactions
        feature1 = np.random.normal(0, 1, n_samples)
        feature2 = np.random.normal(0, 1, n_samples)
        feature3 = np.random.normal(0, 1, n_samples)
        
        y = (
            2 * feature1 +
            3 * feature2 +
            -2 * feature3 +
            4 * feature1 * feature2 +
            np.exp(-feature3**2)
        )
        y += np.random.normal(0, noise * np.std(y), n_samples)
        
        df = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'target': y
        })
    
    # Add some categorical features
    df['category1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    df['category2'] = np.random.choice(['Low', 'Medium', 'High'], size=n_samples)
    
    # Add datetime feature
    base_date = pd.Timestamp('2024-01-01')
    df['date'] = [base_date + pd.Timedelta(days=x) for x in range(n_samples)]
    
    # Add seasonal component based on date
    df['seasonal_factor'] = np.sin(2 * np.pi * df.index / (n_samples/4))
    
    self.data = df
    self.type = 'regression'
    self.data_type = data_type
    return df


  def plot_regression(self):
    """
    Visualize the regression dataset.
    """
    df = self.data
    data_type = self.data_type
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Feature 1 vs Target
    plt.subplot(131)
    plt.scatter(df['feature1'], df['target'], alpha=0.5,  c=df['target'], cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Target')
    plt.title(f'{data_type.capitalize()}: Feature 1 vs Target')
    
    # Plot 2: Feature 2 vs Target
    plt.subplot(132)
    plt.scatter(df['feature2'], df['target'], alpha=0.5,  c=df['target'], cmap='viridis')
    plt.xlabel('Feature 2')
    plt.ylabel('Target')
    plt.title(f'{data_type.capitalize()}: Feature 2 vs Target')
    
    # Plot 3: Seasonal Component
    plt.subplot(133)
    plt.plot(df['date'][:100], df['seasonal_factor'][:100])
    plt.xlabel('Date')
    plt.ylabel('Seasonal Factor')
    plt.title('Seasonal Component (first 100 days)')
    
    plt.tight_layout()
    plt.show()


  def plot(self):
    if self.type == 'cluster':
        self.plot_clusters()
    else:
        self.plot_regression()