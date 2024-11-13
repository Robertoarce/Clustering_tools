import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from typing import Dict, List, Tuple, Any
import seaborn as sns
import matplotlib.pyplot as plt


class ColAnalyzer:
    def __init__(self, df, max_unique_ratio: float = 0.05, 
                 date_format: str = None):
        """
        Initialize the column analyzer
        
        Parameters:
        -----------
        max_unique_ratio: Maximum ratio of unique values to total values for categorical data
        date_format: Optional date format string for date detection
        """
        self.max_unique_ratio = max_unique_ratio
        self.date_format = date_format
        self.column_types_ = {}
        self.column_stats_ = {}
        self.analyze_columns(df)
        
    def _is_numeric(self, series: pd.Series) -> bool:
        """Check if a series contains numeric data"""
        return pd.api.types.is_numeric_dtype(series) and  series.dtype != bool
        
    def _is_datetime(self, series: pd.Series) -> bool:
        """Check if a series contains datetime data"""
        return pd.api.types.is_datetime64_any_dtype(series)
    
    def _is_categorical(self, series: pd.Series) -> bool:
        """
        Check if a series should be treated as categorical based on:
        1. Explicit categorical dtype
        2. Object dtype with low unique value ratio
        3. Boolean dtype
        4. Numeric dtype with low unique values
        """
        if pd.api.types.is_categorical_dtype(series) or series.dtype == bool:
            return True
            
        n_unique = series.nunique()
        n_total = len(series)
        
        # Handle numeric columns with few unique values
        if pd.api.types.is_numeric_dtype(series):
            return n_unique / n_total < self.max_unique_ratio
            
        # Handle object dtype
        if pd.api.types.is_object_dtype(series):
            return n_unique / n_total < self.max_unique_ratio
            
        return False
    
    def analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze DataFrame columns and classify them by type
        
        Returns:
        --------
        Dictionary containing column classifications and statistics
        """
        self.column_types_ = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'other': []
        }
        
        self.column_stats_ = {}
        
        for column in df.columns:
            series = df[column]
            
            # Classify column type
            if self._is_datetime(series):
                self.column_types_['datetime'].append(column)
                stats = {
                    'type': 'datetime',
                    'missing': series.isnull().sum(),
                    'missing_pct': (series.isnull().sum() / len(series)) * 100,
                    'min': series.min(),
                    'max': series.max()
                }
                
            elif self._is_numeric(series) and not self._is_categorical(series):
                self.column_types_['numeric'].append(column)
                stats = {
                    'type': 'numeric',
                    'missing': series.isnull().sum(),
                    'missing_pct': (series.isnull().sum() / len(series)) * 100,
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'skew': series.skew(),
                    'unique_values': series.nunique(),
                    'unique_ratio': series.nunique() / len(series)
                }
                
            elif self._is_categorical(series):
                self.column_types_['categorical'].append(column)
                value_counts = series.value_counts()
                stats = {
                    'type': 'categorical',
                    'missing': series.isnull().sum(),
                    'missing_pct': (series.isnull().sum() / len(series)) * 100,
                    'unique_values': series.nunique(),
                    'unique_ratio': series.nunique() / len(series),
                    'most_common': value_counts.index[0] if not value_counts.empty else None,
                    'most_common_count': value_counts.iloc[0] if not value_counts.empty else 0,
                    'value_counts': value_counts
                }
                
            else:
                self.column_types_['other'].append(column)
                stats = {
                    'type': 'other',
                    'missing': series.isnull().sum(),
                    'missing_pct': (series.isnull().sum() / len(series)) * 100,
                    'unique_values': series.nunique()
                }
            
            self.column_stats_[column] = stats
            
        return self.column_types_
    
    def get_column_summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame of column statistics
        """
        summaries = []
        for column, stats in self.column_stats_.items():
            summary = {
                'Column': column,
                'Type': stats['type'],
                'Missing': stats['missing'],
                'Missing %': stats['missing_pct']
            }
            
            if stats['type'] == 'numeric':
                summary.update({
                    'Mean': stats['mean'],
                    'Std': stats['std'],
                    'Min': stats['min'],
                    'Max': stats['max'],
                    'Skew': stats['skew'],
                    'Unique Values': stats['unique_values']
                })
            elif stats['type'] == 'categorical':
                summary.update({
                    'Unique Values': stats['unique_values'],
                    'Most Common': stats['most_common'],
                    'Most Common Count': stats['most_common_count']
                })
                
        return pd.DataFrame(summaries)
    
    def plot_column_analysis(self, df: pd.DataFrame):
        """
        Generate visualizations for column analysis
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Column types distribution
        plt.subplot(221)
        type_counts = {t: len(cols) for t, cols in self.column_types_.items() if len(cols) > 0}
        plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        plt.title('Distribution of Column Types')
        
        # Plot 2: Missing values
        plt.subplot(222)
        missing_data = pd.Series({col: stats['missing_pct'] 
                                for col, stats in self.column_stats_.items()})
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            missing_data.plot(kind='bar')
            plt.title('Missing Values by Column (%)')
            plt.xticks(rotation=45)
        
        # Plot 3: Numeric columns distribution
        if self.column_types_['numeric']:
            plt.subplot(223)
            df[self.column_types_['numeric']].boxplot()
            plt.title('Distribution of Numeric Columns')
            plt.xticks(rotation=45)
        
        # Plot 4: Categorical columns distribution
        if self.column_types_['categorical']:
            plt.subplot(224)
            cat_unique = pd.Series({col: self.column_stats_[col]['unique_values'] 
                                  for col in self.column_types_['categorical']})
            cat_unique.plot(kind='bar')
            plt.title('Unique Values in Categorical Columns')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plots for detailed distribution analysis
        self._plot_detailed_distributions(df)
    
    def _plot_detailed_distributions(self, df: pd.DataFrame):
        """
        Plot detailed distributions for numeric and categorical columns
        """
        # Numeric distributions
        if self.column_types_['numeric']:
            n_numeric = len(self.column_types_['numeric'])
            fig, axes = plt.subplots(n_numeric, 2, figsize=(12, 4*n_numeric))
            
            if n_numeric == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(self.column_types_['numeric']):
                # Histogram
                sns.histplot(data=df, x=col, ax=axes[i,0])
                axes[i,0].set_title(f'{col} Distribution')
                
                # QQ plot
                from scipy import stats
                stats.probplot(df[col].dropna(), dist="norm", plot=axes[i,1])
                axes[i,1].set_title(f'{col} Q-Q Plot')
            
            plt.tight_layout()
            plt.show()
        
        # Categorical distributions
        if self.column_types_['categorical']:
            n_cat = len(self.column_types_['categorical'])
            fig, axes = plt.subplots(int(np.ceil(n_cat/2)), 2, figsize=(12, 4*int(np.ceil(n_cat/2))))
            
            if n_cat == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(self.column_types_['categorical']):
                row, col_idx = i // 2, i % 2
                df[col].value_counts().plot(kind='bar', ax=axes[row,col_idx])
                axes[row,col_idx].set_title(f'{col} Value Distribution')
                axes[row,col_idx].tick_params(axis='x', rotation=45)
            
            # Hide empty subplots if odd number of categorical columns
            if n_cat % 2 != 0:
                axes[-1,-1].set_visible(False)
            
            plt.tight_layout()
            plt.show()

 #Example of usage 
    # analyzer = ColumnAnalyzer(max_unique_ratio=0.05)
    # column_types = analyzer.analyze_columns(data)
    
    # Print results
    # print("\nColumn Classifications:")
    # for type_name, columns in column_types.items():
    #     if columns:
    #         print(f"\n{type_name.capitalize()}:")
    #         print(columns)
    
    # # Get detailed summary
    # print("\nColumn Summary:")
    # print(analyzer.get_column_summary())
    
    # # Plot analysis
    # analyzer.plot_column_analysis(data)
