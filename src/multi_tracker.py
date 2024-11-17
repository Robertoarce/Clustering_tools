import mlflow
import wandb
from mlflow.entities import Run as MlflowRun
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Generator, Union
from dataclasses import dataclass
import os

class BaseTracker(ABC):
    @abstractmethod
    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> Any:
        pass
    
    @abstractmethod
    def end_run(self) -> None:
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass
    
    @abstractmethod
    def set_tags(self, tags: Dict[str, Any]) -> None:
        pass

class MLflowTracker(BaseTracker):
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.active_run = None
        mlflow.set_experiment(experiment_name)
        #kill any previous run
        try:
            mlflow.end_run()
        except Exception:
            pass
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> mlflow.ActiveRun:
        if self.active_run is None or not nested:
            # Start a new parent run
            self.active_run = mlflow.start_run(run_name=run_name)
        else:
            # Start a nested run
            self.active_run = mlflow.start_run(run_name=run_name, nested=True)
        return self.active_run
    
    def end_run(self) -> None:
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(metrics)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)
    
    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)
    
    def set_tags(self, tags: Dict[str, Any]) -> None:
        mlflow.set_tags(tags)

class WandbTracker(BaseTracker):
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self._tags = set()
        self.parent_run = None
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> wandb.sdk.wandb_run.Run:
        tags = list(self._tags) if self._tags else None
        
        if not nested:
            # Start a new parent run
            self.parent_run = wandb.init(
                project=self.project_name,
                name=run_name,
                group=self.experiment_name,
                reinit=True,
                tags=tags,
                config={}
            )
            return self.parent_run
        else:
            # Start a new child run
            return wandb.init(
                project=self.project_name,
                name=run_name,
                group=self.experiment_name,
                reinit=True,
                tags=tags,
                config={},
                job_type="child"
            )
    
    def end_run(self) -> None:
        if wandb.run is not None:
            wandb.finish()
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if wandb.run is not None:
            wandb.log(metrics)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        if wandb.run is not None:
            wandb.config.update(params)
    
    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=os.path.basename(local_path),
                type='dataset' if artifact_path is None else artifact_path
            )
            artifact.add_file(local_path)
            wandb.log_artifact(artifact)
    
    def set_tags(self, tags: Dict[str, Any]) -> None:
        if wandb.run is not None:
            # Convert tag key-value pairs to wandb config
            wandb.config.update({f"tag_{k}": v for k, v in tags.items()})
            
            # Update tags set
            self._tags.update(tags.keys())
            
            # Update wandb run tags
            if wandb.run.tags:
                new_tags = set(wandb.run.tags) | set(tags.keys())
            else:
                new_tags = set(tags.keys())
            wandb.run.tags = tuple(new_tags)

class MultiTracker:
    def __init__(
        self,
        tracking_platforms: List[str],
        experiment_name: str,
        project_name: str,
        tags: Optional[Dict[str, Any]] = None
    ):
        self.trackers: List[BaseTracker] = []
        self.active_runs: List[Any] = []
        self.tags = tags or {}
        
        for platform in tracking_platforms:
            if platform.lower() == "mlflow":
                self.trackers.append(MLflowTracker(experiment_name))
            elif platform.lower() == "wandb":
                self.trackers.append(WandbTracker(project_name, experiment_name))
            else:
                raise ValueError(f"Unsupported tracking platform: {platform}")
    
    def __enter__(self) -> 'MultiTracker':
        self.active_runs = []
        for tracker in self.trackers:
            run = tracker.start_run()
            self.active_runs.append(run)
            if self.tags:
                tracker.set_tags(self.tags)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for tracker in self.trackers:
            tracker.end_run()
        self.active_runs = []
    
    @contextmanager
    def child_run(self, run_name: str) -> Generator[List[Any], None, None]:
        child_runs = []
        for tracker in self.trackers:
            run = tracker.start_run(run_name=run_name, nested=True)
            if self.tags:
                tracker.set_tags(self.tags)
            child_runs.append(run)
        
        try:
            yield child_runs
        finally:
            for tracker in self.trackers:
                tracker.end_run()
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.log_params(params)
    
    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        for tracker in self.trackers:
            tracker.log_artifacts(local_path, artifact_path)
    
    def set_tags(self, tags: Dict[str, Any]) -> None:
        self.tags.update(tags)
        for tracker in self.trackers:
            tracker.set_tags(tags)


#############
#  Example  #
#############


# # Initialize the multi-platform tracker
# tracker = MultiTracker(
#     tracking_platforms=["mlflow", "wandb"],
#     experiment_name="clustering_comparison",
#     project_name="clustering_analysis",
#     tags={"task": "clustering", "dataset": data.shape}
# )



# #Get columns info
# data = dg.data.copy()
# cola = ColAnalyzer(data).column_types_

# # Defining column transformations for preprocessing
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), cola['numeric']),
#         ('cat', OneHotEncoder(), cola['categorical'])
#     ]
# )

# # Model Pipelines with parameter grids
# model_pipelines = {
#     'kmeans': {
#         'model': KMeans(),
#         'params': {'model__n_clusters': [2, 3, 4]}
#     },
#     'gmm': {
#         'model': GaussianMixture(),
#         'params': {'model__n_components': [2, 3, 4]}
#     },
#     'dbscan': {
#         'model': DBSCAN(),
#         'params': {'model__eps': [0.3, 0.5, 0.7], 'model__min_samples': [5, 10]}
#     },
#     'spectral': {
#         'model': SpectralClustering(),
#         'params': {'model__n_clusters': [2, 3, 4], 'model__affinity': ['nearest_neighbors', 'rbf']}
#     }
# }

# # Set up K-Fold cross-validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)


# # Dictionary to store cross-validated results for each model
# results = {}



# from sklearn.metrics import make_scorer, calinski_harabasz_score, silhouette_score

# def ch_scorer(estimator, X):
#     """Calinski-Harabasz scorer for GridSearchCV"""
#     labels = estimator.named_steps['model'].fit_predict(X)
#     return calinski_harabasz_score(X, labels)

# def silhouette_scorer(estimator, X):
#     """Silhouette scorer for GridSearchCV"""
#     labels = estimator.named_steps['model'].fit_predict(X)
#     return silhouette_score(X, labels)

# # Create custom scorers
# scoring = {
#     'calinski_harabasz': make_scorer(ch_scorer),
#     'silhouette': make_scorer(silhouette_scorer)
# }

# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import make_scorer, calinski_harabasz_score, silhouette_score
# from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
# from sklearn.mixture import GaussianMixture
# import mlflow
# import wandb
# from abc import ABC, abstractmethod
# from contextlib import contextmanager
# from typing import List, Dict, Any, Optional, Generator, Union
# from dataclasses import dataclass
# import os

# class BaseTracker(ABC):
#     @abstractmethod
#     def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> Any:
#         pass
    
#     @abstractmethod
#     def end_run(self) -> None:
#         pass
    
#     @abstractmethod
#     def log_metrics(self, metrics: Dict[str, float]) -> None:
#         pass
    
#     @abstractmethod
#     def log_params(self, params: Dict[str, Any]) -> None:
#         pass
    
#     @abstractmethod
#     def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
#         pass
    
#     @abstractmethod
#     def set_tags(self, tags: Dict[str, Any]) -> None:
#         pass

# class MLflowTracker(BaseTracker):
#     def __init__(self, experiment_name: str):
#         self.experiment_name = experiment_name
#         self.active_run = None
#         mlflow.set_experiment(experiment_name)
#         try:
#             mlflow.end_run()
#         except Exception:
#             pass
    
#     def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> mlflow.ActiveRun:
#         if nested:
#             self.active_run = mlflow.start_run(run_name=run_name, nested=True)
#         else:
#             if mlflow.active_run():
#                 mlflow.end_run()
#             self.active_run = mlflow.start_run(run_name=run_name)
#         return self.active_run
    
#     def end_run(self) -> None:
#         if self.active_run:
#             mlflow.end_run()
#             self.active_run = None
    
#     def log_metrics(self, metrics: Dict[str, float]) -> None:
#         if self.active_run:
#             metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
#             mlflow.log_metrics(metrics)
    
#     def log_params(self, params: Dict[str, Any]) -> None:
#         if self.active_run:
#             mlflow.log_params(params)
    
#     def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
#         if self.active_run:
#             mlflow.log_artifact(local_path, artifact_path)
    
#     def set_tags(self, tags: Dict[str, Any]) -> None:
#         if self.active_run:
#             mlflow.set_tags(tags)

# class WandbTracker(BaseTracker):
#     def __init__(self, project_name: str, experiment_name: str):
#         self.project_name = project_name
#         self.experiment_name = experiment_name
#         self._tags = set()
#         self.parent_run = None
    
#     def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> wandb.sdk.wandb_run.Run:
#         tags = list(self._tags) if self._tags else None
        
#         if not nested:
#             self.parent_run = wandb.init(
#                 project=self.project_name,
#                 name=run_name,
#                 group=self.experiment_name,
#                 reinit=True,
#                 tags=tags,
#                 config={}
#             )
#             return self.parent_run
#         else:
#             return wandb.init(
#                 project=self.project_name,
#                 name=run_name,
#                 group=self.experiment_name,
#                 reinit=True,
#                 tags=tags,
#                 config={},
#                 job_type="child"
#             )
    
#     def end_run(self) -> None:
#         if wandb.run is not None:
#             wandb.finish()
    
#     def log_metrics(self, metrics: Dict[str, float]) -> None:
#         if wandb.run is not None:
#             wandb.log(metrics)
    
#     def log_params(self, params: Dict[str, Any]) -> None:
#         if wandb.run is not None:
#             wandb.config.update(params)
    
#     def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
#         if wandb.run is not None:
#             artifact = wandb.Artifact(
#                 name=os.path.basename(local_path),
#                 type='dataset' if artifact_path is None else artifact_path
#             )
#             artifact.add_file(local_path)
#             wandb.log_artifact(artifact)
    
#     def set_tags(self, tags: Dict[str, Any]) -> None:
#         if wandb.run is not None:
#             wandb.config.update({f"tag_{k}": v for k, v in tags.items()})
#             self._tags.update(tags.keys())
#             if wandb.run.tags:
#                 new_tags = set(wandb.run.tags) | set(tags.keys())
#             else:
#                 new_tags = set(tags.keys())
#             wandb.run.tags = tuple(new_tags)

# class MultiTracker:
#     def __init__(
#         self,
#         tracking_platforms: List[str],
#         experiment_name: str,
#         project_name: str,
#         tags: Optional[Dict[str, Any]] = None
#     ):
#         self.trackers: List[BaseTracker] = []
#         self.active_runs: List[Any] = []
#         self.tags = tags or {}
        
#         for platform in tracking_platforms:
#             if platform.lower() == "mlflow":
#                 self.trackers.append(MLflowTracker(experiment_name))
#             elif platform.lower() == "wandb":
#                 self.trackers.append(WandbTracker(project_name, experiment_name))
#             else:
#                 raise ValueError(f"Unsupported tracking platform: {platform}")
    
#     def __enter__(self) -> 'MultiTracker':
#         self.active_runs = []
#         for tracker in self.trackers:
#             run = tracker.start_run()
#             self.active_runs.append(run)
#             if self.tags:
#                 tracker.set_tags(self.tags)
#         return self
    
#     def __exit__(self, exc_type, exc_val, exc_tb) -> None:
#         for tracker in self.trackers:
#             tracker.end_run()
#         self.active_runs = []
    
#     @contextmanager
#     def child_run(self, run_name: str) -> Generator[List[Any], None, None]:
#         child_runs = []
#         for tracker in self.trackers:
#             run = tracker.start_run(run_name=run_name, nested=True)
#             child_runs.append(run)
        
#         try:
#             yield child_runs
#         finally:
#             for tracker in self.trackers:
#                 tracker.end_run()
    
#     def log_metrics(self, metrics: Dict[str, float]) -> None:
#         for tracker in self.trackers:
#             if isinstance(tracker, MLflowTracker):
#                 mlflow_metrics = {k: float(v) for k, v in metrics.items() 
#                                 if isinstance(v, (int, float))}
#                 tracker.log_metrics(mlflow_metrics)
#             else:
#                 tracker.log_metrics(metrics)
    
#     def log_params(self, params: Dict[str, Any]) -> None:
#         for tracker in self.trackers:
#             tracker.log_params(params)
    
#     def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
#         for tracker in self.trackers:
#             tracker.log_artifacts(local_path, artifact_path)
    
#     def set_tags(self, tags: Dict[str, Any]) -> None:
#         self.tags.update(tags)
#         for tracker in self.trackers:
#             tracker.set_tags(tags)

# # Custom scoring functions for clustering metrics
# def ch_scorer(estimator, X):
#     """Calinski-Harabasz scorer for GridSearchCV"""
#     labels = estimator.named_steps['model'].fit_predict(X)
#     return calinski_harabasz_score(X, labels)

# def silhouette_scorer(estimator, X):
#     """Silhouette scorer for GridSearchCV"""
#     labels = estimator.named_steps['model'].fit_predict(X)
#     return silhouette_score(X, labels)

# # Create custom scorers
# scoring = {
#     'calinski_harabasz': make_scorer(ch_scorer),
#     'silhouette': make_scorer(silhouette_scorer)
# }

# # Get columns info
# data = dg.data.copy()
# cola = ColAnalyzer(data).column_types_

# # Defining column transformations for preprocessing
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), cola['numeric']),
#         ('cat', OneHotEncoder(), cola['categorical'])
#     ]
# )

# # Model Pipelines with parameter grids
# model_pipelines = {
#     'kmeans': {
#         'model': KMeans(),
#         'params': {'model__n_clusters': [2, 3, 4]}
#     },
#     'gmm': {
#         'model': GaussianMixture(),
#         'params': {'model__n_components': [2, 3, 4]}
#     },
#     'dbscan': {
#         'model': DBSCAN(),
#         'params': {'model__eps': [0.3, 0.5, 0.7], 'model__min_samples': [5, 10]}
#     },
#     'spectral': {
#         'model': SpectralClustering(),
#         'params': {'model__n_clusters': [2, 3, 4], 'model__affinity': ['nearest_neighbors', 'rbf']}
#     }
# }

# # Set up K-Fold cross-validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Initialize the multi-platform tracker
# tracker = MultiTracker(
#     tracking_platforms=["mlflow", "wandb"],
#     experiment_name="clustering_comparison",
#     project_name="clustering_analysis",
#     tags={"task": "clustering", "dataset": str(data.shape)}
# )

# # Start tracking the experiments
# with tracker:
#     # Log dataset information
#     tracker.log_params({
#         "n_samples": int(data.shape[0]),
#         "n_features": int(data.shape[1]),
#         "numeric_features": len(cola['numeric']),
#         "categorical_features": len(cola['categorical'])
#     })

#     # Loop through each model
#     for model_name, model_info in model_pipelines.items():
#         with tracker.child_run(f"{model_name}_evaluation") as runs:
#             silhouette_scores = []
#             calinski_harabasz_scores = []
            
#             # Create the pipeline
#             pipeline = Pipeline([
#                 ('preprocessor', preprocessor),
#                 ('model', model_info['model'])
#             ])
            
#             # K-Fold Cross-Validation
#             for fold_idx, (train_index, test_index) in enumerate(kf.split(data)):
#                 with tracker.child_run(f"{model_name}_fold_{fold_idx}") as fold_runs:
#                     try:
#                         # Split the data
#                         X_train = data.iloc[train_index]
#                         X_test = data.iloc[test_index]
                        
#                         # Grid search for the current fold
#                         grid_search = GridSearchCV(
#                             pipeline,
#                             param_grid=model_info['params'],
#                             scoring=scoring,
#                             refit='calinski_harabasz',
#                             n_jobs=-1
#                         )
                        
#                         # Fit the model
#                         grid_search.fit(X_train)
                        
#                         # Get the best model
#                         best_model = grid_search.best_estimator_
                        
#                         # Transform test data and get predictions
#                         X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
#                         if isinstance(X_test_transformed, np.ndarray):
#                             X_test_transformed_dense = X_test_transformed
#                         else:  # If sparse matrix
#                             X_test_transformed_dense = X_test_transformed.toarray()
                        
#                         labels = best_model.named_steps['model'].predict(X_test_transformed)
                        
#                         # Calculate metrics
#                         silhouette_avg = silhouette_score(X_test_transformed_dense, labels)
#                         calinski_harabasz = calinski_harabasz_score(X_test_transformed_dense, labels)
                        
#                         # Store scores
#                         silhouette_scores.append(silhouette_avg)
#                         calinski_harabasz_scores.append(calinski_harabasz)
                        
#                         # Log numeric metrics for both platforms
#                         numeric_metrics = {
#                             "silhouette_score": float(silhouette_avg),
#                             "calinski_harabasz_score": float(calinski_harabasz),
#                             "fold": float(fold_idx),
#                             "best_score": float(grid_search.best_score_)
#                         }
#                         tracker.log_metrics(numeric_metrics)
                        
#                         # Log additional info for W&B only
#                         for run in fold_runs:
#                             if isinstance(run, wandb.sdk.wandb_run.Run):
#                                 wandb.log({
#                                     **numeric_metrics,
#                                     "best_params": str(grid_search.best_params_)
#                                 })
                    
#                     except Exception as e:
#                         print(f"Error in fold {fold_idx}: {str(e)}")
#                         continue
            
#             # Calculate and log average metrics across folds
#             avg_metrics = {
#                 "avg_silhouette_score": float(np.mean(silhouette_scores)),
#                 "std_silhouette_score": float(np.std(silhouette_scores)),
#                 "avg_calinski_harabasz_score": float(np.mean(calinski_harabasz_scores)),
#                 "std_calinski_harabasz_score": float(np.std(calinski_harabasz_scores))
#             }
            
#             # Log final metrics
#             tracker.log_metrics(avg_metrics)
            