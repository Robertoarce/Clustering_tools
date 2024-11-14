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