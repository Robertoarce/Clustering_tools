import os
from typing import Any, Dict, Optional, Union, List
import mlflow
import wandb
from datetime import datetime
from abc import ABC, abstractmethod

# Base abstract class for tracking
class BaseTracker(ABC):
    @abstractmethod
    def start_run(self, run_name: Optional[str] = None) -> None:
        pass
    
    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass
    
    @abstractmethod
    def set_tags(self, tags: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def end_run(self) -> None:
        pass

class MLFlowTracker(BaseTracker):
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.tags = tags or {}
        self.config = config or {}
        self.run = None

    def start_run(self, run_name: Optional[str] = None) -> None:
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(run_name=run_name, tags=self.tags)
        if self.config:
            mlflow.log_params(self.config)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        mlflow.set_tags(tags)

    def end_run(self) -> None:
        mlflow.end_run()

class WandBTracker(BaseTracker):
    def __init__(self, project_name: str, experiment_name: str,
                 tags: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.tags = tags or {}
        self.config = config or {}
        self.run = None

    def start_run(self, run_name: Optional[str] = None) -> None:
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = wandb.init(
            project=self.project_name,
            name=run_name,
            config=self.config,
            tags=self.tags,
            reinit=True
        )

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        metrics = {key: value}
        if step is not None:
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if step is not None:
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)

    def log_param(self, key: str, value: Any) -> None:
        self.run.config[key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            self.run.config[key] = value

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        self.run.save(local_path)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        for key, value in tags.items():
            self.run.tags[key] = value

    def end_run(self) -> None:
        self.run.finish()

class ExperimentTracker:
    def __init__(
        self,
        tracking_platform: str = "mlflow",
        experiment_name: str = "default_experiment",
        project_name: str = "default_project",
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.tracking_platform = tracking_platform.lower()
        if self.tracking_platform not in ["mlflow", "wandb"]:
            raise ValueError("tracking_platform must be either 'mlflow' or 'wandb'")
        
        if self.tracking_platform == "mlflow":
            self.tracker = MLFlowTracker(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                tags=tags,
                config=config
            )
        else:
            self.tracker = WandBTracker(
                project_name=project_name,
                experiment_name=experiment_name,
                tags=tags,
                config=config
            )

    def __getattr__(self, name):
        # Delegate all method calls to the appropriate tracker
        return getattr(self.tracker, name)

    def __enter__(self):
        self.tracker.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.end_run()