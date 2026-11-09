#!/usr/bin/env python3

import logging
from typing import Any, Dict, List, Optional

import flwr as fl

from MEDfl.LearningManager.federated_dataset import FederatedDataset
from MEDfl.LearningManager.server import setup_fl_logging

from .client import XGBoostSimulationClient
from .strategy import XGBoostSimulationStrategy

from MEDfl.LearningManager.shap import (
    SHAPConfig,
    LocalXGBoostSHAPExplainer,
    aggregate_federated_shap,
)


class XGBoostSimulationServer:
    """
    Flower/Ray simulation server for federated XGBoost.

    Important:
    Ray must serialize client_fn and everything captured by it.
    Therefore client_fn must not capture the server instance, a SQLAlchemy
    connection, a DatabaseManager, or another non-serializable object.
    """

    backend = "xgboost"

    def __init__(
        self,
        strategy: XGBoostSimulationStrategy,
        num_rounds: int,
        fed_dataset: FederatedDataset,
        task: str = "binary",
        xgb_params: Optional[Dict[str, Any]] = None,
        local_num_boost_round: int = 10,
        threshold: float = 0.5,
        feature_names: Optional[List[str]] = None,
        client_resources: Optional[Dict[str, float]] = None,
        shap_config: Optional[SHAPConfig] = None,
    ):
        self.strategy = strategy
        self.num_rounds = int(num_rounds)
        self.fed_dataset = fed_dataset

        self.task = str(task)
        self.xgb_params = dict(xgb_params or {})
        self.local_num_boost_round = int(
            local_num_boost_round
        )
        self.threshold = float(threshold)

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else getattr(
                fed_dataset,
                "feature_names",
                None,
            )
        )

        self.client_resources = (
            dict(client_resources)
            if client_resources is not None
            else {
                "num_cpus": 1.0,
                "num_gpus": 0.0,
            }
        )

        self.shap_config = (
            shap_config
            if shap_config is not None
            else SHAPConfig(enabled=False)
        )

        self.num_clients = len(
            self.fed_dataset.trainloaders
        )

        self.history = None
        self.shap_results = None

        self._validate()

        self.strategy.strategy_object.min_available_clients = (
            self.num_clients
        )

    def _validate(self) -> None:
        if self.strategy.strategy_object is None:
            raise ValueError(
                "strategy.create_strategy() must be called before "
                "creating XGBoostSimulationServer."
            )

        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be greater than zero.")

        if self.num_clients <= 0:
            raise ValueError(
                "The federated dataset does not contain training clients."
            )

        if len(self.fed_dataset.valloaders) < self.num_clients:
            raise ValueError(
                "A validation-loader entry is required for every client."
            )

        if self.task not in {
            "binary",
            "regression",
            "multiclass",
        }:
            raise ValueError(
                f"Unsupported task '{self.task}'."
            )

    @staticmethod
    def _clean_loader(loader):
        """
        Return None for an unavailable or empty loader.
        """

        if loader is None:
            return None

        try:
            if len(loader.dataset) == 0:
                return None
        except Exception:
            pass

        return loader

    def _detach_database_state(self) -> None:
        """
        Remove live database objects before Ray serializes simulation data.

        The loaders are already in memory, so the XGBoost simulation no longer
        requires the database connection.
        """

        if hasattr(self.fed_dataset, "eng"):
            try:
                connection = self.fed_dataset.eng

                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
            finally:
                self.fed_dataset.eng = None

        # Defensive cleanup in case these attributes are added later.
        for attribute_name in (
            "engine",
            "connection",
            "db_manager",
            "database_manager",
        ):
            if hasattr(self.fed_dataset, attribute_name):
                setattr(
                    self.fed_dataset,
                    attribute_name,
                    None,
                )
    
    @staticmethod
    def _get_loader_input_size(loader) -> int:
        if loader is None:
            raise ValueError(
                "Cannot determine the input size from a None loader"
            )

        for batch in loader:
            if not isinstance(batch, (tuple, list)):
                raise TypeError(
                    "Expected each DataLoader batch to be a "
                    "tuple or list"
                )

            if len(batch) == 0:
                continue

            features = batch[0]

            if hasattr(features, "shape"):
                shape = features.shape
            else:
                features = np.asarray(features)
                shape = features.shape

            if len(shape) != 2:
                raise ValueError(
                    "XGBoost SHAP expects tabular input with shape "
                    "(samples, features). "
                    f"Received shape {shape}."
                )

            return int(shape[1])

        raise ValueError(
            "Cannot determine input size from an empty DataLoader"
        )
    
    def _get_shap_loaders(self):
        """
        Return the selected local partition for every client.
        """

        data_split = self.shap_config.data_split

        if data_split == "train":
            source_loaders = self.fed_dataset.trainloaders

        elif data_split == "validation":
            source_loaders = self.fed_dataset.valloaders

        elif data_split == "test":
            source_loaders = self.fed_dataset.testloaders

        else:
            raise ValueError(
                f"Unsupported SHAP data split: {data_split}"
            )

        selected_loaders = []

        for client_index in range(self.num_clients):
            loader = None

            if client_index < len(source_loaders):
                loader = self._clean_loader(
                    source_loaders[client_index]
                )

            selected_loaders.append(loader)

        return selected_loaders
    
    def _resolve_shap_feature_names(
        self,
        input_size: int,
    ) -> List[str]:
        """
        Resolve feature names in this order:

        1. SHAPConfig.feature_names
        2. XGBoost server feature_names
        3. FederatedDataset feature_names
        4. Automatically generated feature_0, feature_1, ...
        """

        if self.shap_config.feature_names is not None:
            feature_names = list(
                self.shap_config.feature_names
            )

        elif self.feature_names is not None:
            feature_names = list(
                self.feature_names
            )

        else:
            dataset_feature_names = getattr(
                self.fed_dataset,
                "feature_names",
                None,
            )

            if dataset_feature_names is not None:
                feature_names = list(
                    dataset_feature_names
                )
            else:
                feature_names = (
                    self.shap_config.resolved_feature_names(
                        input_size
                    )
                )

        if len(feature_names) != input_size:
            raise ValueError(
                "The number of XGBoost SHAP feature names must "
                "match the input size. "
                f"Expected {input_size}, "
                f"received {len(feature_names)}."
            )

        return feature_names
    
    def _run_federated_shap(self):
        """
        Run the optional post-training federated SHAP phase.

        The same final global XGBoost booster is explained on each
        participating client's selected local data partition.
        """

        if not self.shap_config.enabled:
            print(
                "[XGBoost/SHAP] SHAP is disabled.",
                flush=True,
            )

            self.shap_results = None
            return None

        final_booster = self.strategy.current_booster

        if final_booster is None:
            raise RuntimeError(
                "Cannot calculate XGBoost SHAP because the "
                "federated strategy does not contain a final booster."
            )

        selected_loaders = self._get_shap_loaders()

        first_available_loader = None

        for loader in selected_loaders:
            if loader is not None:
                first_available_loader = loader
                break

        if first_available_loader is None:
            raise ValueError(
                "No client has data for the selected SHAP split: "
                f"{self.shap_config.data_split}"
            )

        input_size = self._get_loader_input_size(
            first_available_loader
        )

        feature_names = self._resolve_shap_feature_names(
            input_size
        )

        self.shap_config.validate_backend(
            input_size=input_size,
            backend="xgboost",
            task=self.task,
        )

        local_explainer = LocalXGBoostSHAPExplainer(
            background_size=(
                self.shap_config.background_size
            ),
            explanation_size=(
                self.shap_config.explanation_size
            ),
            random_seed=(
                self.shap_config.random_seed
            ),
            clipping_value=(
                self.shap_config.clipping_value
            ),
            model_output=(
                self.shap_config.model_output
            ),
        )

        client_results = []
        skipped_clients = []

        print(
            "[XGBoost/SHAP] Starting federated SHAP:",
            {
                "explainer": self.shap_config.explainer,
                "data_split": self.shap_config.data_split,
                "background_size": (
                    self.shap_config.background_size
                ),
                "explanation_size": (
                    self.shap_config.explanation_size
                ),
                "minimum_samples": (
                    self.shap_config.minimum_samples
                ),
                "model_output": (
                    self.shap_config.model_output
                ),
                "num_clients": self.num_clients,
            },
            flush=True,
        )

        for client_index, loader in enumerate(
            selected_loaders
        ):
            client_id = str(client_index)

            if loader is None:
                skipped_clients.append(
                    {
                        "client_id": client_id,
                        "reason": (
                            "Selected DataLoader is unavailable or empty"
                        ),
                    }
                )

                print(
                    f"[XGBoost/SHAP] Skipping client "
                    f"{client_id}: no data.",
                    flush=True,
                )

                continue

            try:
                available_samples = len(loader.dataset)
            except Exception:
                available_samples = None

            if (
                available_samples is not None
                and available_samples
                < self.shap_config.minimum_samples
            ):
                skipped_clients.append(
                    {
                        "client_id": client_id,
                        "reason": (
                            "Insufficient samples"
                        ),
                        "available_samples": int(
                            available_samples
                        ),
                        "minimum_samples": int(
                            self.shap_config.minimum_samples
                        ),
                    }
                )

                print(
                    f"[XGBoost/SHAP] Skipping client "
                    f"{client_id}: available={available_samples}, "
                    f"minimum={self.shap_config.minimum_samples}.",
                    flush=True,
                )

                continue

            print(
                f"[XGBoost/SHAP] Calculating SHAP for "
                f"client {client_id}.",
                flush=True,
            )

            try:
                local_result = local_explainer.calculate(
                    booster=final_booster,
                    loader=loader,
                )

                if (
                    local_result["sample_count"]
                    < self.shap_config.minimum_samples
                ):
                    skipped_clients.append(
                        {
                            "client_id": client_id,
                            "reason": (
                                "Explained sample count is below "
                                "minimum_samples"
                            ),
                            "explained_samples": int(
                                local_result[
                                    "sample_count"
                                ]
                            ),
                            "minimum_samples": int(
                                self.shap_config.minimum_samples
                            ),
                        }
                    )

                    continue

                local_result["client_id"] = client_id
                local_result["feature_names"] = list(
                    feature_names
                )

                client_results.append(local_result)

                print(
                    f"[XGBoost/SHAP] Client {client_id} "
                    f"completed: explained_samples="
                    f"{local_result['sample_count']}.",
                    flush=True,
                )

            except Exception as exc:
                skipped_clients.append(
                    {
                        "client_id": client_id,
                        "reason": str(exc),
                    }
                )

                print(
                    f"[XGBoost/SHAP] Client {client_id} failed: "
                    f"{exc}",
                    flush=True,
                )

        if not client_results:
            self.shap_results = {
                "status": "failed",
                "backend": "xgboost",
                "explainer": "tree",
                "data_split": (
                    self.shap_config.data_split
                ),
                "participating_clients": 0,
                "explained_samples": 0,
                "feature_importance": [],
                "skipped_clients": skipped_clients,
                "error": (
                    "No XGBoost client produced a valid "
                    "local SHAP result"
                ),
            }

            print(
                "[XGBoost/SHAP] Federated SHAP failed: "
                "no valid client results.",
                flush=True,
            )

            return self.shap_results

        federated_result = aggregate_federated_shap(
            client_results=client_results,
            include_client_results=(
                self.shap_config.include_client_results
            ),
        )

        federated_result.update(
            {
                "backend": "xgboost",
                "task": self.task,
                "explainer": "tree",
                "data_split": (
                    self.shap_config.data_split
                ),
                "background_size": int(
                    self.shap_config.background_size
                ),
                "requested_explanation_size": int(
                    self.shap_config.explanation_size
                ),
                "minimum_samples": int(
                    self.shap_config.minimum_samples
                ),
                "random_seed": int(
                    self.shap_config.random_seed
                ),
                "model_output": (
                    self.shap_config.model_output
                ),
                "feature_names": list(
                    feature_names
                ),
                "skipped_clients": skipped_clients,
            }
        )

        self.shap_results = federated_result

        print(
            "[XGBoost/SHAP] Federated SHAP completed:",
            {
                "participating_clients": (
                    federated_result[
                        "participating_clients"
                    ]
                ),
                "explained_samples": (
                    federated_result[
                        "explained_samples"
                    ]
                ),
                "skipped_clients": len(
                    skipped_clients
                ),
            },
            flush=True,
        )

        print(
            "[XGBoost/SHAP] Results:",
            federated_result,
            flush=True,
        )

        return federated_result

    def run(self):
        setup_fl_logging()

        self._detach_database_state()

        # -----------------------------------------------------------
        # Extract only values that Ray can serialize.
        #
        # Do not pass self.client_fn because that bound method captures
        # the complete server object, including fed_dataset and any DB state.
        # -----------------------------------------------------------

        trainloaders = list(
            self.fed_dataset.trainloaders
        )
        valloaders = list(
            self.fed_dataset.valloaders
        )
        testloaders = list(
            self.fed_dataset.testloaders
        )

        task = self.task
        xgb_params = dict(self.xgb_params)
        local_num_boost_round = int(
            self.local_num_boost_round
        )
        threshold = float(self.threshold)

        feature_names = (
            list(self.feature_names)
            if self.feature_names is not None
            else None
        )

        num_clients = int(self.num_clients)
        num_rounds = int(self.num_rounds)

        def client_fn(cid: str):
            """
            Ray-serializable client factory.

            This function captures only DataLoaders, primitive values,
            dictionaries, and lists. It does not capture the server or
            SQLAlchemy objects.
            """

            client_index = int(cid)

            if client_index < 0 or client_index >= num_clients:
                raise IndexError(
                    f"Invalid simulated client ID {cid}. "
                    f"Expected an index in [0, {num_clients - 1}]."
                )

            trainloader = trainloaders[client_index]

            valloader = None
            if client_index < len(valloaders):
                valloader = XGBoostSimulationServer._clean_loader(
                    valloaders[client_index]
                )

            testloader = None
            if client_index < len(testloaders):
                testloader = XGBoostSimulationServer._clean_loader(
                    testloaders[client_index]
                )

            client = XGBoostSimulationClient(
                cid=str(cid),
                trainloader=trainloader,
                valloader=valloader,
                testloader=testloader,
                task=task,
                xgb_params=xgb_params,
                local_num_boost_round=local_num_boost_round,
                threshold=threshold,
                feature_names=feature_names,
            )

            # XGBoostSimulationClient already subclasses fl.client.Client.
            return client

        print(
            "[XGBoostSimulationServer] Starting simulation:",
            {
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "task": task,
                "local_num_boost_round": local_num_boost_round,
                "client_resources": self.client_resources,
            },
            flush=True,
        )

        self.history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(
                num_rounds=num_rounds,
            ),
            strategy=self.strategy.strategy_object,
            client_resources=self.client_resources,
            ray_init_args={
                "include_dashboard": False,
                "log_to_driver": True,
                "logging_level": logging.INFO,
            },
        )

        # -----------------------------------------------------------
        # Optional post-training federated SHAP
        # -----------------------------------------------------------

        if self.shap_config.enabled:
            try:
                self._run_federated_shap()

            except Exception as exc:
                self.shap_results = {
                    "status": "failed",
                    "backend": "xgboost",
                    "explainer": "tree",
                    "data_split": (
                        self.shap_config.data_split
                    ),
                    "error": str(exc),
                }

                print(
                    "[XGBoost/SHAP] Federated SHAP failed:",
                    exc,
                    flush=True,
                )

        return self.history 