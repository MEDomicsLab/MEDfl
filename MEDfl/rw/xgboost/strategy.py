import json
import os
from typing import Dict, Any, Optional
import tempfile
import xgboost as xgb

import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    EvaluateIns,
    parameters_to_ndarrays,
)
from flwr.common import GetPropertiesIns

from MEDfl.rw.xgboost.config import XGBoostConfig
from MEDfl.rw.xgboost.metrics import aggregate_weighted_metrics
from MEDfl.rw.xgboost.serialization import (
    parameters_to_booster,booster_to_parameters,
    save_booster,
    XGBOOST_TENSOR_TYPE,
)
from MEDfl.LearningManager.shap import (
    SHAPConfig,
    aggregate_federated_shap,
)
from MEDfl.rw.holdout import HoldoutConfig, finalize_federated_holdout


class XGBoostStrategy:
    """
    Strategy wrapper for MEDfl real-world federated XGBoost.

    This class follows the same high-level contract as the existing MEDfl
    Strategy wrapper:
        - it exposes create_strategy();
        - it stores the Flower strategy in self.strategy_object;
        - it can be passed directly into FederatedServer.

    First supported mode:
        - bagging-style federated XGBoost using Flower strategy hooks.

    Notes
    -----
    This implementation is conservative and non-invasive:
      - It does not modify the existing PyTorch FedAvg/FedAdam/FedYogi path.
      - It uses Flower Parameters containing serialized XGBoost boosters.
      - It saves XGBoost models as .ubj files instead of .npz files.
    """

    def __init__(
        self,
        mode: str = "bagging",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        task: str = "binary",
        num_classes: Optional[int] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        local_num_boost_round: int = 10,
        threshold: float = 0.5,
        features: str = "",
        target: str = "",
        val_fraction: float = 0.10,
        test_fraction: float = 0.10,
        split_mode: str = "global",
        client_fractions: Optional[Dict[str, Dict[str, Any]]] = None,
        id_col: str = "id",
        savingPath: str = "",
        saveOnRounds: int = 1,
        total_rounds: int = 3,
        shap_config: Optional[SHAPConfig] = None,
        holdout_config: Optional[HoldoutConfig] = None,
    ):
        self.name = "FedXgbBagging"
        self.mode = mode

        if self.mode != "bagging":
            raise ValueError(
                "Only mode='bagging' is implemented in this first integration. "
                "Cyclic mode should be added after bagging is stable."
            )

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        self.task = task
        self.num_classes = num_classes
        self.threshold = threshold

        config = XGBoostConfig(
            task=task,
            mode=mode,
            local_num_boost_round=local_num_boost_round,
            params=xgb_params or {},
            num_classes=num_classes,
        )

        self.xgb_params = config.build_params()
        self.local_num_boost_round = local_num_boost_round

        self._features = features
        self._target = target
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction

        self.split_mode = split_mode
        self.client_fractions = client_fractions or {}
        self.id_col = id_col

        self.savingPath = savingPath
        self.saveOnRounds = saveOnRounds
        self.total_rounds = total_rounds

        self.shap_config = (
            shap_config
            if shap_config is not None
            else SHAPConfig(enabled=False)
        )
        self.federated_shap_result = {
            "status": (
                "pending"
                if self.shap_config.enabled
                else "disabled"
            )
        }
        self.local_shap_results = []

        self.holdout_config = (
            holdout_config
            if holdout_config is not None
            else HoldoutConfig(enabled=False)
        )
        self.holdout_config.validate()
        if self.holdout_config.enabled and self.task != "binary":
            raise ValueError(
                "The federated holdout test supports binary "
                f"classification only (task is '{self.task}')"
            )
        self.federated_holdout_result = {
            "status": (
                "pending"
                if self.holdout_config.enabled
                else "disabled"
            )
        }

        self.strategy_object = None
        self.current_booster: Optional[xgb.Booster] = None
        self.current_num_trees: int = 0

    def create_strategy(self) -> None:
        """
        Create the real Flower strategy object.

        We use FedAvg as the orchestration base but override aggregation behavior
        to keep XGBoost booster exchange explicit.
        """

        def fit_config_fn(server_round: int) -> Dict[str, Any]:
            return {
                "backend": "xgboost",
                "mode": self.mode,
                "task": self.task,
                "threshold": float(self.threshold),
                "features": self._features,
                "target": self._target,
                "val_fraction": float(self._val_fraction),
                "test_fraction": float(self._test_fraction),
                "id_col": self.id_col,
                "local_num_boost_round": int(self.local_num_boost_round),
                "xgb_params": json.dumps(self.xgb_params),
            }

        def evaluate_config_fn(server_round: int) -> Dict[str, Any]:
            is_final_round = (
                int(server_round) == int(self.total_rounds)
            )

            cfg = {
                "backend": "xgboost",
                "mode": self.mode,
                "task": self.task,
                "threshold": float(self.threshold),
                "features": self._features,
                "target": self._target,
                "val_fraction": float(self._val_fraction),
                "test_fraction": float(self._test_fraction),
                "id_col": self.id_col,
                "local_num_boost_round": int(
                    self.local_num_boost_round
                ),
                "xgb_params": json.dumps(self.xgb_params),
                "shap_enabled": bool(
                    self.shap_config.enabled
                    and is_final_round
                ),
            }

            if self.holdout_config.enabled and is_final_round:
                cfg.update(self.holdout_config.to_evaluate_config())

            if cfg["shap_enabled"]:
                cfg.update(
                    {
                        "shap_explainer": (
                            self.shap_config.explainer
                        ),
                        "shap_data_split": (
                            self.shap_config.data_split
                        ),
                        "shap_background_size": int(
                            self.shap_config.background_size
                        ),
                        "shap_explanation_size": int(
                            self.shap_config.explanation_size
                        ),
                        "shap_random_seed": int(
                            self.shap_config.random_seed
                        ),
                        "shap_minimum_samples": int(
                            self.shap_config.minimum_samples
                        ),
                        "shap_clipping_value": (
                            -1.0
                            if self.shap_config.clipping_value
                            is None
                            else float(
                                self.shap_config.clipping_value
                            )
                        ),
                        "shap_feature_names_json": json.dumps(
                            self.shap_config.feature_names or []
                        ),
                        "shap_model_output": (
                            self.shap_config.model_output
                        ),
                    }
                )

            return cfg

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_fit_clients,
            min_evaluate_clients=self.min_evaluate_clients,
            min_available_clients=self.min_available_clients,
            on_fit_config_fn=fit_config_fn,
            on_evaluate_config_fn=evaluate_config_fn,
            fit_metrics_aggregation_fn=aggregate_weighted_metrics,
            evaluate_metrics_aggregation_fn=aggregate_weighted_metrics,
            initial_parameters=Parameters(
                tensors=[],
                tensor_type=XGBOOST_TENSOR_TYPE,
            ),
        )

        self._wrap_configure_fit(strategy)
        self._wrap_configure_evaluate(strategy, evaluate_config_fn)
        self._wrap_aggregate_fit(strategy)
        self._wrap_aggregate_evaluate(strategy)

        self.strategy_object = strategy

    def _wrap_configure_fit(self, strategy) -> None:
        """
        Inject per-client test fractions / test IDs exactly like the existing MEDfl strategy.
        """

        original_configure_fit = strategy.configure_fit

        def wrapped_configure_fit(server_round, parameters, client_manager):
            selected = original_configure_fit(
                server_round=server_round,
                parameters=parameters,
                client_manager=client_manager,
            )

            ins = GetPropertiesIns(config={})

            for client, fit_ins in selected:
                hostname = None

                try:
                    props = client.get_properties(
                        ins=ins,
                        timeout=10.0,
                        group_id=0,
                    )
                    print(
                        f"\n📋 [XGBoost Round {server_round}] "
                        f"Client {client.cid} Properties: {props.properties}"
                    )
                    hostname = props.properties.get("hostname", None)

                except Exception as exc:
                    print(f"⚠️ Failed to get properties from {client.cid}: {exc}")

                if not hostname:
                    hostname = client.cid

                cfg = fit_ins.config

                if self.split_mode == "per_client":
                    per_cfg = (
                        self.client_fractions.get(hostname)
                        or self.client_fractions.get(client.cid)
                        or {}
                    )

                    if "val_fraction" in per_cfg:
                        try:
                            cfg["val_fraction"] = float(per_cfg["val_fraction"])
                        except Exception:
                            pass

                    if "test_ids" in per_cfg and per_cfg["test_ids"]:
                        test_ids_val = per_cfg["test_ids"]

                        if isinstance(test_ids_val, (list, tuple, set)):
                            test_ids_str = ",".join(str(x) for x in test_ids_val)
                        else:
                            test_ids_str = str(test_ids_val)

                        cfg["test_ids"] = test_ids_str

                        if "test_fraction" in cfg:
                            del cfg["test_fraction"]

                        cfg["id_col"] = self.id_col

                    else:
                        if "test_fraction" in per_cfg:
                            try:
                                cfg["test_fraction"] = float(per_cfg["test_fraction"])
                            except Exception:
                                pass

                else:
                    if "test_ids" in cfg:
                        del cfg["test_ids"]

                    cfg["val_fraction"] = float(self._val_fraction)
                    cfg["test_fraction"] = float(self._test_fraction)
                    cfg["id_col"] = self.id_col

            return selected

        strategy.configure_fit = wrapped_configure_fit

    def _wrap_configure_evaluate(
        self,
        strategy,
        evaluate_config_fn,
    ) -> None:
        """
        On the final round, request evaluation/SHAP/holdout from every
        currently connected client so the federated SHAP and holdout
        results represent all available institutions.
        """
        original_configure_evaluate = (
            strategy.configure_evaluate
        )

        def wrapped_configure_evaluate(
            server_round,
            parameters,
            client_manager,
        ):
            is_final_shap_round = (
                (self.shap_config.enabled or self.holdout_config.enabled)
                and int(server_round)
                == int(self.total_rounds)
            )

            if not is_final_shap_round:
                return original_configure_evaluate(
                    server_round=server_round,
                    parameters=parameters,
                    client_manager=client_manager,
                )

            evaluate_ins = EvaluateIns(
                parameters=parameters,
                config=evaluate_config_fn(server_round),
            )

            available_clients = list(
                client_manager.all().values()
            )

            print(
                "[Server/XGBoost] Final round: requesting the final "
                f"evaluation from {len(available_clients)} connected "
                f"client(s) (SHAP: {self.shap_config.enabled}, "
                f"holdout: {self.holdout_config.enabled}).",
                flush=True,
            )

            return [
                (client, evaluate_ins)
                for client in available_clients
            ]

        strategy.configure_evaluate = (
            wrapped_configure_evaluate
        )

    def _wrap_aggregate_fit(self, strategy) -> None:
        """
        Wrap aggregate_fit for XGBoost.

        IMPORTANT:
        XGBoost boosters are NOT NumPy tensors.
        Therefore we MUST NOT call FedAvg tensor aggregation.

        Current strategy:
        - select the booster from the client with the largest dataset
        - aggregate only metrics manually
        - save the selected booster
        """

        def wrapped_aggregate_fit(server_round, results, failures):
            print(f"\n[Server/XGBoost] 🔄 Round {server_round} - Client Training Metrics:")

            for client, fit_res in results:
                print(
                    f" XGB CTM Round {server_round} "
                    f"Client:{client.cid}: {fit_res.metrics}"
                )

            if failures:
                print(f"[Server/XGBoost] ⚠️ Round {server_round} failures: {failures}")

            if not results:
                print("[Server/XGBoost] No fit results received.")
                return None, {}

            # Aggregate metrics manually, same as before.
            metric_results = []

            for _, fit_res in results:
                metric_results.append(
                    (
                        fit_res.num_examples,
                        fit_res.metrics,
                    )
                )

            metrics = aggregate_weighted_metrics(metric_results)

            previous_num_trees = int(self.current_num_trees)
            new_local_boosters = []

            print(
                f"[Server/XGBoost] Previous global tree count before round "
                f"{server_round}: {previous_num_trees}"
            )

            for client, fit_res in results:
                client_booster = parameters_to_booster(fit_res.parameters)

                if client_booster is None:
                    print(
                        f"[Server/XGBoost] Client {client.cid} returned no booster. Skipping."
                    )
                    continue

                client_total_trees = self._num_boosted_rounds(client_booster)

                print(
                    f"[Server/XGBoost] Client {client.cid} returned booster with "
                    f"{client_total_trees} trees."
                )

                # In round 1, previous_num_trees is 0, so this keeps all local trees.
                # In later rounds, this keeps only the new trees added by the client.
                new_trees_booster = self._slice_new_trees(
                    client_booster=client_booster,
                    previous_num_trees=previous_num_trees,
                )

                if new_trees_booster is None:
                    continue

                new_tree_count = self._num_boosted_rounds(new_trees_booster)

                print(
                    f"[Server/XGBoost] Client {client.cid} contributed "
                    f"{new_tree_count} new trees."
                )

                new_local_boosters.append(new_trees_booster)

            if not new_local_boosters:
                print(
                    "[Server/XGBoost] No new local trees received. "
                    "Keeping previous global booster."
                )

                if self.current_booster is None:
                    return None, metrics

                agg_parameters = booster_to_parameters(self.current_booster)
                return agg_parameters, metrics

            boosters_to_combine = []

            if self.current_booster is not None:
                boosters_to_combine.append(self.current_booster)

            boosters_to_combine.extend(new_local_boosters)

            combined_booster = self._combine_boosters_by_json(boosters_to_combine)

            if combined_booster is None:
                print("[Server/XGBoost] Failed to combine boosters.")
                return None, metrics

            self.current_booster = combined_booster
            self.current_num_trees = self._num_boosted_rounds(combined_booster)

            print(
                f"[Server/XGBoost] ✅ Round {server_round} - Flower-style bagging aggregation done."
            )
            print(
                f"[Server/XGBoost] Global booster now has {self.current_num_trees} trees."
            )
            print(f"[Server/XGBoost] Aggregated metrics: {metrics}\n")

            # Optional: print importance directly, no saving required.
            if server_round == self.total_rounds:
                self._print_feature_importance(combined_booster, server_round)

            agg_parameters = booster_to_parameters(combined_booster)

            # Optional: keep your existing save behavior if savingPath is provided.
            self._save_if_needed(server_round, agg_parameters)

            return agg_parameters, metrics
        strategy.aggregate_fit = wrapped_aggregate_fit


    def _wrap_aggregate_evaluate(self, strategy) -> None:
        """
        Wrap aggregate_evaluate for logging and final federated SHAP and
        holdout aggregation.
        """

        original_aggregate_evaluate = (
            strategy.aggregate_evaluate
        )

        def wrapped_aggregate_evaluate(
            server_round,
            results,
            failures,
        ):
            print(
                f"\n[Server/XGBoost] 📊 Round {server_round} - "
                "Client Evaluation Metrics:"
            )

            for client, eval_res in results:
                print(
                    f" XGB CEM Round {server_round} "
                    f"Client:{client.cid}: {eval_res.metrics}"
                )

            loss, metrics = original_aggregate_evaluate(
                server_round,
                results,
                failures,
            )

            print(
                f"[Server/XGBoost] ✅ Round {server_round} - "
                "Aggregated Evaluation:"
            )
            print(
                f"    Loss: {loss}, Metrics: {metrics}\n"
            )

            is_final_round = (
                int(server_round) == int(self.total_rounds)
            )

            if self.shap_config.enabled and is_final_round:
                local_results = []
                shap_failures = []

                for client_proxy, eval_res in results:
                    client_metrics = eval_res.metrics or {}

                    if (
                        client_metrics.get("shap_status")
                        == "completed"
                        and client_metrics.get("shap_result")
                    ):
                        try:
                            local_result = json.loads(
                                client_metrics["shap_result"]
                            )
                            local_results.append(
                                local_result
                            )
                        except Exception as exc:
                            shap_failures.append(
                                {
                                    "client_id": client_proxy.cid,
                                    "error": (
                                        "Invalid SHAP JSON: "
                                        f"{exc}"
                                    ),
                                }
                            )
                    else:
                        shap_failures.append(
                            {
                                "client_id": client_proxy.cid,
                                "error": client_metrics.get(
                                    "shap_error",
                                    "No SHAP result returned",
                                ),
                            }
                        )

                self.local_shap_results = local_results

                if local_results:
                    try:
                        federated_result = (
                            aggregate_federated_shap(
                                client_results=local_results,
                                include_client_results=(
                                    self.shap_config
                                    .include_client_results
                                ),
                            )
                        )

                        federated_result["backend"] = (
                            "xgboost"
                        )
                        federated_result["explainer"] = (
                            self.shap_config.explainer
                        )
                        federated_result["data_split"] = (
                            self.shap_config.data_split
                        )
                        federated_result["background_size"] = (
                            self.shap_config.background_size
                        )
                        federated_result[
                            "requested_explanation_size"
                        ] = (
                            self.shap_config.explanation_size
                        )
                        federated_result["model_output"] = (
                            self.shap_config.model_output
                        )

                        if shap_failures:
                            federated_result[
                                "failed_clients"
                            ] = shap_failures

                        self.federated_shap_result = (
                            federated_result
                        )

                        print(
                            "[Server/SHAP] Federated SHAP "
                            "completed:"
                        )
                        print(
                            json.dumps(
                                self.federated_shap_result,
                                indent=2,
                            ),
                            flush=True,
                        )

                    except Exception as exc:
                        self.federated_shap_result = {
                            "status": "failed",
                            "backend": "xgboost",
                            "error": str(exc),
                            "failed_clients": shap_failures,
                        }

                        print(
                            "[Server/SHAP] XGBoost federated "
                            f"SHAP aggregation failed: {exc}",
                            flush=True,
                        )

                else:
                    self.federated_shap_result = {
                        "status": "failed",
                        "backend": "xgboost",
                        "error": (
                            "SHAP was enabled but no XGBoost "
                            "client returned a valid SHAP result."
                        ),
                        "failed_clients": shap_failures,
                    }

                    print(
                        "[Server/SHAP] No valid XGBoost local "
                        "SHAP results were returned.",
                        flush=True,
                    )

            if self.holdout_config.enabled and is_final_round:
                self.federated_holdout_result = (
                    finalize_federated_holdout(
                        results,
                        failures,
                        self.holdout_config,
                        backend="xgboost",
                    )
                )

            return loss, metrics

        strategy.aggregate_evaluate = (
            wrapped_aggregate_evaluate
        )

    def _save_if_needed(self, server_round: int, parameters: Parameters) -> None:
        """
        Save XGBoost model as .ubj.
        """

        if not self.savingPath:
            return

        should_save = (
            server_round % self.saveOnRounds == 0
            or server_round == self.total_rounds
        )

        if not should_save:
            return

        os.makedirs(self.savingPath, exist_ok=True)

        booster = parameters_to_booster(parameters)

        if booster is None:
            print("[Server/XGBoost] No booster to save.")
            return

        filename = (
            f"round_{server_round}_final_model.ubj"
            if server_round == self.total_rounds
            else f"round_{server_round}_model.ubj"
        )

        filepath = os.path.join(self.savingPath, filename)

        save_booster(booster, filepath)

        print(f"[Server/XGBoost] 💾 Saved booster to: {filepath}")
    def _num_boosted_rounds(self, booster: xgb.Booster) -> int:
        """
        Return the number of boosted rounds/trees in the booster.
        """
        try:
            return int(booster.num_boosted_rounds())
        except Exception:
            try:
                return len(booster.get_dump())
            except Exception:
                return 0


    def _slice_new_trees(
        self,
        client_booster: xgb.Booster,
        previous_num_trees: int,
    ) -> Optional[xgb.Booster]:
        """
        Extract only the trees added by the client in this round.

        Client booster = previous global booster + new local trees.
        We keep only new local trees.
        """

        total_trees = self._num_boosted_rounds(client_booster)

        if total_trees <= previous_num_trees:
            print(
                "[Server/XGBoost] Client booster has no new trees. "
                f"total_trees={total_trees}, previous_num_trees={previous_num_trees}"
            )
            return None

        try:
            return client_booster[previous_num_trees:total_trees]
        except Exception as exc:
            print(f"[Server/XGBoost] Failed to slice new trees: {exc}")
            return None


    def _booster_to_raw_bytes(self, booster: xgb.Booster) -> bytes:
        """
        Serialize booster to raw UBJ bytes.
        """
        with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            booster.save_model(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


    def _booster_from_raw_bytes(self, raw: bytes) -> Optional[xgb.Booster]:
        """
        Deserialize booster from raw UBJ bytes.
        """
        if not raw:
            return None

        with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(raw)

        try:
            booster = xgb.Booster()
            booster.load_model(tmp_path)
            return booster
        except Exception as exc:
            print(f"[Server/XGBoost] Failed to load booster from bytes: {exc}")
            return None
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def _combine_boosters_by_json(
        self,
        boosters: list[xgb.Booster],
    ) -> Optional[xgb.Booster]:
        """
        Combine multiple XGBoost boosters by concatenating their trees.

        Fixes:
        - updates trees
        - updates tree_info
        - updates gbtree_model_param["num_trees"]
        - updates iteration_indptr

        This is the missing part that caused:
        Check failed: model.iteration_indptr.back() == model.param.num_trees
        """

        boosters = [b for b in boosters if b is not None]

        if not boosters:
            return None

        if len(boosters) == 1:
            return boosters[0]

        model_jsons = []

        for booster in boosters:
            raw_json = booster.save_raw(raw_format="json")
            if isinstance(raw_json, bytes):
                raw_json = raw_json.decode("utf-8")
            model_jsons.append(json.loads(raw_json))

        base = model_jsons[0]

        base_model = base["learner"]["gradient_booster"]["model"]
        base_trees = base_model["trees"]
        base_tree_info = base_model["tree_info"]

        # Start from the base model iteration structure
        base_iteration_indptr = base_model.get("iteration_indptr", None)

        if base_iteration_indptr is None:
            # Safe fallback for binary/regression: one tree per iteration
            combined_iteration_indptr = list(range(0, len(base_trees) + 1))
        else:
            combined_iteration_indptr = list(base_iteration_indptr)

        next_tree_id = len(base_trees)

        for extra in model_jsons[1:]:
            extra_model = extra["learner"]["gradient_booster"]["model"]

            extra_trees = extra_model["trees"]
            extra_tree_info = extra_model["tree_info"]

            # Append trees and reassign IDs
            for tree in extra_trees:
                tree["id"] = next_tree_id
                base_trees.append(tree)
                next_tree_id += 1

            # Append tree_info
            base_tree_info.extend(extra_tree_info)

            # Append iteration_indptr correctly
            extra_iteration_indptr = extra_model.get("iteration_indptr", None)

            if extra_iteration_indptr is None:
                # Fallback: one tree per boosting iteration
                extra_iteration_indptr = list(range(0, len(extra_trees) + 1))

            offset = combined_iteration_indptr[-1]

            # Skip the first 0 from the extra booster
            for ptr in extra_iteration_indptr[1:]:
                combined_iteration_indptr.append(offset + ptr)

        # Write back merged model fields
        base_model["trees"] = base_trees
        base_model["tree_info"] = base_tree_info
        base_model["iteration_indptr"] = combined_iteration_indptr

        # Update num_trees
        if "gbtree_model_param" in base_model:
            base_model["gbtree_model_param"]["num_trees"] = str(len(base_trees))

        # Save/load to validate the merged model
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp_path = tmp.name
            json.dump(base, tmp)

        try:
            combined = xgb.Booster()
            combined.load_model(tmp_path)
            return combined
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    def _print_feature_importance(
        self,
        booster: xgb.Booster,
        server_round: int,
    ) -> None:
        """
        Print feature importance for the current global booster.
        """
        if booster is None:
            return

        importance_gain = booster.get_score(importance_type="gain")
        importance_weight = booster.get_score(importance_type="weight")
        importance_cover = booster.get_score(importance_type="cover")

      

        print(f"\n[Server/XGBoost] 📊 Round {server_round} Global Feature Importance - Gain:")
        print(importance_gain)

        print(f"[Server/XGBoost] 📊 Round {server_round} Global Feature Importance - Weight:")
        print(importance_weight)

        print(f"[Server/XGBoost] 📊 Round {server_round} Global Feature Importance - Cover:")
        print(importance_cover)