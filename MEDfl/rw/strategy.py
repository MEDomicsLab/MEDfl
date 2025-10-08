# MEDfl/rw/strategy.py

import os
import json
import numpy as np
import flwr as fl
from typing import Callable, Optional, Dict, Any, List, Tuple
from flwr.common import GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import time
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


# ===== unchanged aggregate_* for metrics (works for both) =====
def aggregate_fit_metrics(results: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total = max(sum(n for n, _ in results), 1)
    loss = sum(m.get("train_loss", 0.0) * n for n, m in results) / total
    acc  = sum(m.get("train_accuracy", 0.0) * n for n, m in results) / total
    auc  = sum(m.get("train_auc", 0.0) * n for n, m in results) / total
    return {"train_loss": loss, "train_accuracy": acc, "train_auc": auc}

def aggregate_eval_metrics(results: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total = max(sum(n for n, _ in results), 1)
    loss = sum(m.get("eval_loss", 0.0) * n for n, m in results) / total
    acc  = sum(m.get("eval_accuracy", 0.0) * n for n, m in results) / total
    auc  = sum(m.get("eval_auc", 0.0) * n for n, m in results) / total
    return {"eval_loss": loss, "eval_accuracy": acc, "eval_auc": auc}

# ========== NEW: helper to decode XGB booster parameters ==========
# --- helpers for XGB bytes <-> Booster (updated to JSON format) ---
def _booster_to_json_bytes(bst) -> bytes:
    """Return XGBoost Booster as JSON bytes (so we can edit trees)."""
    # xgboost>=1.7 supports raw_format='json'
    raw = bst.save_raw(raw_format='json')
    return bytes(raw)

def _ensure_json_bytes(raw_bytes: bytes) -> bytes:
    """If bytes are not JSON (binary model), try converting via Booster load+save."""
    try:
        # Quick sniff: JSON starts with '{' or '['
        b0 = raw_bytes.lstrip()[:1]
        if b0 in (b'{', b'['):
            return raw_bytes
    except Exception:
        pass
    # Convert binary -> JSON
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(bytearray(raw_bytes))
    return _booster_to_json_bytes(bst)

def _get_tree_nums(xgb_model_org: bytes) -> Tuple[int, int]:
    xgb_model = json.loads(bytearray(xgb_model_org))
    tree_num = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
    paral_tree_num = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"])
    return tree_num, paral_tree_num

def _aggregate_trees(bst_prev_org: Optional[bytes], bst_curr_org: bytes) -> bytes:
    """Conduct bagging aggregation by appending trees from current to previous."""
    bst_curr_org = _ensure_json_bytes(bst_curr_org)

    if not bst_prev_org:
        # First model in the round becomes the base
        return bst_curr_org

    bst_prev_org = _ensure_json_bytes(bst_prev_org)

    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    # Update counts
    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(
        tree_num_prev + paral_tree_num_curr
    )

    # Update iteration_indptr
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"]
    iteration_indptr.append(iteration_indptr[-1] + paral_tree_num_curr)

    # Append current trees, re-id them
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for t in range(paral_tree_num_curr):
        trees_curr[t]["id"] = tree_num_prev + t
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(trees_curr[t])
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    return bytes(json.dumps(bst_prev), "utf-8")

class Strategy:
    def __init__(
        self,
        name: str = "FedAvg",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[List[Any]] = None,
        evaluate_fn: Optional[Callable] = None,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        local_epochs: int = 1,
        threshold: float = 0.5,
        learning_rate: float = 0.01,
        optimizer_name: str = "SGD",
        savingPath: str = "",
        saveOnRounds: int = 3,
        total_rounds: int = 3,
        datasetConfig: Dict[str, Any] = {},
    ) -> None:
        self.name = name
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters or []
        self.evaluate_fn = evaluate_fn

        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn or aggregate_fit_metrics
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn or aggregate_eval_metrics

        self.local_epochs = local_epochs
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.savingPath = savingPath
        self.saveOnRounds = saveOnRounds
        self.total_rounds = total_rounds
        self.datasetConfig = datasetConfig

        self.strategy_object: Optional[fl.server.strategy.Strategy] = None

    def create_strategy(self) -> None:
        # ======== Branch: custom XGB strategy ========
        if self.name == "XGBoostBagging":
            self.strategy_object = self._create_xgb_bagging_strategy()
            return

        # ======== Default: use Flower built-in by name (e.g., FedAvg) ========
        StrategyClass = getattr(fl.server.strategy, self.name)

        # ======== Common params for all built-in strategies ========
        def fit_config_fn(server_round: int) -> Dict[str, Any]:
            return {"local_epochs": self.local_epochs, "threshold": self.threshold, "learning_rate": self.learning_rate, "optimizer": self.optimizer_name  , "dataset_config": self.datasetConfig}

        params: Dict[str, Any] = {
            "fraction_fit": self.fraction_fit,
            "fraction_evaluate": self.fraction_evaluate,
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "min_available_clients": self.min_available_clients,
            "evaluate_fn": self.evaluate_fn,
            "on_fit_config_fn": fit_config_fn,
            "fit_metrics_aggregation_fn": self.fit_metrics_aggregation_fn,
            "evaluate_metrics_aggregation_fn": self.evaluate_metrics_aggregation_fn,
        }
        if self.initial_parameters:
            params["initial_parameters"] = fl.common.ndarrays_to_parameters(self.initial_parameters)

        strat = StrategyClass(**params)

        # Wrap aggregate_fit for logging + optional saving
        original_agg_fit = strat.aggregate_fit

        def logged_agg_fit(server_round, results, failures):
            print(f"\n[Server] 🔄 Round {server_round} - Client Training Metrics:")
            for (client_id, fit_res) in results:
                print(f" CTM Round {server_round} Client:{client_id.cid}: {fit_res.metrics}")

            agg_params, metrics = original_agg_fit(server_round, results, failures)
            print(f"[Server] ✅ Round {server_round} - Aggregated Training Metrics: {metrics}\n")

            # ⬇️ Only try to save when we actually have parameters
            should_checkpoint = (
                self.savingPath
                and ((server_round % self.saveOnRounds == 0) or (self.total_rounds and server_round == self.total_rounds))
            )

            if should_checkpoint and agg_params is not None:
                try:
                    arrays = fl.common.parameters_to_ndarrays(agg_params)
                    filename = (
                        f"round_{server_round}_final_model.npz"
                        if (self.total_rounds and server_round == self.total_rounds)
                        else f"round_{server_round}_model.npz"
                    )
                    os.makedirs(self.savingPath, exist_ok=True)
                    np.savez(os.path.join(self.savingPath, filename), *arrays)
                    print(f"[Server] 💾 Saved checkpoint to {os.path.join(self.savingPath, filename)}")
                except Exception as e:
                    print(f"[Server] ⚠️ Skipped saving checkpoint (no parameters or conversion failed): {e}")
            elif should_checkpoint and agg_params is None:
                print("[Server] ⚠️ Skipped checkpoint: aggregate_fit returned None (no aggregated parameters this round).")

            return agg_params, metrics

        strat.aggregate_fit = logged_agg_fit


        original_agg_eval = strat.aggregate_evaluate
        def logged_agg_eval(server_round, results, failures):
            print(f"\n[Server] 📊 Round {server_round} - Client Evaluation Metrics:")
            for (client_id, eval_res) in results:
                print(f" CEM Round {server_round} Client:{client_id.cid}: {eval_res.metrics}")
            loss, metrics = original_agg_eval(server_round, results, failures)
            print(f"[Server] ✅ Round {server_round} - Aggregated Evaluation Metrics:\n    Loss: {loss}, Metrics: {metrics}\n")
            return loss, metrics
        strat.aggregate_evaluate = logged_agg_eval

        original_conf_fit = strat.configure_fit
        def wrapped_conf_fit(server_round: int, parameters, client_manager: ClientManager):
            selected = original_conf_fit(server_round=server_round, parameters=parameters, client_manager=client_manager)
            ins = GetPropertiesIns(config={})
            for client, _ in selected:
                try:
                    props = client.get_properties(ins=ins, timeout=10.0, group_id=0)
                    print(f"\n📋 [Round {server_round}] Client {client.cid} Properties: {props.properties}")
                except Exception as e:
                    print(f"⚠️ Failed to get properties from {client.cid}: {e}")
            return selected
        strat.configure_fit = wrapped_conf_fit

        self.strategy_object = strat

    # ---------- NEW: Custom XGBoost strategy ----------
    def _create_xgb_bagging_strategy_old(self) -> fl.server.strategy.Strategy:
        from flwr.common import FitIns, Parameters, ndarrays_to_parameters
        from flwr.server.client_manager import ClientManager

        class XGBoostBagging(fl.server.strategy.Strategy):
            def __init__(self, outer: "Strategy"):
                self.outer = outer
                self.global_parameters: Parameters | None = None

            def initialize_parameters(self, client_manager: ClientManager):
                return self.global_parameters

            def configure_fit(self, server_round, parameters, client_manager: ClientManager):
                # (unchanged)
                sample_n = max(self.outer.min_fit_clients, 1)
                clients = client_manager.sample(num_clients=sample_n, min_num_clients=self.outer.min_fit_clients)
                fit_ins = FitIns(parameters if parameters is not None else self.global_parameters,
                                {"xgb_rounds": self.outer.local_epochs if self.outer.local_epochs > 0 else 50})
                return [(client, fit_ins) for client in clients]

            def aggregate_fit(self, server_round, results, failures):
                # (unchanged: select-best)
                best = None
                for (client, fit_res) in results:
                    m = fit_res.metrics or {}
                    score = (m.get("train_auc", 0.0), m.get("train_accuracy", 0.0))
                    if (best is None) or (score > best["score"]):
                        best = {"score": score, "parameters": fit_res.parameters}
                if best is not None:
                    self.global_parameters = best["parameters"]

                metrics = self.outer.fit_metrics_aggregation_fn([(r.num_examples, r.metrics) for (_, r) in results])
                print(f"[Server-XGB] ✅ Round {server_round} - Selected best booster (AUC,ACC)={best['score'] if best else None}")

                # Optional saving (unchanged)
                if self.outer.savingPath and ((server_round % self.outer.saveOnRounds == 0) or
                                            (self.outer.total_rounds and server_round == self.outer.total_rounds)):
                    os.makedirs(self.outer.savingPath, exist_ok=True)
                    if self.global_parameters is not None:
                        raw = bytes(self.global_parameters.tensors[0]) if hasattr(self.global_parameters, "tensors") \
                            else bytes(self.global_parameters[0].tolist())
                        with open(os.path.join(self.outer.savingPath, f"round_{server_round}_xgb.model"), "wb") as f:
                            f.write(raw)

                return self.global_parameters, metrics

            def configure_evaluate(self, server_round, parameters, client_manager: ClientManager):
                # No centralized eval: return empty list
                return []

            def aggregate_evaluate(self, server_round, results, failures):
                if not results:
                    return 0.0, {}
                loss = 0.0
                metrics = self.outer.evaluate_metrics_aggregation_fn([(r.num_examples, r.metrics) for (_, r) in results])
                return loss, metrics

            # 🔧 ADD THIS METHOD to satisfy the abstract interface
            def evaluate(self, server_round, parameters):
                # No server-side evaluation; Flower expects Optional[Tuple[float, dict]]
                return None

        strat = XGBoostBagging(self)

        # Log client props, like the NN path
        original_conf_fit = strat.configure_fit
        def wrapped_conf_fit(server_round, parameters, client_manager):
            selected = original_conf_fit(server_round, parameters, client_manager)
            ins = GetPropertiesIns(config={})
            for client, _ in selected:
                try:
                    props = client.get_properties(ins=ins, timeout=10.0, group_id=0)
                    print(f"\n📋 [Round {server_round}] Client {client.cid} Properties: {props.properties}")
                except Exception as e:
                    print(f"⚠️ Failed to get properties from {client.cid}: {e}")
            return selected
        strat.configure_fit = wrapped_conf_fit

        return strat

    def _create_xgb_bagging_strategy(self) -> fl.server.strategy.Strategy:
        from flwr.common import FitIns, Parameters, ndarrays_to_parameters
        from flwr.common import Scalar
        from typing import Union, cast

        class XGBoostBagging(fl.server.strategy.Strategy):
            def __init__(self, outer: "Strategy"):
                self.outer = outer
                self.global_model_bytes: Optional[bytes] = None  # JSON bytes

            def initialize_parameters(self, client_manager: ClientManager):
                if self.global_model_bytes is None:
                    return None
                # wrap the JSON bytes into a uint8 numpy array for NumPyClient
                arr = np.frombuffer(self.global_model_bytes, dtype=np.uint8)
                return ndarrays_to_parameters([arr])


            def configure_fit(self, server_round: int, parameters, client_manager: ClientManager):
                # Sample exactly min_fit_clients (or at least 1)
                sample_n = max(self.outer.min_fit_clients, 1)
                clients = client_manager.sample(
                    num_clients=sample_n, min_num_clients=self.outer.min_fit_clients
                )

                # Tell clients how many trees to add and return
                cfg = {
                    "num_local_round": int(self.outer.local_epochs) if self.outer.local_epochs > 0 else 50,
                    "global_round": int(server_round),
                }
                fit_ins = FitIns(
                    parameters if parameters is not None else self.initialize_parameters(client_manager),
                    cfg,
                )
                return [(client, fit_ins) for client in clients]

            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, "flwr.common.FitRes"]],
                failures: List[Union[Tuple[ClientProxy, "flwr.common.FitRes"], BaseException]],
            ):
                # Accept failures policy like FedAvg (optional)
                if not results:
                    return None, {}
                if not getattr(self, "accept_failures", True) and failures:
                    return None, {}

                # Start from current global; append every client's newly trained trees
                global_bytes = self.global_model_bytes
                for _, fit_res in results:
                    arrays = parameters_to_ndarrays(fit_res.parameters)  # list of np.ndarrays
                    for arr in arrays:
                        raw = arr.tobytes()                  # recover the exact buffer we sent
                        raw_json = _ensure_json_bytes(raw)   # accept binary/JSON; normalize to JSON
                        # (Optional but recommended) slice only the last K trees trained locally:
                        # k = int(self.outer.local_epochs) if self.outer.local_epochs > 0 else 50
                        # raw_json = _slice_last_k_trees(raw_json, k)
                        global_bytes = _aggregate_trees(global_bytes, raw_json)


                self.global_model_bytes = global_bytes

                # Optional metric aggregation for logging
                metrics = self.outer.fit_metrics_aggregation_fn(
                    [(r.num_examples, r.metrics) for (_, r) in results]
                )

                # Optional checkpointing
                if self.outer.savingPath and (
                    (server_round % self.outer.saveOnRounds == 0)
                    or (self.outer.total_rounds and server_round == self.outer.total_rounds)
                ):
                    os.makedirs(self.outer.savingPath, exist_ok=True)
                    if self.global_model_bytes is not None:
                        with open(os.path.join(self.outer.savingPath, f"round_{server_round}_xgb.model"), "wb") as f:
                            f.write(self.global_model_bytes)


                arr = np.frombuffer(self.global_model_bytes, dtype=np.uint8)
                return ndarrays_to_parameters([arr]), metrics


            def configure_evaluate(self, server_round, parameters, client_manager: ClientManager):
                # You can keep distributed evaluation if your clients implement it;
                # otherwise return [] to disable.
                return []

            def aggregate_evaluate(self, server_round, results, failures):
                if not results:
                    return 0.0, {}
                loss = 0.0
                metrics = self.outer.evaluate_metrics_aggregation_fn(
                    [(r.num_examples, r.metrics) for (_, r) in results]
                )
                return loss, metrics

            def evaluate(self, server_round, parameters):
                # No server-side eval by default
                return None

        strat = XGBoostBagging(self)

        # (Optional) property logging identical to your previous wrapper
        original_conf_fit = strat.configure_fit
        def wrapped_conf_fit(server_round, parameters, client_manager):
            selected = original_conf_fit(server_round, parameters, client_manager)
            ins = GetPropertiesIns(config={})
            for client, _ in selected:
                try:
                    props = client.get_properties(ins=ins, timeout=10.0, group_id=0)
                    print(f"\n📋 [Round {server_round}] Client {client.cid} Properties: {props.properties}")
                except Exception as e:
                    print(f"⚠️ Failed to get properties from {client.cid}: {e}")
            return selected
        strat.configure_fit = wrapped_conf_fit

        return strat

