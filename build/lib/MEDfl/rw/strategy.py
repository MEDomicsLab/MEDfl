# File: MEDfl/rw/strategy.py

import os
import numpy as np
import flwr as fl
from flwr.common import GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import time
from MEDfl.rw.model import Net
import torch

# ===================================================
# Custom metric aggregation functions
# ===================================================
def aggregate_fit_metrics(results):
    total = sum(n for n, _ in results)
    loss = sum(m.get("train_loss", 0.0) * n for n, m in results) / total
    acc = sum(m.get("train_accuracy", 0.0) * n for n, m in results) / total
    auc = sum(m.get("train_auc", 0.0) * n for n, m in results) / total
    return {"train_loss": loss, "train_accuracy": acc, "train_auc": auc}

def aggregate_eval_metrics(results):
    total = sum(n for n, _ in results)
    loss = sum(m.get("eval_loss", 0.0) * n for n, m in results) / total
    acc = sum(m.get("eval_accuracy", 0.0) * n for n, m in results) / total
    auc = sum(m.get("eval_auc", 0.0) * n for n, m in results) / total
    return {"eval_loss": loss, "eval_accuracy": acc, "eval_auc": auc}

# ===================================================
# Strategy Wrapper
# ===================================================
class Strategy:
    """
    Flower Strategy wrapper:
      - Dynamic hyperparameters via on_fit_config_fn
      - Custom metric aggregation
      - Per-client & aggregated metric logging
      - Synchronous get_properties() inspection in configure_fit()
      - Saving global parameters every saveOnRounds to savingPath

      Extended:
      - split_mode:
          * "global": use global val_fraction/test_fraction for all clients
          * "per_client": use client_fractions[hostname] if present
      - client_fractions:
          {
            "HOSTNAME_1": {
               "val_fraction": float (optional),
               "test_fraction": float (optional),
               "test_ids": [..] or "id1,id2" (optional)
            },
            ...
          }
      - In per_client mode:
          * if test_ids is present for a client:
                -> send test_ids
                -> do NOT use that client's test_fraction
          * otherwise:
                -> use that client's val_fraction/test_fraction if provided,
                   else fall back to global val_fraction/test_fraction
      - client id in this mapping = hostname from client.get_properties()
      - id_col:
          * column name used on clients to match test_ids (default "id")
    """

    def __init__(
        self,
        name="FedAvg",
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=None,
        evaluate_fn=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        local_epochs=1,
        threshold=0.5,
        learning_rate=0.01,
        optimizer_name="SGD",
        savingPath="",
        saveOnRounds=3,
        total_rounds=3,
        features="",
        target="",
        val_fraction=0.10,
        test_fraction=0.10,
        # NEW: splitting control (added at the end to not break existing calls)
        split_mode="global",                  # "global" or "per_client"
        client_fractions=None,
        # NEW: id column for test_ids mapping
        id_col="id",
    ):
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

        # Dynamic hyperparams
        self.local_epochs = local_epochs
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.savingPath = savingPath
        self.saveOnRounds = saveOnRounds
        self.total_rounds = total_rounds
        self._features = features  # comma-separated or ""
        self._target = target      # or ""
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction

        # NEW
        self.split_mode = split_mode
        self.client_fractions = client_fractions or {}
        self.id_col = id_col

        self.strategy_object = None

    def create_strategy(self):
        # 1) Pick the Flower Strategy class
        StrategyClass = getattr(fl.server.strategy, self.name)

        # 2) Define on_fit_config_fn _before_ instantiation (global defaults)
        def fit_config_fn(server_round):
            return {
                "local_epochs":   self.local_epochs,
                "threshold":      self.threshold,
                "learning_rate":  self.learning_rate,
                "optimizer":      self.optimizer_name,
                "features":       self._features,
                "target":         self._target,
                "val_fraction":   float(self._val_fraction),
                "test_fraction":  float(self._test_fraction),
                # NEW: always send id_col so clients know which column to use for test_ids
                "id_col":         self.id_col,
            }

        # 3) Build params including on_fit_config_fn
        params = {
            "fraction_fit":                     self.fraction_fit,
            "fraction_evaluate":                self.fraction_evaluate,
            "min_fit_clients":                  self.min_fit_clients,
            "min_evaluate_clients":             self.min_evaluate_clients,
            "min_available_clients":            self.min_available_clients,
            "evaluate_fn":                      self.evaluate_fn,
            "on_fit_config_fn":                 fit_config_fn,
            "fit_metrics_aggregation_fn":       self.fit_metrics_aggregation_fn,
            "evaluate_metrics_aggregation_fn":  self.evaluate_metrics_aggregation_fn,
        }
        if self.initial_parameters:
            params["initial_parameters"] = fl.common.ndarrays_to_parameters(self.initial_parameters)
        else:
            # derive initial params from server-specified features
            feat_cols = [c.strip() for c in (self._features or "").split(",") if c.strip()]
            if not feat_cols:
                raise ValueError(
                    "No initial_parameters provided and 'features' is empty. "
                    "Provide Strategy(..., features='col1,col2,...') or pass initial_parameters."
                )
            input_dim = len(feat_cols)
            _model = Net(input_dim)
            _arrays = [t.detach().cpu().numpy() for t in _model.state_dict().values()]
            params["initial_parameters"] = fl.common.ndarrays_to_parameters(_arrays)

        # 4) Instantiate the real Flower strategy
        strat = StrategyClass(**params)

        # 5) Wrap aggregate_fit for logging (prints unchanged)
        original_agg_fit = strat.aggregate_fit

        def logged_agg_fit(server_round, results, failures):
            print(f"\n[Server] 🔄 Round {server_round} - Client Training Metrics:")
            for i, (client_id, fit_res) in enumerate(results):
                print(f" CTM Round {server_round} Client:{client_id.cid}: {fit_res.metrics}")
            agg_params, metrics = original_agg_fit(server_round, results, failures)
            print(f"[Server] ✅ Round {server_round} - Aggregated Training Metrics: {metrics}\n")
            # save the model parameters if savingPath is set on each saveOnRounds 
            if self.savingPath and (
                (server_round % self.saveOnRounds == 0)
                or (self.total_rounds and server_round == self.total_rounds)
            ):
                arrays = fl.common.parameters_to_ndarrays(agg_params)
                # Determine filename: final_model on last round else round_{n}
                filename = (
                    f"round_{server_round}_final_model.npz"
                    if server_round == self.total_rounds
                    else f"round_{server_round}_model.npz"
                )
                filepath = os.path.join(self.savingPath, filename)
                np.savez(filepath, *arrays)
            return agg_params, metrics

        strat.aggregate_fit = logged_agg_fit

        # 6) Wrap aggregate_evaluate for logging (prints unchanged)
        original_agg_eval = strat.aggregate_evaluate

        def logged_agg_eval(server_round, results, failures):
            print(f"\n[Server] 📊 Round {server_round} - Client Evaluation Metrics:")
            for i, (client_id, eval_res) in enumerate(results):
                print(f" CEM Round {server_round} Client:{client_id.cid}: {eval_res.metrics}")
            loss, metrics = original_agg_eval(server_round, results, failures)
            print(f"[Server] ✅ Round {server_round} - Aggregated Evaluation Metrics:")
            print(f"    Loss: {loss}, Metrics: {metrics}\n")
            return loss, metrics

        strat.aggregate_evaluate = logged_agg_eval

        # 7) Wrap configure_fit to:
        #    - log client properties (unchanged)
        #    - apply split_mode/client_fractions to fit_ins.config (NEW)
        original_conf_fit = strat.configure_fit

        def wrapped_conf_fit(
            server_round,
            parameters,
            client_manager
        ):
            selected = original_conf_fit(
                server_round=server_round,
                parameters=parameters,
                client_manager=client_manager
            )

            ins = GetPropertiesIns(config={})

            for client, fit_ins in selected:
                hostname = None
                try:
                    props = client.get_properties(ins=ins, timeout=10.0, group_id=0)
                    print(f"\n📋 [Round {server_round}] Client {client.cid} Properties: {props.properties}")
                    hostname = props.properties.get("hostname", None)
                except Exception as e:
                    print(f"⚠️ Failed to get properties from {client.cid}: {e}")

                # Fallback: if no hostname returned, use Flower cid
                if not hostname:
                    hostname = client.cid

                # Keep same object
                cfg = fit_ins.config

                if self.split_mode == "per_client":
                    # Lookup by hostname (preferred) or cid
                    per_cfg = (
                        self.client_fractions.get(hostname)
                        or self.client_fractions.get(client.cid)
                        or {}
                    )

                    # val_fraction: per-client override if present
                    if "val_fraction" in per_cfg:
                        try:
                            cfg["val_fraction"] = float(per_cfg["val_fraction"])
                        except Exception:
                            pass  # keep existing if invalid

                    # test: prefer test_ids if provided
                    if "test_ids" in per_cfg and per_cfg["test_ids"]:
                        test_ids_val = per_cfg["test_ids"]
                        if isinstance(test_ids_val, (list, tuple, set)):
                            test_ids_str = ",".join(str(x) for x in test_ids_val)
                        else:
                            test_ids_str = str(test_ids_val)
                        cfg["test_ids"] = test_ids_str
                        # when using explicit IDs, do not force a test_fraction for this client
                        if "test_fraction" in cfg:
                            del cfg["test_fraction"]
                        # ensure id_col is sent so client can map IDs
                        cfg["id_col"] = self.id_col
                    else:
                        # no test_ids -> use per-client test_fraction if present
                        if "test_fraction" in per_cfg:
                            try:
                                cfg["test_fraction"] = float(per_cfg["test_fraction"])
                            except Exception:
                                pass  # keep existing if invalid
                        # if no test_ids: id_col not strictly required, leave as-is
                else:
                    # split_mode == "global": enforce global fractions, clear any test_ids
                    if "test_ids" in cfg:
                        del cfg["test_ids"]
                    cfg["val_fraction"] = float(self._val_fraction)
                    cfg["test_fraction"] = float(self._test_fraction)
                    # also send id_col so clients know column name if needed
                    cfg["id_col"] = self.id_col

            return selected

        strat.configure_fit = wrapped_conf_fit

        # 8) Save the ready-to-use strategy
        self.strategy_object = strat
