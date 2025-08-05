import flwr as fl
from typing import Callable, Optional, Dict, Any, Tuple, List
from flwr.common import GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import time

# ===================================================
# Custom metric aggregation functions
# ===================================================

def aggregate_fit_metrics(
    results: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
    total = sum(n for n, _ in results)
    loss = sum(m.get("train_loss", 0.0) * n for n, m in results) / total
    acc = sum(m.get("train_accuracy", 0.0) * n for n, m in results) / total
    auc = sum(m.get("train_auc", 0.0) * n for n, m in results) / total
    return {"train_loss": loss, "train_accuracy": acc, "train_auc": auc}

def aggregate_eval_metrics(
    results: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
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
    - Custom metric aggregation
    - Per-client & aggregated metric logging
    - Synchronous get_properties() inspection in configure_fit()
    """

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

        self.strategy_object: Optional[fl.server.strategy.Strategy] = None
        self.local_epochs = local_epochs

    def create_strategy(self) -> None:
        StrategyClass = getattr(fl.server.strategy, self.name)
        params: Dict[str, Any] = {
            "fraction_fit": self.fraction_fit,
            "fraction_evaluate": self.fraction_evaluate,
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "min_available_clients": self.min_available_clients,
            "evaluate_fn": self.evaluate_fn,
            "fit_metrics_aggregation_fn": self.fit_metrics_aggregation_fn,
            "evaluate_metrics_aggregation_fn": self.evaluate_metrics_aggregation_fn,
        }
        if self.initial_parameters:
            params["initial_parameters"] = fl.common.ndarrays_to_parameters(self.initial_parameters)

        strat = StrategyClass(**params)

        # Wrap aggregate_fit to log metrics
        original_agg_fit = strat.aggregate_fit
        def logged_agg_fit(server_round, results, failures):
            print(f"\n[Server] 🔄 Round {server_round} - Client Training Metrics:")
            for i, (client_id, fit_res) in enumerate(results):
                print(f" CTM Round {server_round} Client:{client_id.cid}: {fit_res.metrics}")
            agg_params, metrics = original_agg_fit(server_round, results, failures)
            print(f"[Server] ✅ Round {server_round} - Aggregated Training Metrics: {metrics}\n")
            return agg_params, metrics
        strat.aggregate_fit = logged_agg_fit

        # original_handle_fit = strat.handle_fit
        # def realtime_handle_fit(server_round, results, failures):
        #     with self.lock:
        #         for client_id, fit_res in results:
        #             if client_id.cid not in self.current_round_results[server_round]:
        #                 # New completion
        #                 completion_time = time.time()
        #                 self.current_round_results[server_round][client_id.cid] = completion_time
                        
        #                 # Get current leader
        #                 times = self.current_round_results[server_round]
        #                 if times:
        #                     leader = min(times, key=times.get)
        #                     lead_time = times[leader]
        #                     delay = completion_time - lead_time
                            
        #                     print(f"[Server] 🚦 Round {server_round} - "
        #                           f"Client {client_id.cid} finished: "
        #                           f"{'🏁 FIRST' if delay == 0 else f'+{delay:.2f}s'}")
            
        #     return original_handle_fit(server_round, results, failures)
        
        # strat.handle_fit = realtime_handle_fit

        # Wrap aggregate_evaluate to log metrics
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

        # Wrap configure_fit to inspect client properties synchronously
        original_conf_fit = strat.configure_fit
        def wrapped_conf_fit(
            server_round: int,
            parameters,
            client_manager: ClientManager
        ):
            selected = original_conf_fit(
                server_round=server_round,
                parameters=parameters,
                client_manager=client_manager
            )

            # Synchronously fetch & log properties
            ins = GetPropertiesIns(config={})
            for client, _ in selected:
                try:
                    props = client.get_properties(ins=ins, timeout=10.0, group_id=0)
                    print(f"\n📋 [Round {server_round}] Client {client.cid} Properties: {props.properties}")
                    
                except Exception as e:
                    print(f"⚠️ Failed to get properties from {client.cid}: {e}")

            return selected
        
        def fit_config_fn(server_round: int) -> Dict[str, Any]:
            return {"local_epochs": self.local_epochs}

        params["on_fit_config_fn"] = fit_config_fn

        strat.configure_fit = wrapped_conf_fit
        self.strategy_object = strat
