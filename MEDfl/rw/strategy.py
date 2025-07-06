import flwr as fl
from typing import Callable, Optional, Dict, Any, Tuple, List

# Custom aggregation for client-returned metrics

from typing import List, Tuple, Dict

# Custom aggregation for client-returned metrics
def aggregate_fit_metrics(
    results: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
    """
    Weighted aggregation of training metrics across clients.
    Expects each tuple (num_examples, metrics_dict) to include
    'train_loss', 'train_accuracy', 'train_auc'.
    """
    # Sum total examples
    total_examples = sum(num_examples for num_examples, _ in results)
    # Weighted averages
    loss = sum(metrics.get("train_loss", 0.0) * num_examples
               for num_examples, metrics in results) / total_examples
    accuracy = sum(metrics.get("train_accuracy", 0.0) * num_examples
                   for num_examples, metrics in results) / total_examples
    auc = sum(metrics.get("train_auc", 0.0) * num_examples
              for num_examples, metrics in results) / total_examples
    return {"train_loss": loss, "train_accuracy": accuracy, "train_auc": auc}


def aggregate_eval_metrics(
    results: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
    """
    Weighted aggregation of evaluation metrics across clients.
    Expects each tuple (num_examples, metrics_dict) to include
    'eval_loss', 'eval_accuracy', 'eval_auc'.
    """
    total_examples = sum(num_examples for num_examples, _ in results)
    loss = sum(metrics.get("eval_loss", 0.0) * num_examples
               for num_examples, metrics in results) / total_examples
    accuracy = sum(metrics.get("eval_accuracy", 0.0) * num_examples
                   for num_examples, metrics in results) / total_examples
    auc = sum(metrics.get("eval_auc", 0.0) * num_examples
              for num_examples, metrics in results) / total_examples
    return {"eval_loss": loss, "eval_accuracy": accuracy, "eval_auc": auc}

class Strategy:
    """
    A wrapper for Flower server strategies, with custom metric aggregation
    and console logs on aggregation/evaluation completion.
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
        evaluate_fn: Optional[Callable[
            [int, fl.common.Parameters, Dict[str, Any]],
            Tuple[float, Dict[str, float]]
        ]] = None,
        fit_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, fl.common.FitRes]]], Dict[str, float]]
        ] = None,
        evaluate_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, fl.common.EvaluateRes]]], Dict[str, float]]
        ] = None,
    ) -> None:
        self.name = name
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters or []
        self.evaluate_fn = evaluate_fn
        # Use custom aggregators if provided, else default to ours
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn or aggregate_fit_metrics
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn or aggregate_eval_metrics
        self.strategy_object: Optional[fl.server.strategy.Strategy] = None

    def create_strategy(self) -> None:
        # 1) Instantiate the underlying Flower strategy
        StrategyClass = getattr(fl.server.strategy, self.name)
        params: Dict[str, Any] = {
            "fraction_fit": self.fraction_fit,
            "fraction_evaluate": self.fraction_evaluate,
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "min_available_clients": self.min_available_clients,
            "evaluate_fn": self.evaluate_fn,
            # Plug in our custom aggregators
            "fit_metrics_aggregation_fn": self.fit_metrics_aggregation_fn,
            "evaluate_metrics_aggregation_fn": self.evaluate_metrics_aggregation_fn,
        }
        if self.initial_parameters:
            params["initial_parameters"] = fl.common.ndarrays_to_parameters(
                self.initial_parameters
            )

        strat = StrategyClass(**params)

        # 2) Wrap aggregate_fit to log
        original_agg_fit = strat.aggregate_fit
        def logged_aggregate_fit(rnd, results, failures):
            # Print individual client fit metrics
            print(f"\n[Server] 🔄 Round {rnd} - Client Training Metrics:")
            for i, (client_id, fit_res) in enumerate(results):
                print(f" CTM Round {rnd} Client:{client_id.cid}: {fit_res.metrics}")

            # Call original aggregation function
            aggregated_params, metrics = original_agg_fit(rnd, results, failures)

            # Print aggregated metrics
            print(f"[Server] ✅ Round {rnd} - Aggregated Training Metrics: {metrics}\n")
            return aggregated_params, metrics

        strat.aggregate_fit = logged_aggregate_fit  # type: ignore

        # 3) Wrap aggregate_evaluate to log
        original_agg_eval = strat.aggregate_evaluate
        def logged_aggregate_evaluate(rnd, results, failures):
            # Print individual client evaluation metrics
            print(f"\n[Server] 📊 Round {rnd} - Client Evaluation Metrics:")
            for i, (client_id, eval_res) in enumerate(results):
                print(f" CEM Round {rnd} Client:{client_id.cid}: {eval_res.metrics}")

            # Call original aggregation function
            loss, metrics = original_agg_eval(rnd, results, failures)

            # Print aggregated metrics
            print(f"[Server] ✅ Round {rnd} - Aggregated Evaluation Metrics:")
            print(f"  Loss: {loss}, Metrics: {metrics}\n")
            return loss, metrics

        strat.aggregate_evaluate = logged_aggregate_evaluate  # type: ignore

        self.strategy_object = strat
