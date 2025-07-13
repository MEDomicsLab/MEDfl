import flwr as fl
from typing import Callable, Optional, Dict, Any, Tuple, List

# ===================================================
# Custom metric aggregation functions for Flower
# ===================================================


def aggregate_fit_metrics(
    results: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
    """
    Perform weighted aggregation of training metrics across clients.

    Args:
        results (List of (num_examples, metrics)): 
            List of tuples where each tuple contains the number of examples 
            and a metrics dictionary for a client.

    Returns:
        Dict[str, float]: Weighted average metrics: 
            {'train_loss', 'train_accuracy', 'train_auc'}.
    """
    total_examples = sum(num_examples for num_examples, _ in results)

    # Weighted mean of each metric
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
    Perform weighted aggregation of evaluation metrics across clients.

    Args:
        results (List of (num_examples, metrics)): 
            List of tuples where each tuple contains the number of examples 
            and a metrics dictionary for a client.

    Returns:
        Dict[str, float]: Weighted average metrics: 
            {'eval_loss', 'eval_accuracy', 'eval_auc'}.
    """
    total_examples = sum(num_examples for num_examples, _ in results)

    # Weighted mean of each metric
    loss = sum(metrics.get("eval_loss", 0.0) * num_examples
               for num_examples, metrics in results) / total_examples
    accuracy = sum(metrics.get("eval_accuracy", 0.0) * num_examples
                   for num_examples, metrics in results) / total_examples
    auc = sum(metrics.get("eval_auc", 0.0) * num_examples
              for num_examples, metrics in results) / total_examples

    return {"eval_loss": loss, "eval_accuracy": accuracy, "eval_auc": auc}


# ===================================================
# Custom Strategy Wrapper with Logging
# ===================================================

class Strategy:
    """
    A wrapper for Flower server strategies providing:
        - Custom training and evaluation metric aggregation.
        - Console logging of per-client and aggregated metrics.
        - Flexible configuration of strategy parameters.

    Attributes:
        name (str): Name of the Flower strategy (default: 'FedAvg').
        fraction_fit (float): Fraction of clients participating in training.
        fraction_evaluate (float): Fraction of clients participating in evaluation.
        min_fit_clients (int): Minimum number of training clients.
        min_evaluate_clients (int): Minimum number of evaluation clients.
        min_available_clients (int): Minimum number of total available clients.
        initial_parameters (list): Optional initial model parameters.
        evaluate_fn (callable): Optional custom evaluation function.
        fit_metrics_aggregation_fn (callable): Aggregation function for training metrics.
        evaluate_metrics_aggregation_fn (callable): Aggregation function for evaluation metrics.
        strategy_object (fl.server.strategy.Strategy): Instantiated Flower strategy.

    Methods:
        create_strategy():
            Build and customize the Flower strategy object.
            Adds logging to the aggregation steps.
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
        """
        Initialize the strategy wrapper.

        Args:
            name (str): Strategy name (e.g., 'FedAvg').
            fraction_fit (float): Fraction of clients used in fit.
            fraction_evaluate (float): Fraction of clients used in evaluate.
            min_fit_clients (int): Minimum training clients.
            min_evaluate_clients (int): Minimum evaluation clients.
            min_available_clients (int): Minimum total available clients.
            initial_parameters (list): Optional initial model parameters.
            evaluate_fn (callable): Optional evaluation function.
            fit_metrics_aggregation_fn (callable): Optional fit aggregation function.
            evaluate_metrics_aggregation_fn (callable): Optional eval aggregation function.
        """
        self.name = name
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters or []
        self.evaluate_fn = evaluate_fn

        # Use custom or default aggregators
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn or aggregate_fit_metrics
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn or aggregate_eval_metrics

        self.strategy_object: Optional[fl.server.strategy.Strategy] = None

    def create_strategy(self) -> None:
        """
        Instantiate and customize the underlying Flower strategy.

        - Wraps aggregate_fit and aggregate_evaluate to print client and aggregated metrics.
        """
        # Resolve strategy class from Flower
        StrategyClass = getattr(fl.server.strategy, self.name)

        # Prepare parameters for strategy
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

        # Optionally add initial parameters
        if self.initial_parameters:
            params["initial_parameters"] = fl.common.ndarrays_to_parameters(
                self.initial_parameters
            )

        # Instantiate strategy
        strat = StrategyClass(**params)

        # ------------------------------------------------------------
        # Wrap aggregate_fit to log client and aggregated training metrics
        # ------------------------------------------------------------
        original_agg_fit = strat.aggregate_fit

        def logged_aggregate_fit(rnd, results, failures):
            print(f"\n[Server] 🔄 Round {rnd} - Client Training Metrics:")
            for i, (client_id, fit_res) in enumerate(results):
                print(f" CTM Round {rnd} Client:{client_id.cid}: {fit_res.metrics}")

            aggregated_params, metrics = original_agg_fit(rnd, results, failures)

            print(f"[Server] ✅ Round {rnd} - Aggregated Training Metrics: {metrics}\n")
            return aggregated_params, metrics

        strat.aggregate_fit = logged_aggregate_fit  # type: ignore

        # ------------------------------------------------------------
        # Wrap aggregate_evaluate to log client and aggregated eval metrics
        # ------------------------------------------------------------
        original_agg_eval = strat.aggregate_evaluate

        def logged_aggregate_evaluate(rnd, results, failures):
            print(f"\n[Server] 📊 Round {rnd} - Client Evaluation Metrics:")
            for i, (client_id, eval_res) in enumerate(results):
                print(f" CEM Round {rnd} Client:{client_id.cid}: {eval_res.metrics}")

            loss, metrics = original_agg_eval(rnd, results, failures)

            print(f"[Server] ✅ Round {rnd} - Aggregated Evaluation Metrics:")
            print(f"  Loss: {loss}, Metrics: {metrics}\n")
            return loss, metrics

        strat.aggregate_evaluate = logged_aggregate_evaluate  # type: ignore

        # Save strategy object
        self.strategy_object = strat
