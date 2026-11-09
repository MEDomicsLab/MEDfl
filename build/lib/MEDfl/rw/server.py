import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.server import ServerConfig
from MEDfl.rw.strategy import Strategy
from MEDfl.LearningManager.shap import SHAPConfig
from MEDfl.rw.holdout import HoldoutConfig
import asyncio
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import GetPropertiesIns
from flwr.common import GetPropertiesIns


class FederatedServer:
    """
    FederatedServer wraps the launch and configuration of a Flower federated learning server.

    Attributes:
        server_address (str): Server host and port in the format "host:port".
        server_config (ServerConfig): Configuration for the Flower server.
        strategy_wrapper (Strategy): Wrapper around the actual Flower strategy.
        strategy (flwr.server.Strategy): Actual Flower strategy instance.
        certificates (Any): Optional TLS certificates.
        connected_clients (list): List of connected client IDs.

    Methods:
        start():
            Launch the Flower server with the specified strategy and log client connections.
    """

    def __init__(
        self,
        host="0.0.0.0",
        port=8080,
        num_rounds=3,
        strategy=None,
        certificates=None,
        shap_config=None,
        holdout_config=None,
    ):
        """
        Initialize the FederatedServer.

        Args:
            host (str): Hostname or IP to bind the server to.
            port (int): Port to listen on.
            num_rounds (int): Number of federated learning rounds to execute.
            strategy (Optional[Strategy]): Optional custom strategy wrapper.
            certificates (Optional[Any]): Optional TLS certificates.
            shap_config (Optional[SHAPConfig]): Optional final-round federated SHAP.
            holdout_config (Optional[HoldoutConfig]): Optional final-round federated
                holdout test. When None, the strategy's own holdout_config is kept.
        """
        # Server address and configuration
        self.server_address = f"{host}:{port}"
        self.server_config = ServerConfig(num_rounds=num_rounds)

        # Use custom or default strategy
        self.strategy_wrapper = strategy or Strategy()

        self.shap_config = (
            shap_config
            if shap_config is not None
            else SHAPConfig(enabled=False)
        )

        # The Strategy owns the final-round SHAP orchestration.
        self.strategy_wrapper.shap_config = self.shap_config

        # The Strategy also owns the final-round holdout test.
        if holdout_config is not None:
            holdout_config.validate()
            self.strategy_wrapper.holdout_config = holdout_config
        self.holdout_config = getattr(
            self.strategy_wrapper,
            "holdout_config",
            None,
        ) or HoldoutConfig(enabled=False)

        # Keep the strategy's final-round marker synchronized with
        # the actual Flower ServerConfig.
        self.strategy_wrapper.total_rounds = int(num_rounds)

        # Validate SHAP against the active real-world backend.
        if self.shap_config.enabled:
            feature_columns = [
                column.strip()
                for column in (
                    getattr(
                        self.strategy_wrapper,
                        "_features",
                        "",
                    )
                    or ""
                ).split(",")
                if column.strip()
            ]

            if not feature_columns:
                raise ValueError(
                    "Federated SHAP requires the Strategy 'features' "
                    "configuration so the model input size can be "
                    "validated."
                )

            backend = (
                "xgboost"
                if getattr(
                    self.strategy_wrapper,
                    "name",
                    "",
                ) == "FedXgbBagging"
                else "pytorch"
            )

            task = (
                getattr(
                    self.strategy_wrapper,
                    "task",
                    None,
                )
                if backend == "xgboost"
                else None
            )

            self.shap_config.validate_backend(
                input_size=len(feature_columns),
                backend=backend,
                task=task,
            )

        self.strategy_wrapper.create_strategy()
        if self.strategy_wrapper.strategy_object is None:
            raise ValueError(
                "Strategy object not initialized. "
                "Call create_strategy() first."
            )
        self.strategy = self.strategy_wrapper.strategy_object

        self.certificates = certificates
        self.connected_clients = []  # Track connected client IDs

        self.federated_shap_result = {
            "status": (
                "pending"
                if self.shap_config.enabled
                else "disabled"
            )
        }
        self.local_shap_results = []

        self.federated_holdout_result = {
            "status": (
                "pending"
                if self.holdout_config.enabled
                else "disabled"
            )
        }


    def start(self):
        """
        Start the Flower server with the configured strategy and track client connections.
        """
        print(f"Using strategy: {self.strategy_wrapper.name}")
        print(f"Starting Flower server on {self.server_address} with strategy {self.strategy_wrapper.name}")

        # Use a custom client manager that logs client connections
        client_manager = TrackingClientManager(self)

        # Launch the Flower server
        fl.server.start_server(
            server_address=self.server_address,
            config=self.server_config,
            strategy=self.strategy,
            certificates=self.certificates,
            client_manager=client_manager,
        )

        # The strategy aggregates SHAP during the final evaluation.
        self.federated_shap_result = (
            self.strategy_wrapper.federated_shap_result
        )
        self.local_shap_results = (
            self.strategy_wrapper.local_shap_results
        )

        if self.shap_config.enabled:
            print(
                "[Server/SHAP] Final federated SHAP status: "
                f"{self.federated_shap_result.get('status')}",
                flush=True,
            )

        # The strategy aggregates the holdout test during the final evaluation.
        self.federated_holdout_result = getattr(
            self.strategy_wrapper,
            "federated_holdout_result",
            self.federated_holdout_result,
        )

        if self.holdout_config.enabled:
            print(
                "[Server/Holdout] Final federated holdout status: "
                f"{self.federated_holdout_result.get('status')}",
                flush=True,
            )


class TrackingClientManager(fl.server.client_manager.SimpleClientManager):
    """
    TrackingClientManager extends the default SimpleClientManager to log client connections.

    Attributes:
        server (FederatedServer): The FederatedServer instance this manager belongs to.
        client_properties (dict): Placeholder for storing client-specific properties.
    """

    def __init__(self, server):
        """
        Initialize the TrackingClientManager.

        Args:
            server (FederatedServer): Reference to the FederatedServer.
        """
        super().__init__()
        self.server = server
        self.client_properties = {}

    def register(self, client):
        """
        Register a client and log its connection.

        Args:
            client (ClientProxy): The client proxy being registered.

        Returns:
            bool: True if the client was registered successfully.
        """
        success = super().register(client)
        if success and client.cid not in self.server.connected_clients:
            # Run the asynchronous hostname fetch synchronously
            asyncio.run(self._fetch_and_log_hostname(client))
        return success

    async def _fetch_and_log_hostname(self, client):
        """
        Asynchronously fetch and log the client's hostname or CID.

        Args:
            client (ClientProxy): The client proxy.
        """
        # Optional: uncomment to fetch hostname from client properties
        # try:
        #     ins = GetPropertiesIns(config={})
        #     props = await client.get_properties(ins=ins, timeout=10.0, group_id=0)
        #     hostname = props.properties.get("hostname", "unknown")
        # except Exception as e:
        #     hostname = f"Error: {e}"

        print(f"✅ Client connected - CID: {client.cid}")
        self.server.connected_clients.append(client.cid)