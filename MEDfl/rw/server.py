import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.server import ServerConfig
from typing import Optional, Any, List, Tuple, Dict, Callable
from MEDfl.rw.strategy import Strategy
from MEDfl.rw.verbose_server import VerboseServer
import time
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import GetPropertiesIns
import asyncio

class FederatedServer:
    """
    Wrapper for launching a Flower federated-learning server,
    using a Strategy instance as its strategy attribute.
    Now with client connection tracking.
    """
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        num_rounds: int = 3,
        strategy: Optional[Strategy] = None,
        certificates: Optional[Any] = None,
    ):
        self.server_address = f"{host}:{port}"
        self.server_config = ServerConfig(num_rounds=num_rounds)
        # If no custom strategy provided, use default
        self.strategy_wrapper = strategy or Strategy()
        # Build the actual Flower strategy object
        self.strategy_wrapper.create_strategy()
        if self.strategy_wrapper.strategy_object is None:
            raise ValueError("Strategy object not initialized. Call create_strategy() first.")
        self.strategy = self.strategy_wrapper.strategy_object
        self.certificates = certificates
        self.connected_clients = []

    def start(self) -> None:
        """
        Start the Flower server with the configured strategy.
        Now tracks and logs client connections before starting.
        """
        print(f"Using strategy: {self.strategy_wrapper.name}")
        print(f"Starting Flower server on {self.server_address} with strategy {self.strategy_wrapper.name}")
        
        # Create a custom client manager to track connections
        client_manager = TrackingClientManager(self)
        
        fl.server.start_server(
            server_address=self.server_address,
            config=self.server_config,
            strategy=self.strategy,
            certificates=self.certificates,
            client_manager=client_manager,
        )

class TrackingClientManager(fl.server.client_manager.SimpleClientManager):
    """
    Custom client manager that tracks and logs client connections.
    """
    def __init__(self, server: FederatedServer):
        super().__init__()
        self.server = server
        self.client_properties = {}  # Store client properties

    def register(self, client: ClientProxy) -> bool:
        success = super().register(client)
        if success and client.cid not in self.server.connected_clients:
            # Run the async fetch synchronously
            asyncio.run(self._fetch_and_log_hostname(client))
        return success

    async def _fetch_and_log_hostname(self, client: ClientProxy):
        # try:
        #     ins = GetPropertiesIns(config={})
        #     props = await client.get_properties(ins=ins, timeout=10.0, group_id=0)
        #     hostname = props.properties.get("hostname", "unknown")
        # except Exception as e:
        #     hostname = f"Error: {e}"
        print(f"✅ Client connected - CID: {client.cid}")
        self.server.connected_clients.append(client.cid)
