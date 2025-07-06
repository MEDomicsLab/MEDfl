from flwr.server import Server
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import INFO

class VerboseServer(Server):
    def __init__(self, strategy):
        super().__init__(client_manager=SimpleClientManager(), strategy=strategy)

    def client_manager_fn(self):
        return self.client_manager

    def on_client_connect(self, client: ClientProxy):
        super().on_client_connect(client)
        log(INFO, f"[Server] ➕ Client connected: {client.cid}")
        log(INFO, f"[Server] Currently connected: {len(self.client_manager.all().values())} clients")

    def on_client_disconnect(self, client: ClientProxy):
        super().on_client_disconnect(client)
        log(INFO, f"[Server] ❌ Client disconnected: {client.cid}")
