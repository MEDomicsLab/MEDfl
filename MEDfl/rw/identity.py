"""
Name a real-world client reports to the server (properties, metrics, SHAP and holdout results).

The MEDfl agent ID (MEDFL_AGENT_ID, inherited from the agent that starts the client) when it is set,
so several agents on the same machine are told apart. Else the machine hostname, as before.
"""
import os
import socket


def client_name() -> str:
    return (os.getenv("MEDFL_AGENT_ID") or "").strip() or socket.gethostname()
