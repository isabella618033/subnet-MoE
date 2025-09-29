from .round_manager import RoundManager
from ..shared.metrics import start_metrics_server

def run():
    start_metrics_server(port=8003)  # validator metrics
    RoundManager().run_forever()
