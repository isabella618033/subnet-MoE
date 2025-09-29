from prometheus_client import Counter, Gauge, Histogram

miner_requests = Counter("miner_requests_total", "Requests received")
miner_submissions = Counter("miner_submissions_total", "Submissions made")
validator_scores = Histogram("validator_scores", "Miner scores")

def start_metrics_server(port: int = 8001) -> None:
    from prometheus_client import start_http_server
    start_http_server(port)
