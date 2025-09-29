from ..shared import messaging, scoring, blockchain
from .evaluator import Evaluator
from .aggregator import Aggregator

class RoundManager:
    def __init__(self):
        self.evaluator = Evaluator()
        self.aggregator = Aggregator()

    def open_round(self, spec):  # define dataset/steps/turn order/etc.
        messaging.publish_request(spec)

    def collect_and_score(self, round_id: str, timeout_s: int = 600):
        # Consume submissions for round; score them
        rows = []
        for sub in self.evaluator.consume_for_round(round_id, timeout_s=timeout_s):
            rows.append(self.evaluator.score_submission(sub))
        return rows

    def finalize(self, round_id: str, rows: list[dict]):
        agg = self.aggregator.aggregate(rows)
        tx = blockchain.submit_score(round_id, agg)
        return {"aggregate": agg, "tx": tx}

    def run_forever(self):
        # basic loop: open → collect → score → aggregate → submit
        while True:
            spec = self.plan_next_spec()
            self.open_round(spec)
            rows = self.collect_and_score(spec["round_id"])
            result = self.finalize(spec["round_id"], rows)
            self.log_result(result)

    def plan_next_spec(self):
        # Turn-taking logic (request miners in turn), dataset params, etc.
        ...
    def log_result(self, result): ...
