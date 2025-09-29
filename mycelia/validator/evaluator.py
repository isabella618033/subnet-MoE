from ..shared import messaging, scoring, metrics

class Evaluator:
    def consume_for_round(self, round_id, timeout_s=600):
        for sub in messaging.consume_submissions(group=f"validator-{round_id}"):
            if sub["round_id"] == round_id:
                yield sub

    def score_submission(self, sub):
        scores = scoring.evaluate_weights(sub["artifact_uri"], dataset_id="eval")
        # update Prometheus
        # metrics.validator_scores.observe(scores["primary_metric"])
        return {"miner_id": sub["miner_id"], "scores": scores}
