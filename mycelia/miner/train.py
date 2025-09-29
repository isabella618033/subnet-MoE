from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Dict, Any
from mycelia.shared import messaging, training, weights, storage, model, datasets
from mycelia.settings import Settings

@dataclass
class TrainSpec:
    dataset_id: str
    subset: str | None
    steps: int
    lr: float
    seed: int
    partial_layers: list[int]  # e.g., last N transformer blocks

@dataclass
class TrainResult:
    weights_path: Path  # artifact (e.g., safetensors/pt)
    metrics: Dict[str, float]  # training/eval metrics

class Trainer(Protocol):
    def train_partial(self, spec: TrainSpec) -> TrainResult: ...

class MinerScheduler:
    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or Settings()

    def start(self):
        for req in messaging.consume_requests(group=self.cfg.miner_id):
            self.handle_request(req)

    def handle_request(self, req):
        # 1) Prepare data/model
        ds = datasets.load(req["dataset_id"], subset=req.get("subset"))
        mdl = model.load_base(self.cfg.model_name, device=self.cfg.device)

        # 2) Train partial
        spec = training.TrainSpec(
            dataset_id=req["dataset_id"],
            subset=req.get("subset"),
            steps=req["steps"],
            lr=req["lr"],
            seed=self.cfg.seed,
            partial_layers=req["partial_layers"],
        )
        result = self.cfg.trainer.train_partial(spec)

        # 3) Package & sign
        blob = open(result.weights_path, "rb").read()
        sig = weights.sign(blob, self.cfg.private_key)

        # 4) Store artifact and submit
        uri = storage.put_bytes(blob, path_hint=f"{req['round_id']}/{self.cfg.miner_id}.bin")
        messaging.publish_submission({
            "round_id": req["round_id"],
            "miner_id": self.cfg.miner_id,
            "artifact_uri": uri,
            "signature_hex": sig.hex(),
            "train_metrics": result.metrics,
        })