from typing import TypedDict, Literal

Role = Literal["miner","validator"]

class WeightRequest(TypedDict):
    round_id: str
    dataset_id: str
    subset: str | None
    steps: int
    lr: float
    partial_layers: list[int]
    deadline_ts: float
    miner_id: str | None  # if targeted; otherwise broadcast

class WeightSubmission(TypedDict):
    round_id: str
    miner_id: str
    artifact_uri: str      # where weights live (S3/ipfs)
    signature_hex: str
    train_metrics: dict

def publish_request(req: WeightRequest) -> None: ...
def consume_requests(group: str):  # yields WeightRequest
    yield from ()
def publish_submission(sub: WeightSubmission) -> None: ...
def consume_submissions(group: str):  # yields WeightSubmission
    yield from ()
