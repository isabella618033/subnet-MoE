import bittensor
import uvicorn
from fastapi import FastAPI, HTTPException

from mycelia.shared.config import OwnerConfig, parse_args
from mycelia.sn_owner.cycle import PhaseManager, PhaseResponse

app = FastAPI(title="Phase Service")


@app.get("/get_phase", response_model=PhaseResponse)
async def read_phase():
    """
    Returns which phase we're in for the given block height.
    """
    try:
        return phase_manager.get_phase()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/blocks_until_next_phase", response_model=dict[str, int])
async def next_phase():
    """
    Returns which phase we're in for the given block height.
    """
    try:
        return phase_manager.blocks_until_next_phase()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Phase service is running",
        "cycle_length": phase_manager.cycle_length,
        "phases": [{"index": i, "name": p["name"], "length": p["length"]} for i, p in enumerate(phase_manager.phases)],
        "usage": "GET /phase?block_height=123",
    }


if __name__ == "__main__":
    args = parse_args()

    global config
    global phase_manager

    if args.path:
        config = OwnerConfig.from_path(args.path)
    else:
        config = OwnerConfig()

    config.write()
    subtensor = bittensor.Subtensor(network=config.chain.network)
    phase_manager = PhaseManager(config, subtensor)
    uvicorn.run(app, host=config.owner.app_ip, port=config.owner.app_port)
