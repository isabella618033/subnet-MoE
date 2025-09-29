# llm_weightnet/shared/blockchain.py
from __future__ import annotations
from typing import Optional, Dict

# Implement using your chain’s SDK/web3 client.
# For now, return a stub pulled from env/config for demo purposes.

def get_active_validator_info(round_id: Optional[str] = None) -> Optional[Dict]:
    """
    Returns {"api_base": "http://validator:8000"} or None if no validator registered.
    In production: read contract state to discover the currently active validator node.
    """
    import os
    api_base = os.getenv("VALIDATOR_API_BASE")  # e.g., http://localhost:8000
    if not api_base:
        return None
    return {"api_base": api_base, "round_id": round_id}

def submit_score(round_id: str, scores: dict) -> str:  # returns tx hash
    ...
def get_validator_prompt(round_id: str) -> dict:
    # fetch on-chain params for a round if applicable
    ...
