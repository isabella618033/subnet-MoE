from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib

import bittensor as bt
import torch
from substrateinterface import Keypair

@dataclass
class SignedMessage:
    target_hotkey_ss58: str
    origin_hotkey_ss58: str
    block: int
    signature: str   # hex string or raw bytes

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

@dataclass
class SignedDownloadRequestMessage(SignedMessage):
    expert_group_id: int | None

@dataclass
class SignedModelSubmitMessage(SignedMessage):
    pass


def serialize_torch_model_path(model_path: str) -> bytes:
    """
    Load a torch model from disk and serialize its state_dict
    deterministically into raw bytes.
    """
    state = torch.load(model_path, map_location="cpu")

    # If it's a full model, extract state_dict
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    elif not isinstance(state, dict):
        raise ValueError("Model file must contain a state_dict or nn.Module")

    buffer = []
    for key, tensor in state.items():
        buffer.append(key.encode())
        buffer.append(tensor.cpu().numpy().tobytes())

    return b"".join(buffer)


def hash_model_bytes(model_bytes: bytes) -> bytes:
    """
    Blake2b-256 hash (32 bytes) of the model.
    """
    return hashlib.blake2b(model_bytes, digest_size=32).digest()
    

def construct_model_message(
    model_path: str | Path,
    target_hotkey_ss58: str,
    block: int
):
    """
    Sign:
        model_hash(32 bytes) || construct_block_message(...)
    """
    # 1. Serialize model → bytes
    model_bytes = serialize_torch_model_path(model_path)

    # 2. Hash model to 32 bytes
    model_hash = hash_model_bytes(model_bytes)

    # 3. Create pubkey || block message
    block_msg = construct_block_message(target_hotkey_ss58, block)

    # 4. Final message to sign
    full_message = model_hash + block_msg

    return full_message


def construct_block_message(target_hotkey_ss58: str, block: int) -> bytes:
    """
    Construct message: pubkey(32 bytes) || block(u64 big-endian)
    """
    # Convert SS58 → raw 32-byte pubkey
    target_kp = bt.Keypair(ss58_address=target_hotkey_ss58)
    pubkey_bytes = target_kp.public_key

    if len(pubkey_bytes) != 32:
        raise ValueError("Public key must be 32 bytes!")

    # Convert block → 8 bytes big-endian
    block_bytes = block.to_bytes(8, "big")

    # Final message
    return pubkey_bytes + block_bytes


def sign_message(origin_hotkey: Keypair, message: bytes):
    """
    Sign the constructed message using the hotkey.
    """
    return origin_hotkey.sign(message).hex()



def verify_message(origin_hotkey_ss58: str,
                 message: bytes,
                 signature_hex: str) -> bool:
    """
    Verify the signature for the message: pubkey || block
    signed by the hotkey at `my_hotkey_ss58_address`.
    """
    # 1. Rebuild signer keypair from their SS58
    signer_kp = bt.Keypair(ss58_address=origin_hotkey_ss58)

    # 2. Decode signature
    signature = bytes.fromhex(signature_hex)

    # 3. Verify
    return signer_kp.verify(message, signature)
