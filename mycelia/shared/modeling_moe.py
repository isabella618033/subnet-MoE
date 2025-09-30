"""
MoE model utilities: convert dense models to MoE-style, define custom sparse blocks,
and provide helpers for partial (grouped) expert execution.

Contents
--------
- DenseBlock: wraps Qwen3Moe MLP to behave like a "dense" block while preserving API
- myceliaSparseMoeBlock: sparse MoE block that activates only a subset of experts (by group)
- CustomMoE: an Qwen3Moe model variant that interleaves MoE and dense blocks
- get_base_model: load base LLaMA or OLMo, optionally convert to MoE
- get_base_tokenizer: load tokenizer (HF login via env var if provided)
- dense_model_to_moe: transform a dense model into Qwen3Moe parameter layout
- get_layer_expert_id: parse layer/expert indices from parameter names
- partial_moe: drop experts outside the current group and rebuild a partial model

Assumptions
-----------
- Parameter names use Qwen3Moe/transformers conventions (e.g., "model.layers.{i}.mlp.experts.{e}").
- Experts are identified by the "experts.{expert_id}" path segment.
- Your `Config` provides fields used by `get_base_model` and `get_base_tokenizer`.
"""

from __future__ import annotations

import os
import re
import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    OlmoForCausalLM,
    AutoModelForCausalLM,
    PretrainedConfig
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeMLP,  
    Qwen3MoeForCausalLM,
    load_balancing_loss_func,
)
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Attention

from mycelia.config import Config

logger = logging.getLogger(__name__)

class SparseMoeBlock(nn.Module):
    """
    Sparse MoE block that only uses a subset of experts assigned to the current group.

    Parameters
    ----------
    config : Qwen3MoeConfig-like
        Must provide: hidden_size, num_experts, num_experts_per_tok, norm_topk_prob.
    my_group_id : int
        The group this rank belongs to.
    expert_group_assignment : Dict[int, Dict[int, List[int]]]
        Mapping: layer_id -> (group_id -> list of expert ids available to that group).
    layer_id : int
        The layer index used to pick the allowed experts.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        num_experts: int | None = None,
        my_group_id: int | None = None,
        expert_group_assignment: Dict[int, Dict[int, List[int]]] | None = None,
    ):
        super().__init__()
        self.num_experts: int = config.num_experts
        self.top_k: int = config.num_experts_per_tok
        self.norm_topk_prob: bool = getattr(config, "norm_topk_prob", True)
        self.layer_id: int = layer_id
        self.expert_group_assignment = expert_group_assignment

        # Router gate that scores experts
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)

        # Only instantiate experts owned by this group at this layer
        if my_group_id is not None and expert_group_assignment is not None:
            allowed = expert_group_assignment[layer_id][my_group_id]
            self.experts = nn.ModuleDict({str(k): Qwen3MoeMLP(config) for k in allowed})
        elif num_experts is not None:
            allowed = list(range(num_experts))
            self.experts = nn.ModuleDict({str(k): Qwen3MoeMLP(config) for k in allowed})
        else:
            raise KeyError

        self.keep_ids = torch.as_tensor([int(k) for k in self.experts.keys()])

    def _mask_routing_weights(self, x: torch.Tensor, my_group_id: int | None = None, dim: int = 1) -> torch.Tensor:
        """
        Zero-out routing weights for experts not present in this group.
        """

        if my_group_id is None:
            keep_ids = self.keep_ids
        else:
            keep_ids = torch.as_tensor(self.expert_group_assignment[self.layer_id][my_group_id], device=x.device)

        mask_1d = torch.zeros(x.size(dim), dtype=torch.bool, device=x.device)
        mask_1d[keep_ids.to(x.device)] = True

        # Broadcast mask across all other dims
        shape = [1] * x.ndim
        shape[dim] = x.size(dim)
        mask = mask_1d.view(shape).to(dtype=x.dtype)
        return x * mask

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights = self._mask_routing_weights(routing_weights)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in self.experts.keys():
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[int(expert_idx)])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

class MyceliaMoE(Qwen3MoeForCausalLM):
    """
    Qwen3Moe variant that interleaves MoE and dense blocks and optionally restricts experts
    to those owned by the calling group.

    If `partial=True`, MoE blocks become `myceliaSparseMoeBlock` limited to the group’s experts.
    Otherwise, standard `SparseMoeBlock` is used for MoE layers.
    """

    def __init__(
        self,
        config: Config,
        model_config: Qwen3MoeConfig,
        my_group_id: Optional[int] = None,
        expert_group_assignment: Optional[Dict[int, Dict[int, List[int]]]] = None,
        partial: bool = False,
    ):
        super().__init__(model_config)
        layers: List[nn.Module] = []

        for i in range(model_config.num_hidden_layers):
            layer = Qwen3MoeDecoderLayer(model_config, layer_idx=i)

            # Interleave MoE and dense layers (MoE on even indices if interleave=True)
            if getattr(model_config, "interleave", True) and (i + 1) % model_config.decoder_sparse_step == 0:
                if not partial:
                    layer.mlp = SparseMoeBlock(
                        model_config, i, num_experts=config.num_experts, expert_group_assignment=expert_group_assignment
                    )  # full MoE (all experts)
                else:
                    assert (
                        my_group_id is not None and expert_group_assignment is not None
                    ), "partial=True requires my_group_id and expert_group_assignment"
                    layer.mlp = SparseMoeBlock(
                        model_config, i, my_group_id=my_group_id, expert_group_assignment=expert_group_assignment
                    )

            layers.append(layer)

        self.model.layers = nn.ModuleList(layers)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, MoeCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OlmoeForCausalLM

        >>> model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'Hey, are you conscious? Can you talk to me?\nI’m not sure if you’re conscious of this, but I’m'
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss_sum = 0
            aux_count = 0
            for r in outputs.router_logits if return_dict else outputs[-1]:
                aux_loss_sum += load_balancing_loss_func(
                    tuple([r]),
                    attention_mask=attention_mask,
                    num_experts=self.num_experts,
                    top_k=self.num_experts_per_tok,
                )
                aux_count += 1
            aux_loss = (aux_loss_sum / aux_count) - self.num_experts_per_tok

            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def get_base_model(
    config: Config, expert_group_assignment: Dict[int, Dict[int, List[int]]] | None = None, noise: bool = False, full=False
) -> Optional[LlamaForCausalLM | OlmoForCausalLM]:
    """
    Load a base Causal LM by `config.model_path` and optionally convert to MoE.

    Returns
    -------
    Optional[nn.Module]
        A Hugging Face causal LM (LLaMA or OLMo), possibly converted to OLMoE.
    """
    model = None

    if "llama" in config.model.model_path.lower():
        llama_cfg = LlamaConfig.from_pretrained(config.model.model_path, attn_implementation=config.model.attn_implementation)
        model = LlamaForCausalLM.from_pretrained(config.model.model_path, config=llama_cfg)

    elif "olmo" in config.model.model_path:
        model = OlmoForCausalLM.from_pretrained(config.model.model_path)

    elif "Qwen" in config.model.model_path:
        model = AutoModelForCausalLM.from_pretrained(config.model.model_path)

    if model is not None and getattr(config, "dense_to_moe", False):
        noise = getattr(config, "noise", False) and noise
        logger.info(f"dense_model_to_moe noise {noise}")
        model = dense_model_to_moe(
            config,
            model,
            topk=config.moe.full_topk if full else config.moe.partial_topk,
            noise=noise,
            noise_std=getattr(config, "noise_std", 0.02),
            expert_group_assignment=expert_group_assignment,
        )

    if model is not None and getattr(config, "torch_compile", False):
        model = torch.compile(model)

    return model


def get_base_tokenizer(config: Config):
    """
    Load the tokenizer for `config.model.model_path`.

    Notes
    -----
    * If `HF_TOKEN` is set in the environment, we call `huggingface_hub.login` with it.
      Otherwise we rely on cached credentials (e.g., `huggingface-cli login`).
    * Sets `pad_token` to `"</s>"` for causal LM padding compatibility.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, use_fast=True)
    tokenizer.pad_token = "</s>"
    return tokenizer

def noise_injection(weight: torch.Tensor, noise_ratio: float = 0.5, init_std: float = 0.02) -> torch.Tensor:
    """
    Randomly replace a fraction of weights with Gaussian noise.

    Parameters
    ----------
    weight : Tensor
        Source tensor (modified in-place).
    noise_ratio : float
        Fraction of elements to replace with noise.
    init_std : float
        Std for the injected Gaussian noise.

    Returns
    -------
    Tensor
        The modified `weight`.
    """
    mask = torch.FloatTensor(weight.size()).uniform_() < noise_ratio
    mask = mask.to(weight.device)
    rand_weight = torch.nn.init.normal_(copy.deepcopy(weight), mean=0.0, std=init_std)
    weight[mask] = rand_weight[mask]
    return weight


def dense_model_to_moe(
    config: Config,
    dense_model: nn.Module,
    topk: int,
    noise: bool = False,
    noise_std: Optional[float] = None,
    expert_group_assignment: Dict[int, list] | None = None,
) -> MyceliaMoE:
    """
    Convert a dense transformer model to an OLMoE-structured model by:
      * Injecting a router gate and expanding MLP weights into expert shards.
      * Optionally injecting initialization noise into expert parameters.
      * Mapping certain naming differences (e.g., post_feedforward -> input).

    Parameters
    ----------
    dense_model : nn.Module
        A pretrained dense Causal LM (e.g., LLaMA).
    num_experts : int
        Number of experts to create per MoE layer.
    topk : int
        Number of experts to route each token to (num_experts_per_tok).
    noise : bool
        Whether to inject noise when cloning dense weights into expert slots.
    noise_std : Optional[float]
        Std of the initialization noise (default 0.02).
    interleave : bool
        If True, convert every other MLP layer to MoE (even indices).

    Returns
    -------
    CustomMoE
        An OLMoE-structured model initialized from the dense weights.
    """
    mlp_layer_name = {"w1": "gate_proj", "w2": "up_proj", "w3": "down_proj"}
    layer_name_mapping = {"post_feedforward": "input"}  # compatibility tweak

    sd = dense_model.state_dict()

    hidden_size = dense_model.config.hidden_size
    intermediate_size = dense_model.config.intermediate_size

    moe_sd: Dict[str, torch.Tensor] = {}

    # Some models may lack q/k normalization; synthesize if needed later.
    has_layer_norm = "k_norm" in dense_model.model.layers[0].self_attn.__dir__()

    for key in list(sd.keys()):
        # Try to infer the layer index from the parameter name
        if "layers." in key:
            start = key.find("layers.") + len("layers.")
            end = key.find(".", start)
            layer_index = int(key[start:end])
        else:
            layer_index = None

        # Convert target MLP layers into MoE (interleave == even layers or all if interleave=False)
        if mlp_layer_name["w1"] in key and (
            (config.moe.interleave and layer_index is not None and layer_index % 2 == 0) or not config.moe.interleave
        ):
            layer_prefix = key[: key.find("mlp.") + len("mlp.")]
            layer_suffix = key[key.find("mlp.") + len("mlp.") :]

            # Router gate weights (E x H)
            moe_sd[layer_prefix + "gate.weight"] = torch.zeros((config.moe.num_experts, hidden_size), device=sd[key].device)

            if noise and noise_std is not None:
                moe_sd[layer_prefix + "gate.weight"] = moe_sd[layer_prefix + "gate.weight"].normal_(std=noise_std)

            # Gather dense MLP weights we will replicate per-expert
            expert_suffix = {
                "w1": layer_suffix,
                "w2": layer_suffix.replace(mlp_layer_name["w1"], mlp_layer_name["w2"]),
                "w3": layer_suffix.replace(mlp_layer_name["w1"], mlp_layer_name["w3"]),
            }

            if intermediate_size is None:
                intermediate_size = sd[layer_prefix + expert_suffix["w1"]].shape[0]

            expert_layers = {}
            for suf in expert_suffix.values():
                name = layer_prefix + suf
                expert_layers[suf] = sd.pop(name)

            # Replicate dense weights into each expert (optionally with noise)
            for expert_id in range(config.moe.num_experts):
                for suf in expert_suffix.values():
                    src = expert_layers[suf]
                    dst_name = f"{layer_prefix}experts.{expert_id}.{suf}"
                    if noise:
                        moe_sd[dst_name] = noise_injection(src.clone(), init_std=noise_std or 0.02)
                    else:
                        moe_sd[dst_name] = src.clone()

        # Pass-through for everything else (with minor renaming)
        elif key in sd:
            new_key = key
            for old, new in layer_name_mapping.items():
                new_key = new_key.replace(old, new)
            moe_sd[new_key] = sd.pop(key)

            # # If attention lacks q/k norms, synthesize them when we see o_proj
            if not has_layer_norm and "self_attn.o_proj.weight" in new_key:
                moe_sd[new_key.replace("o_proj", "q_norm")] = torch.ones(hidden_size, device=moe_sd[new_key].device)
                moe_sd[new_key.replace("o_proj", "k_norm")] = torch.ones(hidden_size, device=moe_sd[new_key].device)

        else:
            # already processed
            pass

    # Start from a base OLMoE config, then copy overlapping fields from the dense config
    moe_config = get_model_config(config, topk)
    dense_config = dense_model.config
    for k, v in dense_config.to_dict().items():
        setattr(moe_config, k, v)

    model = MyceliaMoE(config, moe_config, expert_group_assignment=expert_group_assignment)

    _missing, _unexpected = model.load_state_dict(moe_sd, strict=True)  # will raise if mismatch
    return model

def get_model_config(config: Config, topk: int, org_model: nn.Module = None) -> PretrainedConfig:

    # get the base config from qwen model 
    base_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    
    # merge our subnet config to the base config 
    base_config.num_experts = int(config.moe.num_experts)
    base_config.num_experts_per_tok = int(topk)
    base_config.interleave = bool(config.moe.interleave)
    base_config.output_router_logits = getattr(config, "aux_load_balance", False)
    base_config.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", False)
    base_config.norm_topk_prob = True

    # merge the existing model config into the base config
    if org_model is not None:
        org_model_config = org_model.config
        for k, v in org_model_config.to_dict().items():
            setattr(base_config, k, v)
    
    return base_config

def get_layer_expert_id(layer_name: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract (layer_id, expert_id) from a parameter name.

    Examples
    --------
    "model.layers.3.mlp.experts.7.w1.weight" -> (3, 7)
    "model.layers.5.mlp.gate.weight"         -> (5, None)
    """
    m = re.search(r"model\.layers\.(\d+)(?:\.mlp\.experts\.(\d+))?", layer_name)
    if not m:
        return None, None
    layer_id = int(m.group(1))
    expert_id = int(m.group(2)) if m.group(2) is not None else None
    return layer_id, expert_id


def partial_moe(
    config: Config,
    moe_model: MyceliaMoE,
    my_group_id: int,
    expert_group_assignment: Dict[int, Dict[int, List[int]]],
) -> MyceliaMoE:
    """
    Build a partial MoE that retains only experts for `my_group_id`.

    Parameters
    ----------
    moe_model : CustomMoE
        A full OLMoE-structured model.
    my_group_id : int
        Group to retain.
    expert_group_assignment : Dict[int, Dict[int, List[int]]]
        Mapping: layer_id -> (group_id -> list of expert ids).

    Returns
    -------
    CustomMoE
        A partial model with only this group's experts instantiated/loaded.
    """
    sd = moe_model.state_dict()

    # Remove parameters of experts not owned by this group
    for k in list(sd.keys()):
        layer_id, expert_id = get_layer_expert_id(k)
        if expert_id is not None and expert_id not in expert_group_assignment[layer_id][my_group_id]:
            del sd[k]

    topk = getattr(moe_model.config, "num_experts_per_tok", 2)
    moe_config = get_model_config(config, topk, moe_model)

    partial = MyceliaMoE(
        config, moe_config, my_group_id=my_group_id, expert_group_assignment=expert_group_assignment, partial=True
    )

    if partial is not None and getattr(config, "torch_compile", False):
        partial = torch.compile(partial)

    partial.load_state_dict(sd, strict=True)  # partial by design
    return partial
