"""
MoE model utilities: convert dense models to MoE-style, define custom sparse blocks,
and provide helpers for partial (grouped) expert execution.

Contents
--------
- DenseBlock: wraps DeepseekV3 MLP to behave like a "dense" block while preserving API
- myceliaSparseMoeBlock: sparse MoE block that activates only a subset of experts (by group)
- CustomMoE: an DeepseekV3 model variant that interleaves MoE and dense blocks
- get_base_model: load base LLaMA or OLMo, optionally convert to MoE
- get_base_tokenizer: load tokenizer (HF login via env var if provided)
- dense_model_to_moe: transform a dense model into DeepseekV3 parameter layout
- get_layer_expert_id: parse layer/expert indices from parameter names
- partial_moe: drop experts outside the current group and rebuild a partial model

Assumptions
-----------
- Parameter names use DeepseekV3/transformers conventions (e.g., "model.layers.{i}.mlp.experts.{e}").
- Experts are identified by the "experts.{expert_id}" path segment.
- Your `Config` provides fields used by `get_base_model` and `get_base_tokenizer`.
"""

from __future__ import annotations

import os
import re
import copy
import logging
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

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
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MoE,  
    DeepseekV3MLP,  
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3TopkRouter,
)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.utils.deprecation import deprecate_kwarg
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from mycelia.shared.config import MinerConfig, ValidatorConfig
from mycelia.shared.app_logging import structlog  
from mycelia.shared.modeling.modeling_deepseek import DeepseekAttention  
from mycelia.shared.helper import *  

logger = structlog.get_logger(__name__)

class TopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config, available_experts = None):
        super().__init__(config)
        if available_experts is not None:
            self.available_experts = torch.as_tensor(available_experts)

        self.weight = nn.Parameter(torch.zeros((self.n_routed_experts, config.hidden_size)))

    # @torch.no_grad()
    def _mask_routing_weights(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Zero-out routing weights for experts not present in this group.
        """
        mask_1d = torch.zeros(x.size(dim), dtype=torch.bool, device=x.device)
        mask_1d[self.available_experts.to(x.device)] = True

        # Broadcast mask across all other dims
        shape = [1] * x.ndim
        shape[dim] = x.size(dim)
        mask = mask_1d.view(shape).to(dtype=x.dtype)

        # logger.info("masking", mask)
        return x * mask
    
    # @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        scores_for_choice = self._mask_routing_weights(scores_for_choice)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    # TODO: no customisation needed here, remove
    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights
class SparseMoeBlock(DeepseekV3MoE):
    """
    Sparse MoE block that only uses a subset of experts assigned to the current group.

    Parameters
    ----------
    config : DeepseekV3Config-like
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
        super().__init__(config)
        self.num_experts: int = config.num_experts
        self.top_k: int = config.num_experts_per_tok
        self.norm_topk_prob: bool = getattr(config,"norm_topk_prob", True)
        self.layer_id: int = layer_id
        self.expert_group_assignment = expert_group_assignment

        # Only instantiate experts owned by this group at this layer
        if my_group_id is not None and expert_group_assignment is not None:
            allowed = expert_group_assignment[layer_id][my_group_id]
        
        elif num_experts is not None:
            allowed = list(range(num_experts))
        
        self.experts = nn.ModuleDict({str(k): DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size) for k in allowed})

        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        
        self.available_experts = torch.as_tensor([int(k) for k in self.experts.keys()])

        self.gate = TopkRouter(config, self.available_experts)

    # TODO: double check is there any customization here, may remove
    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in self.available_experts.tolist():
            expert = self.experts[str(expert_idx)]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        
        return final_hidden_states.type(hidden_states.dtype)
class MyceliaMoE(DeepseekV3ForCausalLM):
    """
    DeepseekV3 variant that interleaves MoE and dense blocks and optionally restricts experts
    to those owned by the calling group.

    If `partial=True`, MoE blocks become `myceliaSparseMoeBlock` limited to the group’s experts.
    Otherwise, standard `SparseMoeBlock` is used for MoE layers.
    """

    def __init__(
        self,
        config: MinerConfig,
        model_config: PretrainedConfig,
        my_group_id: Optional[int] = None,
        expert_group_assignment: Optional[Dict[int, Dict[int, List[int]]]] = None,
        partial: bool = False,
    ):
        super().__init__(model_config)
        layers: List[nn.Module] = []

        for i in range(model_config.num_hidden_layers):
            layer = DeepseekV3DecoderLayer(model_config, layer_idx=i)

            # layer.self_attn = DeepseekAttention(model_config)
            
            # Interleave MoE and dense layers (MoE on odd indices if interleave=True)
            if getattr(model_config, "interleave", True) and (i + 1) % model_config.decoder_sparse_step == 0:
                if not partial:
                    layer.mlp = SparseMoeBlock(
                        model_config, i, num_experts=config.moe.num_experts, expert_group_assignment=expert_group_assignment
                    )  # full MoE (all experts)
                else:
                    assert (
                        my_group_id is not None and expert_group_assignment is not None
                    ), "partial=True requires my_group_id and expert_group_assignment"
                    layer.mlp = SparseMoeBlock(
                        model_config, i, my_group_id=my_group_id, expert_group_assignment=expert_group_assignment
                    )
            
            elif i == 0:
                layer.mlp = DeepseekV3MLP(model_config, intermediate_size=10944)

            else:
                layer.mlp = DeepseekV3MLP(model_config)
            
            layers.append(layer)

        self.model.layers = nn.ModuleList(layers)

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     output_router_logits: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     logits_to_keep: Union[int, torch.Tensor] = 0,
    #     **kwargs,
    # ) -> Union[tuple, MoeCausalLMOutputWithPast]:
    #     r"""
    #     labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #         Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
    #         config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
    #         (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    #     Example:

    #     ```python
    #     >>> from transformers import AutoTokenizer, OlmoeForCausalLM

    #     >>> model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
    #     >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

    #     >>> prompt = "Hey, are you conscious? Can you talk to me?"
    #     >>> inputs = tokenizer(prompt, return_tensors="pt")

    #     >>> # Generate
    #     >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    #     >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     'Hey, are you conscious? Can you talk to me?\nI’m not sure if you’re conscious of this, but I’m'
    #     ```
    #     """
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_router_logits = (
    #         output_router_logits if output_router_logits is not None else self.config.output_router_logits
    #     )

    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
    #     # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         output_router_logits=output_router_logits,
    #         return_dict=return_dict,
    #         cache_position=cache_position,
    #     )
        
    #     hidden_states = outputs[0]
    #     # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    #     slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    #     logits = self.lm_head(hidden_states[:, slice_indices, :])

    #     loss = None
    #     if labels is not None:
    #         loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

    #     aux_loss = None
    #     if output_router_logits:
    #         aux_loss_sum = 0
    #         aux_count = 0
    #         for r in outputs.router_logits if return_dict else outputs[-1]:
    #             aux_loss_sum += load_balancing_loss_func(
    #                 tuple([r]),
    #                 attention_mask=attention_mask,
    #                 num_experts=self.num_experts,
    #                 top_k=self.num_experts_per_tok,
    #             )
    #             aux_count += 1
    #         aux_loss = (aux_loss_sum / aux_count) - self.num_experts_per_tok

    #         if labels is not None:
    #             loss += self.router_aux_loss_coef * aux_loss.to(
    #                 loss.device
    #             )  # make sure to reside in the same device

    #     if not return_dict:
    #         output = (logits,) + outputs[1:]
    #         if output_router_logits:
    #             output = (aux_loss,) + output
    #         return (loss,) + output if loss is not None else output

    #     return MoeCausalLMOutputWithPast(
    #         loss=loss,
    #         aux_loss=aux_loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #         router_logits=outputs.router_logits if output_router_logits else None,
    #     )

# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def get_base_model(
    config: MinerConfig, expert_group_assignment: Dict[int, Dict[int, List[int]]] | None = None, noise: bool = False, full=False
) -> Optional[LlamaForCausalLM | OlmoForCausalLM]:
    """
    Load a base Causal LM by `config.model.model_path` and optionally convert to MoE.

    Returns
    -------
    Optional[nn.Module]
        A Hugging Face causal LM (LLaMA or OLMo), possibly converted to OLMoE.
    """
    model = None

    if config.model.foundation:
        model_config = AutoConfig.from_pretrained(config.model.model_path , trust_remote_code=True) # TODO: need miner agreement
        moe_config = get_moe_model_config(
            config, 
            config.moe.full_topk if full else config.moe.partial_topk, 
            org_model_config = model_config
        )
        model = MyceliaMoE(config, moe_config, expert_group_assignment=expert_group_assignment)

    elif "llama" in config.model.model_path.lower():
        llama_cfg = LlamaConfig.from_pretrained(config.model.model_path, attn_implementation=config.model.attn_implementation)
        model = LlamaForCausalLM.from_pretrained(config.model.model_path, config=llama_cfg)

    elif "olmo" in config.model.model_path:
        model = OlmoForCausalLM.from_pretrained(config.model.model_path)

    else:
        model = AutoModelForCausalLM.from_pretrained(config.model.model_path, trust_remote_code=True) #TODO: need miner agreement to trust remote code

    # if model is not None and get_nested_attr(config,"moe.dense_to_moe", False):
    #     noise = get_nested_attr(config,"moe.noise", False) and noise
    #     # logger.info(f"extract_model_from_shared_expert", noise = noise)
    #     model = extract_model_from_shared_expert(
    #         config,
    #         model,
    #         topk=config.moe.full_topk if full else config.moe.partial_topk,
    #         noise=noise,
    #         noise_std=get_nested_attr(config,"moe.noise_std", 0.02),
    #         expert_group_assignment=expert_group_assignment,
    #     )

    if model is not None and get_nested_attr(config,"model.torch_compile", False):
        model = torch.compile(model)

    return model

def get_base_tokenizer(config: MinerConfig | ValidatorConfig):
    """
    Load the tokenizer for `config.model.model_path`.

    Notes
    -----
    * If `HF_TOKEN` is set in the environment, we call `huggingface_hub.login` with it.
      Otherwise we rely on cached credentials (e.g., `huggingface-cli login`).
    * Sets `pad_token` to `"</s>"` for causal LM padding compatibility.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, use_fast=True)
    # tokenizer.pad_token = "</s>"
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

def extract_model_from_shared_expert(   
    config: MinerConfig,
    model: nn.Module,
    topk: int,
    noise: bool = False,
    noise_std: Optional[float] = None,
    expert_group_assignment: Dict[int, list] | None = None,
):

    gate_mat_pat = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.gate\.weight$"
    )

    state_dict = model.state_dict()
    moe_config = get_moe_model_config(config, topk, org_model_config = model.config)

    # 1) Find per-layer top expert index from gate.weight
    # TODO: change top expert selection based on sigmoid gate weight & may need to select multiple expert 
    layer_top_expert = {}
    for k, v in state_dict.items():
        m = gate_mat_pat.match(k)
        if not m:
            continue

        layer = int(m.group(1))
        gate_w = v
        try:
            scores = gate_w.sum(dim=1)         # [num_experts]
        
        except Exception as ex:
            raise RuntimeError(
                f"Failed to compute per-expert sums for {k} "
                f"(shape={tuple(getattr(gate_w, 'shape', []))})."
            ) from ex

        try:
            # torch.Tensor argmax
            top_idx = int(scores.argmax().item())
        except Exception:
            # Fallback for non-torch arrays
            top_idx = int(scores.argmax())

        layer_top_expert[layer] = top_idx

    # logger.info("top experts", layer_top_expert)

    # 2) Build new state dict, copying everything except experts that we drop/overwrite
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        layer, expert_id = get_layer_expert_id(k)

        # no layer -> directly copy old 
        if layer is None:
            new_sd[k] = v
            continue 
        

        # no expert -> if layer is even -> skip, else directly copy 
        if expert_id is None:
            if (layer + 1) % 2 == 0: 
                if "mlp.gate.weight" in k:
                    gate_sd = TopkRouter(moe_config).state_dict()
                    new_sd[k] = gate_sd['weight']
                    new_sd[k.replace('weight', 'e_score_correction_bias')] = gate_sd['e_score_correction_bias']
                    continue
                else:
                    new_sd[k] = v
                    continue 

            elif "gate.weight" not in k and "shared" not in k:
                new_sd[k] = v
                continue 
            else:
                continue
            
        if expert_id >= config.moe.num_experts:
            continue

        if layer not in layer_top_expert:
            new_sd[k] = v
            continue 

        top_eid = layer_top_expert[layer]
        # We keep the slot but replace its weight with the top expert's weight
        src_key = k.replace(f"expert.{expert_id}", f"expert.{top_eid}")        
        if (layer + 1) % 2 == 0 :
            src_w = state_dict[src_key]
            try:
                new_sd[k] = src_w.clone()
            except AttributeError:
                new_sd[k] = src_w

        elif expert_id == 0:
            new_sd[k.replace(f"experts.{expert_id}.", "")] = state_dict[src_key].clone()

    # Start from a base OLMoE config, then copy overlapping fields from the dense config
    model = MyceliaMoE(config, moe_config, expert_group_assignment=expert_group_assignment)

    # TODO: strict should be true
    _missing, _unexpected = model.load_state_dict(new_sd, strict=False)  # will raise if mismatch
    return model

def dense_model_to_moe(
    config: MinerConfig,
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
            (config.moe.interleave and layer_index is not None and (layer_index + 1) % 2 == 0) or not config.moe.interleave
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
    moe_config = get_moe_model_config(config, topk, org_model_config = dense_model.config)

    model = MyceliaMoE(config, moe_config, expert_group_assignment=expert_group_assignment)
    
    _missing, _unexpected = model.load_state_dict(moe_sd, strict=True)  # will raise if mismatch
    return model

def get_moe_model_config(config: MinerConfig, topk: int, org_model_config: AutoConfig = None) -> PretrainedConfig:

    # get the base config from qwen model 
    base_config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3") #TODO: need user permission
    # base_config = AutoConfig.from_pretrained(config.model.model_path, trust_remote_code=True) #TODO: need user permission
    
    # merge the existing model config into the base config
    if org_model_config is not None:
        for k, v in org_model_config.to_dict().items():
            setattr(base_config, k, v)
    
    # merge our subnet config to the base config 
    base_config.num_experts = int(config.moe.num_experts)
    base_config.n_routed_experts = int(config.moe.num_experts)
    base_config.n_group = 1
    base_config.topk_group = 1
    base_config.num_experts_per_tok = int(topk)
    base_config.interleave = bool(config.moe.interleave)
    base_config.intermediate_size = base_config.moe_intermediate_size
    base_config.decoder_sparse_step = 2 if bool(config.moe.interleave) else 1
    base_config.output_router_logits = get_nested_attr(config,"moe.aux_load_balance", False)
    base_config.router_aux_loss_coef = get_nested_attr(config,"moe.router_aux_loss_coef", False)
    base_config.norm_topk_prob = True
    base_config.max_position_embeddings = config.data.sequence_length
    
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
    config: MinerConfig,
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
    moe_config = get_moe_model_config(config, topk, moe_model.config)

    partial = MyceliaMoE(
        config, moe_config, my_group_id=my_group_id, expert_group_assignment=expert_group_assignment, partial=True
    )

    if partial is not None and get_nested_attr(config,"model.torch_compile", False):
        partial = torch.compile(partial)

    partial.load_state_dict(sd, strict=True)  # partial by design
    return partial
