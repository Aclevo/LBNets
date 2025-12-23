"""
PhiForLogicalReasoning - Phi with built-in chain-of-thought reasoning.
Features: 
- Per-Token Reasoning (Causal Fix)
- Chunked Processing (OOM Fix)
- Zero-Initialized Gating (Stability Fix)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

# Import base Phi components from transformers
from transformers.models.phi.modeling_phi import (
    PhiDecoderLayer,
    PhiRotaryEmbedding,
    PhiPreTrainedModel,
)

# Import your local config
from .configuration import PhiReasoningConfig


logger = logging.get_logger(__name__)


# =============================================================================
# Output Classes
# =============================================================================

@dataclass
class ReasoningModelOutput(ModelOutput):
    """Output from PhiReasoningModel."""
    last_hidden_state: torch.FloatTensor = None
    reasoning_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    reasoning_used: Optional[torch.BoolTensor] = None
    halting_step: Optional[torch.LongTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass 
class ReasoningCausalLMOutput(ModelOutput):
    """Output from PhiForLogicalReasoning."""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    reasoning_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    reasoning_used: Optional[torch.BoolTensor] = None
    halting_step: Optional[torch.LongTensor] = None
    auxiliary_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# =============================================================================
# Reasoning Components
# =============================================================================

class LatentReasoningTokens(nn.Module):
    """
    Learnable latent tokens representing internal thoughts.
    """
    def __init__(self, config: PhiReasoningConfig):
        super().__init__()
        self.num_tokens = config.num_reasoning_tokens
        self.hidden_size = config.hidden_size
        
        self.embeddings = nn.Parameter(
            torch.randn(1, self.num_tokens, self.hidden_size) * 0.02
        )
        self.step_embeddings = nn.Embedding(config.max_reasoning_steps, self.hidden_size)
        
    def forward(self, batch_size: int, step: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tokens = self.embeddings.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        step_emb = self.step_embeddings(torch.tensor([step], device=device)).unsqueeze(1)
        return tokens + step_emb


class InputComplexityGate(nn.Module):
    def __init__(self, config: PhiReasoningConfig):
        super().__init__()
        self.threshold = config.gating_threshold
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.reasoning_dropout),
            nn.Linear(config.hidden_size // 4, 1),
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        pooled = hidden_states.mean(dim=1)
        score = torch.sigmoid(self.gate(pooled).squeeze(-1))
        needs_reasoning = score > self.threshold
        return score, needs_reasoning


class ReasoningAttention(nn.Module):
    """
    Custom attention for reasoning. Robust to generation (seq_len=1).
    """
    def __init__(self, config: PhiReasoningConfig, is_cross_attention: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.is_cross_attention = is_cross_attention
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.reasoning_dropout)
        
    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Normalize inputs to 3D [Batch, Seq, Dim]
        if hidden_states.dim() == 2: hidden_states = hidden_states.unsqueeze(1)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        if key_value_states is None:
            key_value_states = hidden_states
        elif key_value_states.dim() == 2:
            key_value_states = key_value_states.unsqueeze(1)
        
        kv_seq_len = key_value_states.shape[1]
        
        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(key_value_states)
        v = self.v_proj(key_value_states)
        
        # Reshape [Batch, Heads, Seq, Dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Mask handling (robust for generation)
        if attention_mask is not None:
            # Reshape/Expand mask to match attention scores
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, None, :, :]
            else:
                mask = attention_mask
            
            # During generation, context might be shorter than mask (mask includes history)
            # We slice the mask to match the current KV length
            if mask.shape[-1] != kv_seq_len:
                mask = mask[..., -kv_seq_len:]
                
            mask = (1.0 - mask.to(attn_weights.dtype)) * torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.out_proj(output)


class ReasoningBlock(nn.Module):
    def __init__(self, config: PhiReasoningConfig):
        super().__init__()
        self.cross_attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.cross_attn = ReasoningAttention(config, is_cross_attention=True)
        self.self_attn = ReasoningAttention(config, is_cross_attention=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            ACT2FN[config.hidden_act],
            nn.Dropout(config.reasoning_dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.reasoning_dropout),
        )
        
    def forward(self, reasoning_states: torch.Tensor, context_states: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross Attn
        residual = reasoning_states
        normed = self.cross_attn_norm(reasoning_states)
        reasoning_states = residual + self.cross_attn(normed, key_value_states=context_states, attention_mask=context_mask)
        
        # Self Attn
        residual = reasoning_states
        normed = self.self_attn_norm(reasoning_states)
        reasoning_states = residual + self.self_attn(normed)
        
        # MLP
        residual = reasoning_states
        normed = self.mlp_norm(reasoning_states)
        reasoning_states = residual + self.mlp(normed)
        return reasoning_states


class AdaptiveHalting(nn.Module):
    def __init__(self, config: PhiReasoningConfig):
        super().__init__()
        self.threshold = config.halting_threshold
        self.min_steps = config.min_reasoning_steps
        self.max_steps = config.max_reasoning_steps
        self.halt_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
        )
        
    def forward(self, reasoning_states: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.BoolTensor]:
        pooled = reasoning_states.mean(dim=1)
        halt_prob = torch.sigmoid(self.halt_predictor(pooled).squeeze(-1))
        
        if step < self.min_steps:
            should_halt = torch.zeros_like(halt_prob, dtype=torch.bool)
        elif step >= self.max_steps - 1:
            should_halt = torch.ones_like(halt_prob, dtype=torch.bool)
        else:
            should_halt = halt_prob > self.threshold
        return halt_prob, should_halt


class ReasoningInjector(nn.Module):
    def __init__(self, config: PhiReasoningConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attn = ReasoningAttention(config, is_cross_attention=True)
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(config.reasoning_dropout)

        # --- ✅ CRITICAL FIX: ZERO INITIALIZATION ---
        # Initialize the gate to be effectively closed (output ~0.0)
        # Weight = 0, Bias = -10. Sigmoid(-10) is roughly 0.000045
        # This prevents the random reasoning noise from destroying the model.
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -10.0) 
        # --------------------------------------------
        
    def forward(self, hidden_states: torch.Tensor, reasoning_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_normed = self.norm(hidden_states)
        reasoning_info = self.cross_attn(hidden_normed, key_value_states=reasoning_states)
        
        gate_input = torch.cat([hidden_states, reasoning_info], dim=-1)
        gate_val = self.gate(gate_input)
        return residual + self.dropout(gate_val * reasoning_info)


# =============================================================================
# Main Models
# =============================================================================

class PhiReasoningModel(PhiPreTrainedModel):
    config_class = PhiReasoningConfig
    
    def __init__(self, config: PhiReasoningConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        
        injection_point = config.reasoning_injection_point
        self.pre_reasoning_layers = nn.ModuleList([
            PhiDecoderLayer(config, layer_idx) for layer_idx in range(injection_point)
        ])
        self.post_reasoning_layers = nn.ModuleList([
            PhiDecoderLayer(config, layer_idx + injection_point) for layer_idx in range(config.num_hidden_layers - injection_point)
        ])
        
        self.reasoning_tokens = LatentReasoningTokens(config)
        if config.share_reasoning_layers:
            block = ReasoningBlock(config)
            self.reasoning_blocks = nn.ModuleList([block] * config.num_reasoning_layers)
        else:
            self.reasoning_blocks = nn.ModuleList([ReasoningBlock(config) for _ in range(config.num_reasoning_layers)])
            
        self.input_gate = InputComplexityGate(config) if config.use_input_gating else None
        self.halting = AdaptiveHalting(config) if config.use_adaptive_halting else None
        self.reasoning_injector = ReasoningInjector(config)
        
        self.rotary_emb = PhiRotaryEmbedding(config=config)
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()
        
    def get_input_embeddings(self): return self.embed_tokens
    def set_input_embeddings(self, value): self.embed_tokens = value
        
    def _run_reasoning_loop(self, context_states: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        # Context is [Total_Tokens, 1, Dim]
        total_tokens = context_states.shape[0]
        device = context_states.device
        dtype = context_states.dtype
        
        chunk_size = 128
        final_reasoning_states = []
        dummy_history = [] 

        # Import checkpoint function
        from torch.utils.checkpoint import checkpoint

        for i in range(0, total_tokens, chunk_size):
            ctx_chunk = context_states[i : i + chunk_size]
            current_batch_size = ctx_chunk.shape[0]
            
            # Initialize reasoning tokens
            reasoning_chunk = self.reasoning_tokens(current_batch_size, 0, device, dtype)
            
            # Run steps with manual checkpointing if training
            for step in range(self.config.max_reasoning_steps):
                if step > 0:
                    step_tokens = self.reasoning_tokens(current_batch_size, step, device, dtype)
                    reasoning_chunk = reasoning_chunk + 0.1 * step_tokens
                
                for block in self.reasoning_blocks:
                    # Apply Gradient Checkpointing to save VRAM
                    if self.training and reasoning_chunk.requires_grad:
                        # We must ensure inputs require grad for checkpointing to work 
                        # (reasoning_chunk usually does)
                        reasoning_chunk = checkpoint(block, reasoning_chunk, ctx_chunk, use_reentrant=False)
                    else:
                        reasoning_chunk = block(reasoning_chunk, ctx_chunk)
            
            final_reasoning_states.append(reasoning_chunk)
            
        full_reasoning_states = torch.cat(final_reasoning_states, dim=0)
        return full_reasoning_states, dummy_history, 0
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, output_reasoning_states=True, cache_position=None, **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        # --- SHAPE NORMALIZATION ---
        if inputs_embeds.dim() == 3:
             if inputs_embeds.shape[0] > inputs_embeds.shape[1] and inputs_embeds.shape[1] < 256:
                 inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

        # --- MASK NORMALIZATION ---
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            if attention_mask.dim() == 2 and attention_mask.shape[0] > attention_mask.shape[1]:
                 attention_mask = attention_mask.transpose(0, 1).contiguous()
            if attention_mask.dim() == 2:
                # [B, S] -> [B, 1, 1, S]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        # ---------------------------

        batch_size, seq_length = inputs_embeds.shape[:2]
        device = inputs_embeds.device
        
        if position_ids is None:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device).unsqueeze(0)
            
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
            
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + seq_length, device=device)
            
        hidden_states = self.embed_dropout(inputs_embeds)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Pre-reasoning
        for layer in self.pre_reasoning_layers:
            if output_hidden_states: all_hidden_states += (hidden_states,)
            layer_outputs = layer(hidden_states, position_embeddings, attention_mask, past_key_values, cache_position)
            hidden_states = layer_outputs[0]
            if output_attentions: all_attentions += (layer_outputs[1],)
            
        # Reasoning Loop (Per-Token Causal)
        reasoning_states = None
        all_reasoning_states = None
        
        hidden_flat = hidden_states.view(-1, 1, self.config.hidden_size)
        should_reason = True
        
        if should_reason:
            reasoning_states, _, _ = self._run_reasoning_loop(hidden_flat)
            hidden_flat = self.reasoning_injector(hidden_flat, reasoning_states)
            hidden_states = hidden_flat.view(batch_size, seq_length, self.config.hidden_size)
        
        # Post-reasoning
        for layer in self.post_reasoning_layers:
            if output_hidden_states: all_hidden_states += (hidden_states,)
            layer_outputs = layer(hidden_states, position_embeddings, attention_mask, past_key_values, cache_position)
            hidden_states = layer_outputs[0]
            if output_attentions: all_attentions += (layer_outputs[1],)
            
        hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states: all_hidden_states += (hidden_states,)
        
        return ReasoningModelOutput(
            last_hidden_state=hidden_states,
            reasoning_states=tuple(all_reasoning_states) if all_reasoning_states else None,
            reasoning_used=should_reason,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class PhiForLogicalReasoning(PhiPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: PhiReasoningConfig):
        super().__init__(config)
        self.model = PhiReasoningModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.post_init()
        
    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, value): self.model.embed_tokens = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings
    
    def _compute_auxiliary_loss(self, reasoning_states):
        # Aux loss skipped for chunked implementation to save memory
        return torch.tensor(0.0, device=self.device)
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, output_reasoning_states=None, return_dict=None, cache_position=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
            use_cache, output_attentions, output_hidden_states, output_reasoning_states, cache_position
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss = lm_loss # Aux loss removed for efficiency
            
        if not return_dict:
            return ((loss,) + (logits,) + outputs[1:]) if loss is not None else ((logits,) + outputs[1:])
            
        return ReasoningCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs):
        if past_key_values is not None:
            if input_ids.shape[1] != 1: input_ids = input_ids[:, -1:]
        model_inputs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None and past_key_values is None else {"input_ids": input_ids}
        model_inputs.update({
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "use_cache": kwargs.get("use_cache", True),
        })
        return model_inputs
