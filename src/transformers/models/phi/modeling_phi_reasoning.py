"""
PhiForLogicalReasoning - Optimized for 16GB VRAM.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint # Required for memory saving
from transformers import PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from transformers.models.phi.modeling_phi import (
    PhiDecoderLayer,
    PhiRotaryEmbedding,
    PhiPreTrainedModel,
)
from .configuration import PhiReasoningConfig

logger = logging.get_logger(__name__)

# --- CLASSES ---

@dataclass
class ReasoningModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    reasoning_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    reasoning_used: Optional[torch.BoolTensor] = None
    halting_step: Optional[torch.LongTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass 
class ReasoningCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    reasoning_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    reasoning_used: Optional[torch.BoolTensor] = None
    halting_step: Optional[torch.LongTensor] = None
    auxiliary_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class LatentReasoningTokens(nn.Module):
    def __init__(self, config: PhiReasoningConfig):
        super().__init__()
        self.num_tokens = config.num_reasoning_tokens
        self.hidden_size = config.hidden_size
        self.embeddings = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_size) * 0.02)
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
        if hidden_states.dim() == 2: hidden_states = hidden_states.unsqueeze(1)
        batch_size, seq_len, _ = hidden_states.shape
        if key_value_states is None: key_value_states = hidden_states
        elif key_value_states.dim() == 2: key_value_states = key_value_states.unsqueeze(1)
        kv_seq_len = key_value_states.shape[1]
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value_states).view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value_states).view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            if attention_mask.dim() == 2: attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3: attention_mask = attention_mask[:, None, :, :]
            if attention_mask.shape[-1] != kv_seq_len: attention_mask = attention_mask[..., -kv_seq_len:]
            mask = (1.0 - attention_mask.to(attn_weights.dtype)) * torch.finfo(attn_weights.dtype).min
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
        mlp_size = getattr(config, "reasoning_intermediate_size", 2560)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_size),
            ACT2FN[config.hidden_act],
            nn.Dropout(config.reasoning_dropout),
            nn.Linear(mlp_size, config.hidden_size),
            nn.Dropout(config.reasoning_dropout),
        )
        
    def forward(self, reasoning_states: torch.Tensor, context_states: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = reasoning_states
        normed = self.cross_attn_norm(reasoning_states)
        reasoning_states = residual + self.cross_attn(normed, key_value_states=context_states, attention_mask=context_mask)
        residual = reasoning_states
        normed = self.self_attn_norm(reasoning_states)
        reasoning_states = residual + self.self_attn(normed)
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
        should_halt = (halt_prob > self.threshold) if (step >= self.min_steps and step < self.max_steps - 1) else torch.zeros_like(halt_prob, dtype=torch.bool)
        if step >= self.max_steps - 1: should_halt = torch.ones_like(halt_prob, dtype=torch.bool)
        return halt_prob, should_halt

class ReasoningInjector(nn.Module):
    def __init__(self, config: PhiReasoningConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attn = ReasoningAttention(config, is_cross_attention=True)
        self.gate_scale = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(config.reasoning_dropout)
    def forward(self, hidden_states: torch.Tensor, reasoning_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_normed = self.norm(hidden_states)
        reasoning_info = self.cross_attn(hidden_normed, key_value_states=reasoning_states)
        return residual + self.dropout(reasoning_info * self.gate_scale)

# --- MAIN MODEL ---

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
        self.pre_reasoning_layers = nn.ModuleList([PhiDecoderLayer(config, layer_idx) for layer_idx in range(injection_point)])
        self.post_reasoning_layers = nn.ModuleList([PhiDecoderLayer(config, layer_idx + injection_point) for layer_idx in range(config.num_hidden_layers - injection_point)])
        self.reasoning_tokens = LatentReasoningTokens(config)
        block = ReasoningBlock(config) if config.share_reasoning_layers else None
        self.reasoning_blocks = nn.ModuleList([block if block else ReasoningBlock(config) for _ in range(config.num_reasoning_layers)])
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
        batch_size = context_states.shape[0]
        device = context_states.device
        dtype = context_states.dtype
        
        # ✅ FIX: Chunk size 32 is tiny enough to fit 1024 context on 16GB
        chunk_size = 32
        final_chunks = []
        dummy_history = [] 

        for i in range(0, batch_size, chunk_size):
            ctx_chunk = context_states[i : i + chunk_size]
            current_bs = ctx_chunk.shape[0]
            r_chunk = self.reasoning_tokens(current_bs, 0, device, dtype)
            
            for step in range(self.config.max_reasoning_steps):
                if step > 0:
                    r_chunk = r_chunk + 0.1 * self.reasoning_tokens(current_bs, step, device, dtype)
                
                for block in self.reasoning_blocks:
                    # ✅ FIX: Only checkpoint if the block has trainable parameters
                    # This avoids gradient flow issues with frozen layers
                    has_trainable = any(p.requires_grad for p in block.parameters())
                    if self.training and self.gradient_checkpointing and has_trainable:
                        r_chunk = torch.utils.checkpoint.checkpoint(block, r_chunk, ctx_chunk, use_reentrant=False)
                    else:
                        r_chunk = block(r_chunk, ctx_chunk)
            
            final_chunks.append(r_chunk)
            
        full_reasoning = torch.cat(final_chunks, dim=0)
        return full_reasoning, dummy_history, 0

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, output_reasoning_states=True, cache_position=None, **kwargs):
        if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.dim() == 3 and inputs_embeds.shape[0] > inputs_embeds.shape[1] and inputs_embeds.shape[1] < 256:
             inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool: attention_mask = attention_mask.bool()
            if attention_mask.dim() == 2 and attention_mask.shape[0] > attention_mask.shape[1]: attention_mask = attention_mask.transpose(0, 1).contiguous()
            if attention_mask.dim() == 2: attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        batch_size, seq_length = inputs_embeds.shape[:2]
        device = inputs_embeds.device
        if position_ids is None:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device).unsqueeze(0)
        if use_cache and past_key_values is None: past_key_values = DynamicCache()
        if cache_position is None: cache_position = torch.arange(past_key_values.get_seq_length() if past_key_values is not None else 0, (past_key_values.get_seq_length() if past_key_values is not None else 0) + seq_length, device=device)
            
        hidden_states = self.embed_dropout(inputs_embeds)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for layer in self.pre_reasoning_layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask, past_key_values, cache_position)[0]
            
        # Reasoning
        hidden_flat = hidden_states.view(-1, 1, self.config.hidden_size)
        reasoning_states_flat, _, _ = self._run_reasoning_loop(hidden_flat)
        hidden_flat = self.reasoning_injector(hidden_flat, reasoning_states_flat)
        hidden_states = hidden_flat.view(batch_size, seq_length, self.config.hidden_size)
        
        for layer in self.post_reasoning_layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask, past_key_values, cache_position)[0]
            
        hidden_states = self.final_layernorm(hidden_states)
        return ReasoningModelOutput(last_hidden_state=hidden_states)

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
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, output_reasoning_states=None, return_dict=None, cache_position=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states, output_reasoning_states, cache_position)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        if not return_dict: return ((loss,) + (logits,) + outputs[1:]) if loss is not None else ((logits,) + outputs[1:])
        return ReasoningCausalLMOutput(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs):
        if past_key_values is not None:
            if input_ids.shape[1] != 1: input_ids = input_ids[:, -1:]
        model_inputs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None and past_key_values is None else {"input_ids": input_ids}
        model_inputs.update({"past_key_values": past_key_values, "attention_mask": attention_mask, "cache_position": cache_position, "use_cache": kwargs.get("use_cache", True)})
        return model_inputs
