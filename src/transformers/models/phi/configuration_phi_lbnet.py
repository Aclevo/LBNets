from transformers import PretrainedConfig

class PhiReasoningConfig(PretrainedConfig):
    model_type = "phi_reasoning"
    
    def __init__(
        self,
        # Standard Phi
        vocab_size: int = 51200,
        hidden_size: int = 2560,
        intermediate_size: int = 10240,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: int = None,
        hidden_act: str = "gelu_new",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 0.5,
        qk_layernorm: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        attention_dropout: float = 0.0,
        embd_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        
        # Reasoning Config
        num_reasoning_tokens: int = 32,
        num_reasoning_steps: int = 8,
        num_reasoning_layers: int = 4,
        max_reasoning_steps: int = 16,
        min_reasoning_steps: int = 2,
        reasoning_dropout: float = 0.1,
        # ✅ NEW: Custom smaller size for reasoning MLPs to save VRAM
        reasoning_intermediate_size: int = 2560, 
        
        use_adaptive_halting: bool = True,
        halting_threshold: float = 0.8,
        use_input_gating: bool = True,
        gating_threshold: float = 0.3,
        
        ponder_loss_weight: float = 0.01,
        consistency_loss_weight: float = 0.001,
        share_reasoning_layers: bool = False,
        reasoning_injection_point: int = None,
        
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.qk_layernorm = qk_layernorm
        self.attention_dropout = attention_dropout
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        
        self.num_reasoning_tokens = num_reasoning_tokens
        self.num_reasoning_steps = num_reasoning_steps
        self.num_reasoning_layers = num_reasoning_layers
        self.max_reasoning_steps = max_reasoning_steps
        self.min_reasoning_steps = min_reasoning_steps
        self.reasoning_dropout = reasoning_dropout
        # Use provided value or default to 1/4th of main model
        self.reasoning_intermediate_size = reasoning_intermediate_size
        
        self.use_adaptive_halting = use_adaptive_halting
        self.halting_threshold = halting_threshold
        self.use_input_gating = use_input_gating
        self.gating_threshold = gating_threshold
        self.ponder_loss_weight = ponder_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.share_reasoning_layers = share_reasoning_layers
        self.reasoning_injection_point = reasoning_injection_point or (num_hidden_layers // 2)
        
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
        }
