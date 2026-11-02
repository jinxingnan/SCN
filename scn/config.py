"""
SCN Configuration

Defines all hyperparameters and settings for the SCN model.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class SCNConfig:
    """
    Configuration for SCN (Structurally Constrained Navigation) model.

    SCN uses bidirectional alignment:
    - Input side: Soft prompts inject KGE vectors into LLM (solves missing name problem)
    - Output side: Vector navigation maps LLM intent to KGE space

    The model consists of:
    - Frozen LLM backbone with optional LoRA adapters
    - Input Projector (KGE -> LLM): Maps entity vectors to soft tokens
    - Output Projector (LLM -> KGE): Maps hidden states to navigation vectors
    - Frozen KGE embeddings (RotatE) for knowledge graph navigation
    """

    # === LLM Backbone ===
    llm_model_name: str = "meta-llama/Llama-2-7b-hf"
    llm_hidden_size: int = 4096

    # === LoRA Configuration ===
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    # Customizable target modules - default to q_proj and v_proj
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    # Gradient checkpointing - trade computation for memory (saves 30-50% VRAM)
    use_gradient_checkpointing: bool = False

    # === KGE (Knowledge Graph Embeddings) ===
    kge_dim: int = 512                      # KGE embedding dimension (matches RotatE)
    kge_raw_dim: int = 512                  # RotatE raw dimension
    fusion_type: str = "weighted"           # "rotate", "transe", or "weighted" (SCN default)
    fusion_lambda: float = 0.7              # Weight for relation in weighted fusion (λ)
                                            # Higher = more relation-dominant (recommended: 0.5-0.9)

    # === Input Projector (KGE -> LLM space) [SCN] ===
    input_projector_hidden_dim: Optional[int] = None  # Default: (kge_dim + llm_hidden_size) // 2
    input_projector_dropout: float = 0.3    # Critical for preventing overfitting to KGE patterns
    input_projector_lr: float = 1e-4        # Learning rate for input projector

    # === Output Projector (LLM -> KGE space) ===
    projector_hidden_dims: List[int] = field(
        default_factory=lambda: [2048, 1024]
    )
    projector_dropout: float = 0.1
    projector_activation: str = "gelu"      # "gelu" or "relu"

    # === Training ===
    temperature: float = 0.1                # InfoNCE temperature τ (0.1 recommended for stability)
    hard_negative_weight: float = 2.0       # α: weight for spurious path negatives
    max_hops: int = 10                      # Maximum reasoning steps
    use_multi_path_per_pair: bool = False   # If True: train all paths for same (topic, answer) pair
                                            # If False: only one path per (topic, answer) pair
                                            # (still trains different pairs from same question)

    # Learning rate settings
    learning_rate: float = 1e-4
    projector_lr: float = 1e-4              # Can use different LR for projector
    lora_lr: float = 1e-4                   # Can use different LR for LoRA
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Batch and accumulation
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 20

    # === Inference ===
    beam_size: int = 5
    inference_temperature: float = 1.0      # Temperature during inference

    # === Data ===
    train_ratio: float = 0.9                # Train/val split ratio when only train file provided
    train_file: Optional[str] = None
    val_file: Optional[str] = None

    # === Paths ===
    kge_embeddings_dir: Optional[str] = None
    kge_mapping_dir: Optional[str] = None
    output_dir: str = "./output_scn"

    # === Logging ===
    log_interval: int = 10
    val_interval: int = 1                   # Validate every N epochs
    save_best_only: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_fusion_types = ("rotate", "transe", "weighted")
        if self.fusion_type not in valid_fusion_types:
            raise ValueError(f"fusion_type must be one of {valid_fusion_types}, got '{self.fusion_type}'")

        if self.fusion_lambda < 0.0 or self.fusion_lambda > 1.0:
            raise ValueError(f"fusion_lambda must be in [0, 1], got {self.fusion_lambda}")

        if self.input_projector_dropout < 0.0 or self.input_projector_dropout > 1.0:
            raise ValueError(f"input_projector_dropout must be in [0, 1], got {self.input_projector_dropout}")

        if self.projector_activation not in ("gelu", "relu"):
            raise ValueError(f"projector_activation must be 'gelu' or 'relu', got '{self.projector_activation}'")

        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

        if self.hard_negative_weight < 1.0:
            raise ValueError(f"hard_negative_weight should be >= 1.0, got {self.hard_negative_weight}")

        if self.max_hops < 1:
            raise ValueError(f"max_hops must be at least 1, got {self.max_hops}")

        if self.beam_size < 1:
            raise ValueError(f"beam_size must be at least 1, got {self.beam_size}")

    # Deprecated parameters that should be ignored when loading old configs
    _DEPRECATED_PARAMS = {"first_step_weight"}

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SCNConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Ignore deprecated configuration parameters
        for param in cls._DEPRECATED_PARAMS:
            if param in config_dict:
                print(f"  [Warning] Ignoring deprecated config parameter: {param}")
                del config_dict[param]

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict

        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        from dataclasses import asdict
        return asdict(self)
