from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 16000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    hidden_dim: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0

# V2: 88M parameter model
V2_CONFIG = ModelConfig(
    vocab_size=16000,
    dim=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=4,
    hidden_dim=2048,
    max_seq_len=1024,
    rope_theta=10000.0
)

# V3: 154M Titan parameter model (Legacy)
V3_CONFIG = ModelConfig(
    vocab_size=16000,
    dim=896,
    n_layers=16,
    n_heads=14,
    n_kv_heads=2,
    hidden_dim=2560,
    max_seq_len=4096,
    rope_theta=50000.0
)

# V4: 196M Chinchilla parameter model (Tensor Core Optimized)
V4_CONFIG = ModelConfig(
    vocab_size=16000,
    dim=1024,
    n_layers=16,
    n_heads=16,
    n_kv_heads=4,
    hidden_dim=2816,
    max_seq_len=4096,
    rope_theta=50000.0
)
