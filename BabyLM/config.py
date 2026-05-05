from dataclasses import dataclass

DATASET_PATH = "data/"
MODEL_BASE_PATH = "models/"


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    dropout: float = 0.1

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
