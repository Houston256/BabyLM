import os

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from BabyLM.config import MODEL_BASE_PATH
from BabyLM.data_handler.data_handler import load_data

class BabyLMTokenizer:
    base_path = MODEL_BASE_PATH
    def __init__(self, vocab_size: int = 32000, save_path: str = ""):
        self.vocab_size = vocab_size
        
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()])
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer_path = os.path.join(MODEL_BASE_PATH, save_path)

    @staticmethod
    def get_training_corpus(dataset, batch_size=1000):
        """Generator to yield batches of text for tokenizer training."""
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    def train(self):
        """Trains the tokenizer on the dataset provided by the data handler."""
        dataset_dict = load_data(download_locally=True)
        dataset = dataset_dict["train"] if "train" in dataset_dict else dataset_dict

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            show_progress=True
        )

        print("Training tokenizer...")
        self.tokenizer.train_from_iterator(self.get_training_corpus(dataset), trainer=trainer)
        
        os.makedirs(os.path.dirname(self.tokenizer_path), exist_ok=True)
        self.tokenizer.save(self.tokenizer_path)
        print(f"Tokenizer saved successfully to {self.tokenizer_path}")

    @classmethod
    def load(cls, load_path=os.path.join(MODEL_BASE_PATH, "tokenizer.json")):
        """Loads a pre-trained tokenizer from a file."""
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(load_path)
        return instance
