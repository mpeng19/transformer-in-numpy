import numpy as np
from datasets import load_dataset
from typing import Iterator, Optional, Dict, Any
from transformers import GPT2Tokenizer

class FineWebDataLoader:
    """DataLoader for FineWeb dataset using only numpy."""
    def __init__(
        self,
        sequence_length: int = 512,
        num_examples: Optional[int] = None,
        batch_size: int = 32,
        name: str = "CC-MAIN-2024-10",
        streaming: bool = True,
        pad_token_id: int = 0,
        use_hf_token: bool = True
    ):
        self.sequence_length = sequence_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.name = name
        self.streaming = streaming
        self.pad_token_id = pad_token_id
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = self._load_dataset()
        self.current_example = 0
        
    def _load_dataset(self):
        """Load the FineWeb dataset."""
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu", 
                name=self.name, 
                split="train", 
                streaming=self.streaming,
            )
        return dataset
    
    def _tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize text using GPT-2 tokenizer."""
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return np.array(tokens, dtype=np.int32)
    
    def _pad_or_truncate(self, tokens: np.ndarray) -> np.ndarray:
        """Pad or truncate tokens to the specified sequence length."""
        if len(tokens) >= self.sequence_length:
            return tokens[:self.sequence_length]
        else:
            padding_length = self.sequence_length - len(tokens)
            padding = np.full(padding_length, self.tokenizer.eos_token_id, dtype=np.int32)
            return np.concatenate([tokens, padding])
    
    def _process_example(self, example: Dict[str, Any]) -> np.ndarray:
        """Process a single example from the dataset."""
        text = example.get('text', '')
        if not text:
            return np.full(self.sequence_length, self.tokenizer.eos_token_id, dtype=np.int32)
        
        tokens = self._tokenize_text(text)
        return self._pad_or_truncate(tokens)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over examples, yielding batches."""
        batch = []
        examples_processed = 0
        
        for example in self.dataset:
            if self.num_examples and examples_processed >= self.num_examples:
                break
                
            processed_example = self._process_example(example)
            batch.append(processed_example)
            examples_processed += 1
            
            if len(batch) == self.batch_size:
                yield np.stack(batch, axis=0)
                batch = []
        
        if batch:
            yield np.stack(batch, axis=0)
    
    def get_single_example(self) -> Optional[np.ndarray]:
        """Get a single processed example."""
        example = next(iter(self.dataset))
        return self._process_example(example)
    
    def get_batch(self, size: Optional[int] = None) -> np.ndarray:
        """Get a single batch of examples."""
        if size is None:
            size = self.batch_size
            
        batch = []
        examples_processed = 0
        
        for example in self.dataset:
            if examples_processed >= size:
                break
                
            processed_example = self._process_example(example)
            batch.append(processed_example)
            examples_processed += 1
        
        return np.stack(batch, axis=0) if batch else np.empty((0, self.sequence_length), dtype=np.int32)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size for GPT-2 tokenizer."""
        return self.tokenizer.vocab_size
    
    def decode_tokens(self, tokens: np.ndarray) -> str:
        """Decode tokens back to text using GPT-2 tokenizer."""
        tokens = tokens[tokens != self.tokenizer.eos_token_id] #remove padding tokens
        return self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
    
    def reset(self):
        """Reset the dataloader (reload dataset for streaming)."""
        self.dataset = self._load_dataset()
        self.current_example = 0
