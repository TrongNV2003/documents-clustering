import torch
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer
from underthesea import sent_tokenize


class Embedding():
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str,
        pooling_type: str = "cls",
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()
        self.pooling_type = pooling_type
        self.max_length = self.tokenizer.model_max_length

    
    def pool(self, hidden_states, attention_mask):
        if self.pooling_type == "mean":
            masked = hidden_states * attention_mask.unsqueeze(-1)
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1, keepdim=True)
            return summed / counts
        elif self.pooling_type == "max":
            masked = hidden_states.masked_fill(attention_mask.unsqueeze(-1)==0, -1e9)
            return masked.max(dim=1).values
        elif self.pooling_type == "cls":
            return hidden_states[:, 0]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
    
    def embedding(self, chunk: str):
        inputs = self.tokenizer(
            chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        pooled = self.pool(outputs.last_hidden_state, inputs.attention_mask)
        return pooled[0]
    
    def embed_document(self, text: str) -> np.ndarray:
        chunks = chunking_document(text, tokenizer=self.tokenizer, max_tokens=self.max_length)
        embs = [self.embedding(c) for c in chunks]
            
        doc_emb = torch.stack(embs, dim=0).mean(dim=0).cpu().numpy()  # [n_chunks, hidden_size] -> [hidden_size]
        return doc_emb


@staticmethod
def chunking_document(text: str, tokenizer: AutoTokenizer, max_tokens: int = 256) -> list:
    """
    Chia text thành các chunk ≤ max_tokens sub‑token, với overlap nhẹ để giữ context.
    """
    sentences = sent_tokenize(text) # Tách câu
    length = 0
    buf = []
    chunks = []

    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        if length + len(ids) > max_tokens - 2:  # chừa [CLS],[SEP]
            if buf:
                chunks.append(' '.join(buf))
            buf = buf[-1:]
            length = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in buf)
        buf.append(sent)
        length += len(ids)
    if buf:
        chunks.append(". ".join(buf))
    return chunks