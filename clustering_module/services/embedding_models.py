import torch
import numpy as np
import torch.nn as nn
from typing import List, Optional
from underthesea import sent_tokenize
from transformers import AutoTokenizer


class Embedding:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
        pooling_type: str = "weighted",
        max_length: int = 256,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.pooling_type = pooling_type
        self.model = model.to(self.device)
        self.max_length = self.tokenizer.model_max_length if max_length is None else max_length
        self.model.eval()

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
        elif self.pooling_type == "weighted":
            weights = attention_mask.unsqueeze(-1).float()
            weighted_sum = (hidden_states * weights).sum(dim=1)
            normalization = weights.sum(dim=1)
            return weighted_sum / normalization
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")


    def embed_document(self, text: str) -> np.ndarray:
        chunks = self.chunking_doc(text, tokenizer=self.tokenizer, max_tokens=self.max_length)
        embs = self.embedding(chunks)
        doc_emb = embs.mean(dim=0).cpu().numpy()
        return doc_emb


    def embedding(self, chunks: List[str]):
        batch_size = 16
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            inputs = self.tokenizer(
                batch_chunks,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            pooled = self.pool(outputs.last_hidden_state, inputs.attention_mask)
            all_embeddings.append(pooled)
            
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            return embeddings
        else:
            return torch.zeros(1, self.model.config.hidden_size, device=self.device)
    

    def chunking_doc(
        self, text: str, tokenizer: AutoTokenizer, max_tokens: int = 256, overlap_ratio: float = 0.2
    ) -> List[str]:
        """
        Split the text into chunks of ≤ max_tokens sub-tokens, with slight overlap for context retention.
        Returns a list of text chunks.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            tokenized_length = len(tokenizer.encode(sent, add_special_tokens=False, truncation=True, max_length=max_tokens))
            
            if current_length + tokenized_length > max_tokens - 2:  # [CLS] and [SEP]
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                overlap_size = max(1, int(len(current_chunk) * overlap_ratio))
                current_chunk = current_chunk[-overlap_size:]
                current_length = sum(len(tokenizer.encode(
                    s, add_special_tokens=False, truncation=True, max_length=max_tokens
                )) for s in current_chunk)
            
            current_chunk.append(sent)
            current_length += tokenized_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

# if __name__ == "__main__":
#     from transformers import AutoModel, AutoTokenizer

#     model_name = "models/vietnamese-sbert"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)

#     embedding_model = Embedding(model=model, tokenizer=tokenizer, pooling_type="weighted")
    
#     text = "Xin chào bạn."
#     embedding = embedding_model.embed_document(text)
#     print(embedding)