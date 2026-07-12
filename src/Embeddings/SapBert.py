"""
File này chưa class embedding cho sapbert.
Sẽ được sử dụng để embed text vào vectorstore.
Ngoài ra, còn được dùng để embed test data trước khi truy vấn.
"""
from langchain_core.embeddings import Embeddings
import torch
import torch.nn.functional as F


class SapBERTEmbeddings(Embeddings):
    """
    Adapter giúp LangChain/Chroma sử dụng SapBERT.

    - embed_documents(): embedding danh sách RxNorm terms.
    - embed_query(): embedding câu truy vấn.
    """
    def __init__(
        self,
        tokenizer,
        model,
        device,
        batch_size=128,
        max_length=64
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def embedding(self, texts: list[str]) -> list[list[float]]:
        """
        Chuyển danh sách text thành normalized SapBERT embeddings.
        """

        all_embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

                # Lấy vector của token [CLS]
                batch_embeddings = outputs.last_hidden_state[:, 0, :]

                # L2 normalization
                batch_embeddings = F.normalize(
                    batch_embeddings,
                    p=2,
                    dim=1
                )

            # Chroma cần list Python, không nhận GPU tensor
            batch_embeddings = (
                batch_embeddings
                .cpu()
                .numpy()
                .tolist()
            )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embedding(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedding([text])[0]
    
    
        
