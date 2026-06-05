import  numpy as np
import faiss

class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = None
    
    def build_index(self,embeddings,documents):
        embeddings = np.array(embeddings).astype('float32')

        dimensions = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dimensions, 32)

        self.index.add(embeddings)
        self.documents = documents

    def search(self, query_vector,top_k):
        distances, indicies = self.index.search(query_vector, top_k)
        return [self.documents[i] for i in indicies[0]]
