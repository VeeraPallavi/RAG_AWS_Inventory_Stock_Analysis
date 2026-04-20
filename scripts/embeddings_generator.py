from sentence_transformers import SentenceTransformer

class EmbeddingsGenerator:
    def __init__(self,model):
        self.model = SentenceTransformer(model)
        
    def generate(self,texts):
        return self.model.encode(texts,show_progress_bar = True)

