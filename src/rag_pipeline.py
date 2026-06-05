import numpy as np 
from scripts.data_preprocessing import DataPreprocessing
from scripts.embeddings_generator import EmbeddingsGenerator
from scripts.llm_generator import LLMGenerator
from scripts.vector_store import VectorStore

class RAGPipeline:
    def __init__(self,data_path,model_name):
        self.processor = DataPreprocessing(data_path)
        self.embedder = EmbeddingsGenerator(model_name)
        self.llm_generator = LLMGenerator()
        self.vector_store = VectorStore()

    def row_to_text(self, row):
        return f"""
        Warehouse: {row['warehouse']}
        Region: {row['region']}
        Product: {row['product']}
        Quantity: {row['order_qty']}
        Status: {row['status']}
        Delivery Time: {row['delivery_time_days']} days
        """
    def prepare_documents(self):
        df = self.processor.load_data()
        df = self.processor.clean_data()
        self.documents = df.apply(self.row_to_text,axis = 1).tolist()        
        return self.documents
    
    def build(self):
        documents = self.prepare_documents()
        embeddings = self.embedder.generate(documents)
        self.vector_store.build_index(embeddings, documents)

    def query(self,query,top_k):

        query_vector = self.embedder.model.encode(query)
        query_vector = np.array([query_vector]).astype('float32')
        retrieved_docs = self.vector_store.search(query_vector, top_k)

        context = "\n".join(retrieved_docs)
        answer = self.llm_generator.generate(query, context)

        return answer, retrieved_docs