from scripts.rag_pipeline import RAGPipeline
from config import DATA_PATH, EMBEDDING_MODEL,TOP_K

def main():
    rag = RAGPipeline(DATA_PATH,EMBEDDING_MODEL)

    print("Building RAG Pipeline")
    rag.build()

    while True:
        query = input("Enter your Query(or EXit):")
        if query.lower() == "exit":
            return 
    
        answer, docs = rag.query(query, TOP_K)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
