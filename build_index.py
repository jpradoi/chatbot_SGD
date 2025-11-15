import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define los documentos y dónde guardar sus índices
document_map = {
    "firmagob": {
        "pdf_path": "firmagob.pdf",
        "index_path": "faiss_index_firmagob"
    },
    "docdigital": {
        "pdf_path": "docdigital.pdf",
        "index_path": "faiss_index_docdigital"
    }
}

def build_indices():
    print("Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
    embeddings_local = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    total_start_time = time.time()

    for key, paths in document_map.items():
        print(f"\n--- Procesando: {key} ---")
        pdf_path = paths["pdf_path"]
        index_path = paths["index_path"]
        
        start_time = time.time()
        
        print(f"Cargando {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        print(f"Documento cargado. {len(docs)} páginas.")

        print("Dividiendo documentos...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"Documento dividido en {len(splits)} trozos (chunks).")

        print("Construyendo índice FAISS... (Esto tomará un momento)")
        vectorstore = FAISS.from_documents(
            documents=splits, 
            embedding=embeddings_local
        )

        print(f"Índice construido. Guardando en carpeta '{index_path}'...")
        vectorstore.save_local(index_path)
        
        end_time = time.time()
        print(f"Índice '{key}' guardado. Tiempo: {end_time - start_time:.2f} seg.")

    total_end_time = time.time()
    print(f"\nProceso completado. Tiempo total: {total_end_time - total_start_time:.2f} seg.")

if __name__ == "__main__":
    build_indices()