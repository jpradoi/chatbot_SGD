import time
import re
import pickle
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Mapeo de documentos
document_map = {
    "firmagob": {
        "pdf_path": "firmagob.pdf",
        "index_path": "faiss_index_firmagob",
        "bm25_path": "bm25_firmagob.pkl" # Nuevo: Guardaremos el índice de keywords
    },
    "docdigital": {
        "pdf_path": "docdigital.pdf",
        "index_path": "faiss_index_docdigital",
        "bm25_path": "bm25_docdigital.pkl"
    }
}

def clean_text(text):
    """
    Limpieza básica para eliminar ruido común en PDFs gubernamentales.
    """
    # Eliminar números de página sueltos (ej: "--- PAGE 1 ---")
    text = re.sub(r'--- PAGE \d+ ---', '', text)
    # Eliminar saltos de línea excesivos que rompen oraciones
    text = re.sub(r'\n(?!\n)', ' ', text) 
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_indices():
    print("Cargando modelo de embeddings (paraphrase-multilingual-MiniLM-L12-v2)...")
    embeddings_local = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    total_start_time = time.time()

    for key, paths in document_map.items():
        print(f"\n--- Procesando: {key} ---")
        pdf_path = paths["pdf_path"]
        index_path = paths["index_path"]
        bm25_path = paths["bm25_path"]
        
        start_time = time.time()
        
        # 1. Carga
        print(f"Cargando {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # 2. Limpieza (Paso Nuevo)
        print("Limpiando contenido...")
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            # Agregamos metadatos útiles para el router si los necesitas luego
            doc.metadata["source_type"] = key 

        print(f"Documento cargado y limpiado. {len(docs)} páginas.")

        # 3. Splitting (Ajustado: Chunks más pequeños para precisión)
        print("Dividiendo documentos...")
        # Bajamos a 500 para que datos como '3 años' no se pierdan en el ruido
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100, 
            separators=["\n\n", ". ", " ", ""], 
            length_function=len
        )
        splits = text_splitter.split_documents(docs)
        print(f"Documento dividido en {len(splits)} trozos (chunks).")

        # 4. Construcción FAISS (Denso)
        print("Construyendo índice Vectorial (FAISS)...")
        vectorstore = FAISS.from_documents(
            documents=splits, 
            embedding=embeddings_local
        )
        vectorstore.save_local(index_path)
        print(f"-> Índice FAISS guardado en '{index_path}'")

        # 5. Construcción BM25 (Sparse - Palabras Clave)
        # Esto es lo que te permitirá encontrar "vigencia" aunque el vector falle
        print("Construyendo índice de Keywords (BM25)...")
        bm25_retriever = BM25Retriever.from_documents(splits)
        # BM25 no tiene método .save_local nativo, usamos pickle
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print(f"-> Índice BM25 guardado en '{bm25_path}'")
        
        end_time = time.time()
        print(f"Proceso '{key}' finalizado. Tiempo: {end_time - start_time:.2f} seg.")

    total_end_time = time.time()
    print(f"\nProceso completado. Tiempo total: {total_end_time - total_start_time:.2f} seg.")

if __name__ == "__main__":
    build_indices()