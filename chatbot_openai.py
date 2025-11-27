import random
import json
import pickle
import numpy as np
import os
import nltk
from dotenv import load_dotenv
from operator import itemgetter # Para lógica conversacional
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# --- IMPORTS RAG (Híbrido: Embeddings Locales + LLM OpenAI) ---
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


# --- Carga de API OpenAI----
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    print("FATAL: OPENAI_API_KEY no encontrada.")
    exit()

# Descarga de NLTK (si es local)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

#Importamos los archivos generados en el código anterior
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# --- UMBRAL DE CONFIANZA ---
# 0.7 para que sea más estricto con el clasificador.
# 0.3 para que el RAG se active menos.
ERROR_THRESHOLD = 0.8

# --- CONFIGURACIÓN DE RUTAS DE INDICES (FAISS + BM25) ---
PATHS = {
    "firmagob": {
        "faiss": "faiss_index_firmagob",
        "bm25": "bm25_firmagob.pkl"
    },
    "docdigital": {
        "faiss": "faiss_index_docdigital",
        "bm25": "bm25_docdigital.pkl"
    }
}

# --- FUNCIONES CLASIFICADOR ---
#Pasamos las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    # print(bag)
    return np.array(bag)

#Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    
    # Predice la categoría, pero solo si la confianza supera el umbral.
    # Si no, devuelve un tag especial de fallback.
    
    bow = bag_of_words(sentence)
    # verbose=0 para que Keras no imprima "1/1 [==============================]"
    res = model.predict(np.array([bow]), verbose=0)[0]

    # Encuentra la probabilidad más alta
    best_prob = np.max(res)
    max_index = np.where(res == best_prob)[0][0]
    category = classes[max_index]
    
    print(f"[Debug] Mejor suposición: {category} (Conf: {best_prob:.2f})")
    
    # Devuelve AMBOS, el tag y la probabilidad
    return category, best_prob

#Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
            result = random.choice(i['responses'])
            break
    return result

# --- LOGICA RAG ---

def create_rag_chain(doc_key: str, embeddings_model):
    """
    Carga, procesa un PDF y devuelve una cadena RAG lista para invocar.
    Usa Embeddings locales y LLM de OpenAI.
    """
    print(f"[RAG Setup] Inicializando sistema híbrido para '{doc_key}'...")

    paths_config = PATHS.get(doc_key)
    if not paths_config:
        raise ValueError(f"FATAL: La clave '{doc_key}' no existe en la configuración PATHS.")
    
    # 1. Cargar Indice Denso (FAISS)
    try:
        vectorstore = FAISS.load_local(
            paths_config["faiss"],
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        print(f"   -> FAISS cargado desde {paths_config['faiss']}")
    except Exception as e:
        print(f"FATAL: Error cargando el índice FAISS: {e}")
        raise e

    # 2. Cargar Indice Disperso (BM25)
    try:
        with open(paths_config["bm25"], "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = 4 # Aseguramos que traiga top 4 también
        print(f"   -> BM25 cargado desde {paths_config['bm25']}")
    except Exception as e:
        print(f"FATAL: Error cargando BM25 para {doc_key}: {e}")
        raise e

    # 3. Fusión de Cerebros (Ensemble)
    # 50% peso semántico, 50% peso literal
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    
    # 2. LLM
    # gpt-4o-mini es rápido y eficiente
    llm = ChatOpenAI(model="gpt-4o-mini") # Para OpenAI

    # 3. Prompt Template
    template = """
    Eres un asistente experto de Secretaría de Gobierno Digital. Responde la pregunta del usuario basándote en el siguiente historial y contexto. Si la información no se encuentra en el contexto, di "No tengo información al respecto, pero te recomiendo que levantes un ticket.".

    Historial del chat:
    {chat_history}

    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Cadena (Chain) RAG
    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'desconocido')
            page = doc.metadata.get('page', '?')
            content = doc.page_content.replace("\n", " ") # Limpiamos saltos de línea
            formatted.append(f"--- Fuente: {source} (Pág {page}) ---\n{content}")
        return "\n\n".join(formatted)
    
    def format_history(history_list):   # lógica de historial
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_list])

    # Cadena actualizada, con nuevos dict para "question" y "chat_history"
    rag_chain = (
        {
            "context": itemgetter("question") | ensemble_retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history") | RunnableLambda(format_history)
         
         }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"[RAG Setup] Pipeline RAG para '{doc_key}' listo.") # Mensaje de log actualizado
    return rag_chain

# --- FIN LÓGICA RAG ---

# --- INICIALIZACIÓN ---
# (Envuelto en un 'if __name__ == "__main__":' para que Lambda pueda importarlo)

print("Cargando modelo Keras...")
# ... (Cargas del modelo Keras ya hechas arriba) ...
print("Modelo Keras cargado.")

print("[RAG Setup] Cargando modelo de embeddings locales (HuggingFace)...")
embeddings_local = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Cargamos los índices FAISS
# Cargamos los RAGs usando las claves del diccionario PATHS
rag_chain_firmagob = create_rag_chain("firmagob", embeddings_local)
rag_chain_docdigital = create_rag_chain("docdigital", embeddings_local)
print("Todos los pipelines RAG (Híbridos OpenAI) están listos.")


# --- ROUTER PRINCIPAL ---

KEYWORDS_DOCDIGITAL = [
    "docdigital", "visar", "visación", "visador", "oficina de partes", 
    "cadena de responsabilidad", "numerar", "distribuir", "memorando", 
    "circular", "oficio"
]

KEYWORDS_FIRMAGOB = [
    "firmagob", "ministro de fe", "otp", "google authenticator", "token", 
    "desatendida", "propósito general", "revocar", "operador", 
    "certificado", "vencimiento"
]

def respuesta(message: str, history_list: list):
    # Actua como router, califica la intención y decide si usar RAG o respuestas estáticas. Modificado para lógica historial

    message_lower = message.lower()
    ints_tag, prob = predict_class(message)

    if prob > ERROR_THRESHOLD:
        print(f"[Router] -> Confianza alta ({prob:.2f}). Enviando a get_response (JSON)")
        res = get_response(ints_tag, intents)
    
    else:
        # Detección booleana rápida
        is_doc_term = any(k in message_lower for k in KEYWORDS_DOCDIGITAL)
        is_firma_term = any(k in message_lower for k in KEYWORDS_FIRMAGOB)

        # Regla A: Vocabulario explícito de DocDigital
        if is_doc_term:
            print(f"[Router] -> Keyword DocDigital detectada. Forzando RAG-DocDigital.")
            res = rag_chain_docdigital.invoke({
                "question": message,
                "chat_history": history_list
            })

        # Regla B: Vocabulario explícito de FirmaGob
        elif is_firma_term:
            print(f"[Router] -> Keyword FirmaGob detectada. Forzando RAG-FirmaGob.")
            res = rag_chain_firmagob.invoke({
                "question": message,
                "chat_history": history_list
            })

        # Regla C: Si no hay keywords claras, volvemos a tu lógica original (Fallback al clasificador)
        elif ints_tag.startswith("docdigital_"):
            print(f"[Router] -> Sin keywords. Clasificador sugiere DocDigital ({prob:.2f}).")
            res = rag_chain_docdigital.invoke({
                "question": message,
                "chat_history": history_list
            })

        # Regla D: Fallback final a FirmaGob (tu default original)
        else:
            print(f"[Router] -> Sin keywords. Clasificador sugiere FirmaGob/Ambiguo ({prob:.2f}).")
            res = rag_chain_firmagob.invoke({
                "question": message,
                "chat_history": history_list
            })

    # Actualización de historial
    history_list.append({"role": "user", "content": message})
    history_list.append({"role": "assistant", "content": res})
    
    if len(history_list) > 10:
        history_list = history_list[-10:]

    return res, history_list

# --- Bucle para pruebas sin uso de web service ---
if __name__ == "__main__":
    print("\n--- Bot listo (Modo Híbrido: Embeddings Locales + OpenAI LLM) ---")
    chat_history = []
    while True:
        message = input("\nTú: ")
        if message.lower() == 'salir':
            break
        bot_response, chat_history = respuesta(message, chat_history)
        print(f"Bot: {bot_response}")