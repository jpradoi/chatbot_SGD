import random
import json
import pickle
import numpy as np
import os
import nltk
from dotenv import load_dotenv
from operator import itemgetter
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# --- IMPORTS RAG (Híbrido: Embeddings Locales + LLM Gemini) ---
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
# IMPORTANTE: Usamos Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings


# --- Carga de API Key de Google ----
load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    print("FATAL: GOOGLE_API_KEY no encontrada. Asegúrate de tenerla en un archivo .env")
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

# Carga de archivos del clasificador
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

ERROR_THRESHOLD = 0.8

# --- FUNCIONES CLASIFICADOR (Idénticas) ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    best_prob = np.max(res)
    max_index = np.where(res == best_prob)[0][0]
    category = classes[max_index]
    print(f"[Debug] Mejor suposición: {category} (Conf: {best_prob:.2f})")
    return category, best_prob

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
            result = random.choice(i['responses'])
            break
    return result

# --- LOGICA RAG (Gemini) ---

def create_rag_chain(vectorstore_path: str, embeddings_model):
    print(f"[RAG Setup] Cargando indice FAISS desde '{vectorstore_path}'...")
    try:
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"FATAL: Error cargando el índice FAISS: {e}")
        raise e

    retriever = vectorstore.as_retriever()
    
    # --- LLM ---
    # Usamos Gemini 2.0 flash. Temperature=0 para respuestas más fácticas.
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    template = """
    Eres un asistente experto de Secretaría de Gobierno Digital. Responde la pregunta del usuario basándote única y exclusivamente en el siguiente historial y contexto. Si la información no se encuentra en el contexto, di "No tengo información sobre eso".

    Historial del chat:
    {chat_history}

    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def format_history(history_list):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_list])

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history") | RunnableLambda(format_history)
         }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"[RAG Setup] Pipeline RAG (Gemini) para '{vectorstore_path}' listo.")
    return rag_chain

# --- INICIALIZACIÓN ---

print("Cargando modelo Keras...")
# ... (Cargas del modelo Keras ya hechas arriba) ...
print("Modelo Keras cargado.")

print("[RAG Setup] Cargando modelo de embeddings locales (HuggingFace)...")
# Usamos el MISMO modelo de embeddings que usaste para crear el índice
embeddings_local = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Cargamos los índices FAISS (que ya tienes creados)
rag_chain_firmagob = create_rag_chain("faiss_index_firmagob", embeddings_local)
rag_chain_docdigital = create_rag_chain("faiss_index_docdigital", embeddings_local)
print("Todos los pipelines RAG (Híbridos Gemini) están listos.")

# --- ROUTER PRINCIPAL ---
# (Esta función es IDÉNTICA a la de OpenAI/Llama)
def respuesta(message: str, history_list: list):
    ints_tag, prob = predict_class(message)

    if prob > ERROR_THRESHOLD:
        print(f"[Router] -> Confianza alta ({prob:.2f}). Enviando a get_response (JSON)")
        res = get_response(ints_tag, intents)
    elif ints_tag.startswith("firmagob_"):
        print(f"[Router] -> Confianza baja ({prob:.2f}). Tag '{ints_tag}'. Enviando a RAG-FirmaGob (Gemini)...")
        res = rag_chain_firmagob.invoke({
            "question": message,
            "chat_history": history_list
        })
    elif ints_tag.startswith("docdigital_"):
        print(f"[Router] -> Confianza baja ({prob:.2f}). Tag '{ints_tag}'. Enviando a RAG-DocDigital (Gemini)...")
        res = rag_chain_docdigital.invoke({
            "question": message,
            "chat_history": history_list
        })
    else:
        print(f"[Router] -> Confianza baja ({prob:.2f}). Tag ambiguo. Usando RAG-FirmaGob por defecto.")
        res = rag_chain_firmagob.invoke({
            "question": message,
            "chat_history": history_list
        })

    history_list.append({"role": "user", "content": message})
    history_list.append({"role": "assistant", "content": res})
    
    if len(history_list) > 10:
        history_list = history_list[-10:]

    return res, history_list

# --- Bucle para pruebas ---
if __name__ == "__main__":
    print("\n--- Bot listo (Modo Híbrido: Embeddings Locales + Gemini) ---")
    chat_history = []
    while True:
        message = input("\nTú: ")
        if message.lower() == 'salir':
            break
        bot_response, chat_history = respuesta(message, chat_history)
        print(f"Bot: {bot_response}")