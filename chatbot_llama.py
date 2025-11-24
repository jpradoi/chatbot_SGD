import random
import json
import pickle
import numpy as np
from dotenv import load_dotenv
from operator import itemgetter # Para lógica conversacional

import nltk
from nltk.stem import WordNetLemmatizer

# --- IMPORTS RAG ---
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

from keras.models import load_model

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

def create_rag_chain(vectorstore_path: str, embeddings_model):
    """
    Carga, procesa un PDF y devuelve una cadena RAG lista para invocar.
    Usa OpenAI para embeddings y generación.
    """
    print(f"[RAG Setup] Cargando indice FAISS desde '{vectorstore_path}'...")
    # 1. Carga índice creado con build_index
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings_model,
        allow_dangerous_deserialization=True
    )


    retriever = vectorstore.as_retriever()
    
    # 2. LLM
    llm = ChatOllama(model="llama3")

    # 3. Prompt Template
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

    # 4. Cadena (Chain) RAG
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    def format_history(history_list):   # lógica de historial
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_list])

    # Cadena actualizada, con nuevos dict para "question" y "chat_history"
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
    
    print("[RAG Setup] Pipeline RAG listo.")
    return rag_chain

# --- FIN LÓGICA RAG ---

# --- INICIALIZACIÓN ---

print("Cargando modelo Keras y archivos pickle...")
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
print("Modelo Keras cargado.")

print("[RAG Setup] Cargando modelo de embeddings locales (HuggingFace)...")
# Cargamos el modelo de embeddings UNA SOLA VEZ
embeddings_local = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

rag_chain_firmagob = create_rag_chain("faiss_index_firmagob", embeddings_local)
rag_chain_docdigital = create_rag_chain("faiss_index_docdigital", embeddings_local)
print("Todos los pipelines RAG están listos.")


# --- ROUTER PRINCIPAL ---

def respuesta(message: str, history_list: list):
    # Actua como router, califica la intención y decide si usar RAG o respuestas estáticas. Modificado para lógica historial
    
    # Clasifica intención
    ints_tag, prob = predict_class(message)

    # Caso 1: Alta confianza -> uso de clasificador
    if prob > ERROR_THRESHOLD:
        # Confianza alta -> respuesta estática
        print(f"[Router] -> Confianza alta ({prob:.2f}. Intención '{ints_tag}'. Enviando a get_response (JSON)")
        res = get_response(ints_tag, intents)

    # Caso 2: Baja confianza, tag sugiere FirmaGob -> Uso de RAG Firmagob
    elif ints_tag.startswith("firmagob_"):
        print(f"[Router] -> Confianza baja ({prob:.2f}). Tag '{ints_tag}'. Enviando a RAG-FirmaGob...")
        res = rag_chain_firmagob.invoke({
            "question": message,
            "chat_history": history_list
        })

    # Caso 3: Baja confianza, tag sugiere DocDigital -> Uso de RAG DocDigital
    elif ints_tag.startswith("docdigital_"):
        print(f"[Router] -> Confianza baja ({prob:.2f}). Tag '{ints_tag}'. Enviando a RAG-DocDigital...")
        res = rag_chain_docdigital.invoke({
            "question": message,
            "chat_history": history_list
        })

    # Caso 4: Baja confianza y tag ambiguo -> Fallback
    else:
        print(f"[Router] -> Confianza baja ({prob:.2f}). Tag ambiguo '{ints_tag}'. Usando RAG-FirmaGob por defecto.")
        # Decidimos enviar a un RAG por defecto (ej. FirmaGob) como último recurso.
        res = rag_chain_firmagob.invoke({
            "question": message,
            "chat_history": history_list
        })

    # --- Actualización de memoria ---
    # 1. Añade el mensaje actual del usuario
    history_list.append({"role": "user", "content": message})
    # 2. añade respuesta del chat
    history_list.append({"role": "assistant", "content": res})

    # 3. Poda historial para mantener últimos 10 mensajes (5 turnos)
    if len(history_list) > 10:
        history_list = history_list[-10:]

    return res, history_list
'''
# --- Bucle para pruebas: comentar si pasa a AWS ---
chat_history = []
while True:
    message = input("\nTú: ")
    if message.lower() == 'salir':
        break

    bot_response, chat_history = respuesta(message, chat_history)
    print(f"Bot: {bot_response}")
'''