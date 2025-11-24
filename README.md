# Chatbot de Mesa de Ayuda SGD
Este repositorio contiene el código fuente y las instrucciones de despliegue para el Chatbot de la Mesa de Ayuda, diseñado para ejecutarse en AWS Lambda utilizando contenedores (Docker).
El chatbot utiliza una arquitectura híbrida:
1. **Clasificador de Intenciones (Keras/TensorFlow):** Para responder preguntas frecuentes de forma rápida y sin costo.
2. **Sistema RAG (Retrieval-Augmented Generation):** Para responder preguntas complejas consultando documentación oficial, utilizando OpenAI (gpt-4o-mini) como motor de respuesta y HuggingFace para embeddings locales.
## Requisitos previos
Para desplegar este proyecto, se requiere:
- Acceso a AWS (Lambda, ECR, API Gateway, CloudWatch).
- Docker instalado localmente para construir la imagen.
- Una API Key de OpenAI válida.
## Estructura del proyecto
- `chatbot_openai.py`: Lógica principal del chatbot (Router, Clasificador y RAG).
- `index.py`: Servidor FastAPI + adaptador Mangum para recibir peticiones HTTPS en Lambda.
- `build_index_openai.py`: Script de utilidad (offline) para generar los índices vectoriales FAISS.
- `dockerfile`: Instrucciones para construir la imagen de contenedor para Lambda.
- `requirements.txt`: Dependencias de Python.
- `faiss_index_*/`: Carpetas que contienen la base de conocimiento vectorial pre-construida.
- `chatbot_model.h5`, `words.pkl`, `classes.pkl`: Archivos del modelo clasificador entrenado.
## Archivos extras
- `chatbot_gemini.py`: Versión Gemini de lógica principal del chatbot (Router, Clasificador y RAG).
- `chatbot_llama.py`: Versión Llama de lógica principal del chatbot (Router, Clasificador y RAG).
- `index_gemini.py`: Versión Gemini de servidor FastAPI objetivo de uso con uvicorn
- `index_llama.py`: Versión Llama de servidor FastAPI objetivo de uso con uvicorn
## Instrucciones de Despliegue (Para Infraestructura)
1. **Variables de Entorno (Configuración en Lambda)**
Es obligatorio configurar la siguiente variable de entorno en la consola de AWS Lambda:

|      Variable      |               Descripción               |      Ejemplo       |
|:-------------------|:----------------------------------------|:-------------------|
| `OPENAI_API_KEY`   | Clave API de OpenAI para el motor RAG   | sk-proj-12345...   |

2. Construcción y Subida de la Imagen (Docker)
Desde la raíz del proyecto:
```
# 1. Autenticarse en AWS ECR (reemplazar con región y ID de cuenta reales)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# 2. Construir la imagen Docker
docker build -t chatbot-sgd-lambda .

# 3. Etiquetar la imagen para ECR
docker tag chatbot-sgd-lambda:latest [123456789012.dkr.ecr.us-east-1.amazonaws.com/chatbot-sgd-lambda:latest](https://123456789012.dkr.ecr.us-east-1.amazonaws.com/chatbot-sgd-lambda:latest)

# 4. Subir la imagen al repositorio ECR
docker push [123456789012.dkr.ecr.us-east-1.amazonaws.com/chatbot-sgd-lambda:latest](https://123456789012.dkr.ecr.us-east-1.amazonaws.com/chatbot-sgd-lambda:latest)
```

3. Configuración de la Función Lambda
- **Origen:** Imagen de contenedor (URI del ECR subido).
- **Arquitectura:** x86_64.
- **Memoria:** Mínimo 1024 MB (Recomendado: 2048 MB).
- **Timeout:** 30 segundos.
- **Política IAM:** Permisos básicos de ejecución de Lambda y acceso a CloudWatch Logs.

## Uso de la API

Una vez desplegada la Lambda y conectada a un API Gateway, el endpoint principal es:
`POST /chat`
Cuerpo de la Petición (JSON):
```
{
  "mensaje": "¿Qué es FirmaGob?",
  "history": [
    {"role": "user", "content": "Hola"},
    {"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte?"}
  ]
}
```
Respuesta Exitosa (JSON):
```
{
  "respuesta": "FirmaGob es la plataforma de firma electrónica del Estado...",
  "history": [
    {"role": "user", "content": "Hola"},
    {"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte?"},
    {"role": "user", "content": "¿Qué es FirmaGob?"},
    {"role": "assistant", "content": "FirmaGob es la plataforma..."}
  ]
}
```
## Notas de Mantenimiento
- **Actualización de Documentación:** Si los manuales PDF cambian, se debe ejecutar python build_index.py localmente para regenerar las carpetas faiss_index y volver a construir/desplegar la imagen Docker.
- **Actualización de Intenciones:** Si se edita intents.json, se debe ejecutar python training.py localmente para regenerar el modelo .h5 y volver a construir/desplegar la imagen.