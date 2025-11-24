from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from mangum import Mangum
from chatbot_llama import respuesta 

app = FastAPI()

# Configuración de CORS (Crucial para que funcione desde el navegador/Freshdesk)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Modelo de datos para la entrada (Request Body)
class Entrada(BaseModel):
    mensaje: str
    # El historial es opcional al principio (puede ser una lista vacía)
    history: Optional[List[Dict[str, Any]]] = []

@app.post("/chat")
def chat(entrada: Entrada):
    try:
        # Llamamos a tu función de chatbot pasando mensaje Y historial
        bot_reply, updated_history = respuesta(entrada.mensaje, entrada.history)
        
        # Devolvemos la respuesta Y el historial actualizado al cliente
        return {
            "respuesta": bot_reply,
            "history": updated_history
        }
    except Exception as e:
        # Manejo básico de errores
        print(f"Error en el endpoint /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
def health_check():
    #Endpoint simple para verificar que la Lambda está viva
    return {"status": "ok", "message": "Chatbot API is running"}

# Para ejecutarlo en AWS Lambda
# handler = Mangum(app)

# Para ejecutarlo localmente con uvicorn:
# uvicorn index:app --reload