FROM public.ecr.aws/lambda/python:3.12

# Dependencias
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Optimización NLTK
RUN mkdir -p ${LAMBDA_TASK_ROOT}/nltk_data
RUN python -m nltk.downloader punkt punkt_tab wordnet -d ${LAMBDA_TASK_ROOT}/nltk_data
RUN chmod -R 777 ${LAMBDA_TASK_ROOT}/nltk_data
ENV NLTK_DATA=${LAMBDA_TASK_ROOT}/nltk_data

# Optimización sentence-transformers
RUN mkdir -p ${LAMBDA_TASK_ROOT}/model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='${LAMBDA_TASK_ROOT}/model_cache')"
RUN chmod -R 777 ${LAMBDA_TASK_ROOT}/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=${LAMBDA_TASK_ROOT}/model_cache

# Copia de codigo de la app
COPY chatbot_openai.py ${LAMBDA_TASK_ROOT}
COPY index.py ${LAMBDA_TASK_ROOT}
COPY intents.json ${LAMBDA_TASK_ROOT}
COPY words.pkl ${LAMBDA_TASK_ROOT}
COPY classes.pkl ${LAMBDA_TASK_ROOT}
COPY chatbot_model.h5 ${LAMBDA_TASK_ROOT}

# Copia indice FAISS
COPY faiss_index_firmagob ${LAMBDA_TASK_ROOT}/faiss_index_firmagob
COPY faiss_index_docdigital ${LAMBDA_TASK_ROOT}/faiss_index_docdigital

CMD [ "index.handler" ]