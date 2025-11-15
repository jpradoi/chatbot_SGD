FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt punkt_tab wordnet -d /var/lang/nltk_data

COPY chatbot_openai.py ${LAMBDA_TASK_ROOT}
COPY index.py ${LAMBDA_TASK_ROOT}
COPY intents.json ${LAMBDA_TASK_ROOT}
COPY words.pkl ${LAMBDA_TASK_ROOT}
COPY classes.pkl ${LAMBDA_TASK_ROOT}
COPY chatbot_model.h5 ${LAMBDA_TASK_ROOT}

COPY faiss_index_firmagob ${LAMBDA_TASK_ROOT}/faiss_index_firmagob
COPY faiss_index_docdigital ${LAMBDA_TASK_ROOT}/faiss_index_docdigital

ENV NLTK_DATA=/var/lang/nltk_data

CMD [ "index.handler" ]