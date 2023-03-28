FROM tiangolo/uvicorn-gunicorn:python3.9-slim
RUN mkdir /dogBreedRecognition
COPY requirements.txt /dogBreedRecognition
COPY . /dogBreedRecognition
WORKDIR /dogBreedRecognition
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "FastAPI:app", "--host", "0.0.0.0", "--port", "8000"]