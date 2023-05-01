# Utilisez l'image Python 3.9.5 comme image de base
FROM python:3.9.5

# Définissez le répertoire de travail
WORKDIR /code

# Copiez les dépendances Python
COPY ./requirements.txt /code/requirements.txt

# Installez les dépendances Python
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

# Copier l'application 
COPY ./app /code/app

# Installez curl
RUN apt update && apt install curl -y

# Installez Deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

EXPOSE 8000

# Exécutez le serveur avec Gunicorn, en utilisant UvicornWorker comme type de worker
CMD ["gunicorn", "app.main:app", "--bind", "0.0.0.0:8000", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "1", "--timeout", "1800", "--keep-alive", "1800"]

# CMD gunicorn app.main:app --bind 0.0.0.0:${PORT:-8000} --worker-class uvicorn.workers.UvicornWorker

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# CMD gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
