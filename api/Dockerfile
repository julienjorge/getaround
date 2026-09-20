FROM continuumio/miniconda3:25.3.1-1

RUN apt-get update && apt-get install -y nano curl

# Spécifique à Hugging Face : utilisateur d'id 1000, sinon erreur "Permission denied"
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . $HOME/app

# Port imposé par Hugging Face pour les applications web
EXPOSE 7860
CMD fastapi run app.py --port 7860
