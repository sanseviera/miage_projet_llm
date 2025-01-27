# Utiliser une image Python officielle
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application s'exécute
EXPOSE 5000

# Commande de démarrage de l'application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]