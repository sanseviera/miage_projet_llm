# Utiliser une image Python officielle
FROM python:3.11-slim

# Créer un utilisateur non-root
RUN useradd -ms /bin/bash appuser

# Passer à l'utilisateur non-root
USER appuser

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY --chown=appuser:appuser . /app

# Installer les dépendances (exécutées par l'utilisateur non-root)
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application s'exécute
EXPOSE 5000

# Commande de démarrage de l'application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]