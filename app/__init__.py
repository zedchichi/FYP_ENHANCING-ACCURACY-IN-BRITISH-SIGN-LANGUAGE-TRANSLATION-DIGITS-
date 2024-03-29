from flask import Flask
from .views import views_blueprint

def create_app():
    app = Flask(__name__)
    # Configure your app and register blueprint
    app.register_blueprint(views_blueprint)
    
    return app