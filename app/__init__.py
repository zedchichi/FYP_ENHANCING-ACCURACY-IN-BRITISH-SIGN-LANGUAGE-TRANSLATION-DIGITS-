from flask import Flask
from .views import app as views_blueprint

def create_app():
    app = Flask(__name__)
    # Configure your app and register blueprints here
    app.register_blueprint(views_blueprint)
    
    return app