from flask import Flask

def create_app():
    app = Flask(__name__)
    # Configure your app and register blueprints here

    return app