from flask import Flask

def create_app():
    app = Flask(__name__, static_folder='static')

    from app.views import views_blueprint
    app.register_blueprint(views_blueprint)

    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.config['CAPTURED_FOLDER'] = 'capture/'
    app.config['CLASSIFIED_FOLDER'] = 'classified'

    return app
