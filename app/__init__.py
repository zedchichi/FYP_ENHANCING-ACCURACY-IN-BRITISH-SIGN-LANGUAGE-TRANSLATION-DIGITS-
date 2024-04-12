from flask import Flask

def create_app():
    app = Flask(__name__)


    # from app.views import views_blueprint
    # app.register_blueprint(views_blueprint)

    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.config['CAPTURED_FOLDER'] = 'capture/'
    app.config['CLASSIFIED_FOLDER'] = 'classified/'
    app.config['VGG_CLASSIFIED'] = 'classified/vgg/'
    app.config['MOBILENET_CLASSIFIED'] = 'classified/mobilenet/'
    app.config['mobilenet_path'] = r'C:\Users\anazi\FYP\app\BSL_MobileNet_HD_build_mobilenet_hyper4.h5'
    app.config['vggmodel_path'] = r'C:\Users\anazi\FYP\app\BSL_VGG16_Cus_FT_HD_Best_Model3.h5'

    with app.app_context():
        from app.views import views_blueprint, initialize_models
        initialize_models()
        app.register_blueprint(views_blueprint)

    return app
