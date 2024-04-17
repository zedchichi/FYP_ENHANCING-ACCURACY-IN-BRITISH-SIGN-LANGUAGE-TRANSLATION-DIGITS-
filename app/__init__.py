from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.config['CAPTURED_FOLDER'] = 'capture/'
    app.config['CLASSIFIED_FOLDER'] = 'classified/'
    app.config['VGG_CLASSIFIED'] = 'classified/vgg/'
    app.config['MOBILENET_CLASSIFIED'] = 'classified/mobilenet/'
    app.config['VGG_INCORRECT'] = 'classified/vgg/incorrect'
    app.config['MOBILENET_INCORRECT'] = 'classified/mobilenet/incorrect'
    app.config['mobilenet_path'] = r'C:\Users\anazi\FYP\app\BSL_MobileNet_HD_build_mobilenet_hyper3.h5'
    app.config['vggmodel_path'] = r'C:\Users\anazi\FYP\app\BSL_VGG16_Cus_FT_HD_Best_Model5.h5'
    app.config['mobilenet_save_path'] = '/app/updated_mobilenet_model.h5'
    app.config['vgg_save_path'] = '/app/updated_vgg_model.h5'


    with app.app_context():
        from app.views import views_blueprint, initialize_models
        app.register_blueprint(views_blueprint) #register blueprint
        initialize_models()


    return app