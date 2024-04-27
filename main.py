from apscheduler.schedulers.background import BackgroundScheduler
from app import create_app
import atexit
from flask_talisman import Talisman
import logging
from logging.handlers import RotatingFileHandler


def configure_logging():
    logger = logging.getLogger('BSLModelRetraining')
    logger.setLevel(logging.INFO)  # Set to DEBUG for more verbose output

    # file handler which logs even debug messages
    handler = RotatingFileHandler('retraining.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)

    # Creating logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Adding handler to the logger
    logger.addHandler(handler)
    return logger


logger = configure_logging()

app = create_app()

#Initialising Talisman with strict security policies
Talisman(app, content_security_policy=None)
# Talisman(app,
#          content_security_policy={
#              'default-src': [
#                  '\'self\'',
#                  'https://cdn.jsdelivr.net',
#                  'https://cdn.jsdelivr.net/npm/intro.js@7.2.0/minified/introjs.min.css',
#                  'https://cdn.jsdelivr.net/npm/bootstrap@4.4.1/dist/css/bootstrap.min.css',
#                  'https://cdn.jsdelivr.net/npm/intro.js@7.2.0/intro.min.js',
#                  'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js',
#              ],
#              'style-src': [
#                  '\'self\'',
#                  '\'unsafe-inline\'',
#                  'https://cdn.jsdelivr.net',
#              ],
#              'script-src': [
#                  '\'self\'',
#                  'https://cdn.jsdelivr.net',
#                  '\'unsafe-inline\'',  #inline scripts
#                  '\'unsafe-eval\'',
#              ]
#
#          },
#          content_security_policy_nonce_in=['script-src'])


def schedule_retraining():
    global logger
    try:
        logger.info("Starting scheduled model retraining.")
        from app.views import retrain_models
        retrain_models()
        logger.info("Model retraining completed successfully.")
    except Exception as E:
        logger.error(f"An error occurred during model retraining: {str(E)}")


if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=schedule_retraining, trigger="interval", days=7)
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

    app.run(debug=True)
