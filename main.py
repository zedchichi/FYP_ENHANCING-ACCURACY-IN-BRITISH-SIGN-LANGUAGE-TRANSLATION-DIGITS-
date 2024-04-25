from apscheduler.schedulers.background import BackgroundScheduler
from app import create_app
import atexit
from flask_talisman import Talisman

app = create_app()
Talisman(app, content_security_policy=None) # Enforce HTTPS with default settings, disable CSP for simplicity

def schedule_retraining():
    from app.views import retrain_models
    retrain_models()

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=schedule_retraining, trigger="interval", days=1)  # Retrains every day
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

    app.run(debug=True) #Remove adhoc SSL context for development

