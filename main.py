from apscheduler.schedulers.background import BackgroundScheduler
from app import create_app
import atexit

app = create_app()

def schedule_retraining():
    from app.views import retrain_models
    retrain_models()

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=schedule_retraining, trigger="interval", days=1)  # Retrains every day; adjust as necessary
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

    app.run(debug=True)

