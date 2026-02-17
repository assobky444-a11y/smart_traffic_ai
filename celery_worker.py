import os
import sys
# Ensure project root is on PYTHONPATH so `import app` works in forked worker processes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from celery import Celery

# Broker/back-end can be overridden via env vars (recommended in production)
BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
BACKEND_URL = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

celery = Celery('vehicles_tracking', broker=BROKER_URL, backend=BACKEND_URL)
celery.conf.update({
    'task_track_started': True,
    'task_serializer': 'json',
    'result_serializer': 'json',
    'accept_content': ['json'],
})


@celery.task(bind=True)
def run_process_video(self, job_id, video_path, model_path, output_dir,
                      confidence, iou_threshold, max_age, min_hits, roi_meta=None):
    """Celery wrapper that calls the existing process_video_task from app.py.
    This imports inside the task to avoid circular imports at module import time.
    """
    try:
        # import here so Flask app context and module-level symbols are available
        from app import app as flask_app, processing_jobs, process_video_task
        import json
        import os
        from datetime import datetime

        with flask_app.app_context():
            # Ensure a local in-process entry exists for the Celery worker
            processing_jobs[job_id] = processing_jobs.get(job_id, {})
            processing_jobs[job_id].update({
                'status': 'processing',
                'progress': 0,
                'output_dir': output_dir,
                'start_time': datetime.now().isoformat()
            })

            # write job-sidecar index so web workers can discover job immediately
            try:
                job_index_dir = os.path.join(flask_app.config['OUTPUT_FOLDER'], job_id)
                os.makedirs(job_index_dir, exist_ok=True)
                index_meta = {
                    'status': 'processing',
                    'output_dir': output_dir,
                    'start_time': processing_jobs[job_id].get('start_time'),
                    'video_path': video_path,
                    'analysis_id': job_id
                }
                with open(os.path.join(job_index_dir, 'meta.json'), 'w', encoding='utf-8') as fh:
                    json.dump(index_meta, fh, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # Call the existing processing implementation (runs the heavy work)
        process_video_task(job_id, video_path, model_path, output_dir,
                           confidence, iou_threshold, max_age, min_hits, roi_meta)

        return {'job_id': job_id, 'status': 'completed'}

    except Exception as e:
        # Celery will capture this as task failure; also attempt to mark job as error
        try:
            from app import processing_jobs
            processing_jobs[job_id] = processing_jobs.get(job_id, {})
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['error'] = str(e)
        except Exception:
            pass
        raise
