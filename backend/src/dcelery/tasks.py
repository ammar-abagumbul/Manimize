from .celery import app

@app.task
def render_manim_scene():
    return
