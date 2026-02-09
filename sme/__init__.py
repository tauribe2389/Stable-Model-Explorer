from pathlib import Path

from flask import Flask

from .db import ensure_db
from .routes import bp


def create_app() -> Flask:
    root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder=str(root / "templates"),
        static_folder=str(root / "static"),
        static_url_path="/static",
    )
    app.config.update(
        SECRET_KEY="sme-local-dev",
        DATABASE="instance/sme.db",
        UPLOAD_DIR="uploads",
        ARTIFACT_DIR="artifacts",
    )

    ensure_db(app.config["DATABASE"])
    app.register_blueprint(bp)
    return app
