from pathlib import Path

from flask import Flask

from .db import ensure_schema, run_startup_maintenance
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

    ensure_schema(app.config["DATABASE"])
    maintenance_result = run_startup_maintenance(app.config["DATABASE"])
    app.config["DB_STARTUP_MAINTENANCE"] = maintenance_result
    app.register_blueprint(bp)
    return app
