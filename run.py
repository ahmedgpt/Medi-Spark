"""Application entry point."""
from app import create_app

try:
    from config.settings import Config
except ImportError:  # pragma: no cover
    from .config.settings import Config

app = create_app()

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
