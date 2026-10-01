"""Portable paths for the source and evaluation scripts in this package."""
from pathlib import Path
import os
import sys
import tempfile

PACKAGE_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv(PACKAGE_ROOT / '.env', override=False)


def package_path(*parts):
    """Resolve a package-relative path independently of the working directory."""
    return PACKAGE_ROOT.joinpath(*parts)


def external_path(variable, default_relative, *parts):
    """Use a configured external directory, or an explicit package-relative default."""
    value = os.environ.get(variable, '').strip()
    root = Path(os.path.expandvars(value)).expanduser() if value else package_path(default_relative)
    if not root.is_absolute():
        root = package_path(root)
    return root.joinpath(*parts)


def model_location(variable, default_model):
    """Return an optional local model path or a model identifier."""
    value = os.environ.get(variable, '').strip()
    if not value:
        return default_model
    expanded = os.path.expandvars(os.path.expanduser(value))
    candidate = Path(expanded)
    if candidate.is_absolute():
        return str(candidate)
    if expanded.startswith('.') or package_path(candidate).exists():
        return str(package_path(candidate))
    return value


def evaluation_python():
    value = os.environ.get('FEATLENS_EVAL_PYTHON', '').strip()
    if not value:
        return sys.executable
    path = Path(os.path.expandvars(value)).expanduser()
    return str(path if path.is_absolute() else package_path(path))


def temporary_path(*parts):
    return Path(tempfile.gettempdir()).joinpath('featlens', *parts)
