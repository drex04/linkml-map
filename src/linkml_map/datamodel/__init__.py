from pathlib import Path

from linkml_map.datamodel.transformer_model import FunctionCallConfiguration

SCHEMA_DIR = Path(__file__).parent
TR_SCHEMA = SCHEMA_DIR / "transformer_model.yaml"

__all__ = ["FunctionCallConfiguration", "SCHEMA_DIR", "TR_SCHEMA"]
