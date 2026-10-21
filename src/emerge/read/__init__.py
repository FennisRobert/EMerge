"""File import for EMerge.

    from emerge.read import GDSFile, ODBImport, FileBasedPCB, import_dxf

Each format lives in its own subpackage (_odb, _gerber, _gds, _dxf). Formats that need
optional packages are only imported when first used, so `import emerge.read` always works
and a missing package gives an error that names the extra to install.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from emsutil.inexport.ffdata import import_farfield
from ..auxilliary.touchstone import TouchstoneData
from ._gds.gds import GDSFile, GDSLayer, GDSLabel, GDSPortEdge
from ._gds.gds_xml import GDSFromXML, DielectricStack
from ._gds.stackup_xml import read_stackup_xml

# name -> (module, extra that provides its dependencies)
_LAZY = {
    "ODBImport":        ("._odb.odbpp", "cad"),
    "ODBImportConfig":  ("._odb.odbpp", "cad"),
    "FileBasedPCB":     ("._gerber.filepcb", "cad"),
    "import_dxf":       ("._dxf.dxf", "dxf"),
    "inspect_pcb_from_dxf": ("._dxf.dxf", "dxf"),
}

if TYPE_CHECKING:
    from ._odb.odbpp import ODBImport, ODBImportConfig
    from ._gerber.filepcb import FileBasedPCB
    from ._dxf.dxf import import_dxf, inspect_pcb_from_dxf


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module 'emerge.read' has no attribute '{name}'")
    module, extra = _LAZY[name]
    try:
        value = getattr(import_module(module, __name__), name)
    except ImportError as e:
        raise ImportError(
            f"emerge.read.{name} needs an optional package ({e.name}). "
            f"Install it with: pip install emerge[{extra}]"
        ) from e
    globals()[name] = value
    return value


__all__ = ["GDSFile", "GDSFromXML", "DielectricStack", "read_stackup_xml", "GDSLabel", "GDSLayer", "GDSPortEdge", "TouchstoneData", "import_farfield", *_LAZY]
