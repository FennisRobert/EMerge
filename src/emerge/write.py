"""File export for EMerge.

    from emerge.write import FarFieldExporter, export_dxf, pcb_to_dxf
"""

from importlib import import_module

from emsutil.inexport.ffdata import FarFieldExporter

# DXF export shares its code with DXF import in emerge/read/_dxf and needs ezdxf.
_LAZY = {"export_dxf": ".read._dxf.dxf", "pcb_to_dxf": ".read._dxf.dxf"}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module 'emerge.write' has no attribute '{name}'")
    try:
        value = getattr(import_module(_LAZY[name], "emerge"), name)
    except ImportError as e:
        raise ImportError(
            f"emerge.write.{name} needs an optional package ({e.name}). Install it with: pip install emerge[dxf]"
        ) from e
    globals()[name] = value
    return value


__all__ = ["FarFieldExporter", *_LAZY]
