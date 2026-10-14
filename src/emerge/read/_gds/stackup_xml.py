# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

"""Reader for the gds2palace stack-up XML format (schemaVersion 2.0 - 3.1).

Format reference:
https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2/tree/main/doc/XML_stackup_format

Resolves <Variables> and '=' expressions, Reference/ReferenceEdge positioning, implicit thickness
stacking of dielectrics and <Substrate Offset>. All lengths are returned in meters.
Thermal data (Density, ThermalConductivity, Tables) is not used.
"""

from __future__ import annotations

import ast
import operator
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

SUPPORTED_SCHEMA_VERSION = "3.1"

_LENGTH_UNITS = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9, "mil": 25.4e-6, "in": 25.4e-3}


class StackupXMLError(Exception):
    pass


############################################################
#                   VARIABLES / EXPRESSIONS                #
############################################################

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_expression(expr: str, variables: dict[str, float | str]) -> float | str:
    """Evaluates the arithmetic expression syntax of the format: + - * / **, unary +/-, parentheses,
    numbers and variable names. A string variable may only be used on its own."""
    try:
        tree = ast.parse(expr.strip(), mode="eval").body
    except SyntaxError as e:
        raise StackupXMLError(f"Invalid expression '={expr}': {e}") from e

    if isinstance(tree, ast.Name) and isinstance(variables.get(tree.id), str):
        return variables[tree.id]

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise KeyError(node.id)
            value = variables[node.id]
            if isinstance(value, str):
                raise StackupXMLError(f"String variable '{node.id}' cannot be used in arithmetic ('={expr}').")
            return value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        raise StackupXMLError(f"Unsupported syntax in expression '={expr}'.")

    return ev(tree)


def _parse_literal(value: str, type_: str | None) -> float | str:
    if type_ == "string":
        return value
    try:
        return float(value)
    except ValueError:
        if type_ == "number":
            raise StackupXMLError(f"Variable value '{value}' is declared as a number but is not one.")
        return value


def _resolve_variables(root: ET.Element, overrides: dict[str, float | str] | None) -> dict[str, float | str]:
    raw: dict[str, tuple[str, str | None]] = {}
    section = root.find("Variables")
    if section is not None:
        for v in section.findall("Variable"):
            name = v.get("Name")
            if not name or v.get("Value") is None:
                raise StackupXMLError("Every <Variable> needs a Name and a Value.")
            if name in raw:
                raise StackupXMLError(f"Variable '{name}' is defined twice.")
            raw[name] = (v.get("Value"), v.get("Type"))

    overrides = dict(overrides or {})
    unknown = set(overrides) - set(raw)
    if unknown:
        raise StackupXMLError(f"Variable override(s) {sorted(unknown)} do not exist in the XML <Variables>: {sorted(raw)}")

    resolved: dict[str, float | str] = dict(overrides)
    pending = {k: v for k, v in raw.items() if k not in overrides}
    while pending:  # resolve by dependency, regardless of declaration order
        progress = False
        for name, (value, type_) in list(pending.items()):
            if not value.startswith("="):
                resolved[name] = _parse_literal(value, type_)
            else:
                try:
                    resolved[name] = _eval_expression(value[1:], resolved)
                except KeyError:
                    continue
                result_type = "string" if isinstance(resolved[name], str) else "number"
                if type_ and type_ != result_type:
                    raise StackupXMLError(f"Variable '{name}' is declared as {type_} but evaluates to a {result_type}.")
            del pending[name]
            progress = True
        if not progress:
            raise StackupXMLError(f"Circular or unresolvable variables: {sorted(pending)}")
    return resolved


class _Attrs:
    """Attribute access for one XML element with '=' expressions resolved."""

    def __init__(self, el: ET.Element, variables: dict[str, float | str]):
        self.el = el
        self.variables = variables

    def raw(self, name: str) -> str | None:
        value = self.el.get(name)
        if value is not None and value.startswith("="):
            try:
                return str(_eval_expression(value[1:], self.variables))
            except KeyError as e:
                raise StackupXMLError(f"Unknown variable {e} in {self.el.tag} {name}='{value}'.") from e
        return value

    def str(self, name: str, default: str | None = None) -> str | None:
        value = self.raw(name)
        return default if value is None else value

    def float(self, name: str, default: float | None = None) -> float | None:
        value = self.raw(name)
        return default if value is None else float(value)

    def int(self, name: str) -> int | None:
        value = self.float(name)
        if value is None:
            return None
        if value != int(value):
            raise StackupXMLError(f"{self.el.tag} {name} must be an integer, got {value}.")
        return int(value)


############################################################
#                        DATA CLASSES                      #
############################################################


@dataclass
class XMLMaterial:
    name: str
    type: str                       # Conductor, Dielectric, Semiconductor, Resistor
    permittivity: float = 1.0
    loss_tangent: float = 0.0
    conductivity: float = 0.0
    rs: float = 0.0                 # sheet resistance (Ohm/square), Resistor materials
    color: str | None = None        # 'rrggbb'


@dataclass
class XMLDielectric:
    name: str
    material: str
    thickness: float | None = None
    zmin: float | None = None       # resolved absolute z (m)
    zmax: float | None = None
    reference: str | None = None
    reference_edge: str = "top"
    boundary_layer: int | None = None
    _zmin_offset: float | None = None
    _zmax_offset: float | None = None
    _absolute: bool = False


@dataclass
class XMLLayer:
    name: str
    type: str                       # conductor, via, dielectric, sheet
    material: str
    gds_layer: int
    zmin: float = 0.0               # resolved absolute z (m)
    zmax: float = 0.0
    reference: str | None = None
    reference_edge: str = "top"

    @property
    def thickness(self) -> float:
        return self.zmax - self.zmin


@dataclass
class XMLDerivedLayer:
    name: str
    gds_layer: int
    operation: str                  # AND, OR, XOR, NOT, SIZE
    operands: list[int]
    oversize: float = 0.0           # meters


@dataclass
class XMLStackup:
    materials: dict[str, XMLMaterial]
    dielectrics: list[XMLDielectric]      # top to bottom, as in the file
    layers: list[XMLLayer]
    derived_layers: list[XMLDerivedLayer] = field(default_factory=list)
    variables: dict[str, float | str] = field(default_factory=dict)
    unit: float = 1e-6
    schema_version: str | None = None

    def layer(self, name: str) -> XMLLayer:
        for lay in self.layers:
            if lay.name == name:
                return lay
        raise KeyError(f"No <Layer> named '{name}'. Layers: {[lay.name for lay in self.layers]}")

    def __str__(self) -> str:
        um = 1e6
        lines = ["Dielectrics (top to bottom):"]
        for d in self.dielectrics:
            lines.append(f"  {d.name:16s} {d.material:12s} z {d.zmin * um:9.3f} .. {d.zmax * um:9.3f} um"
                         + (f"  (boundary layer {d.boundary_layer})" if d.boundary_layer is not None else ""))
        lines.append("Layers (by z):")
        for lay in sorted(self.layers, key=lambda x: x.zmin):
            lines.append(f"  {lay.name:16s} {lay.type:10s} GDS {lay.gds_layer:<4d} {lay.material:12s} "
                         f"z {lay.zmin * um:9.3f} .. {lay.zmax * um:9.3f} um")
        for d in self.derived_layers:
            lines.append(f"  derived {d.name}: layer {d.gds_layer} = {d.operation} {d.operands}"
                         + (f", oversize {d.oversize * um:g} um" if d.oversize else ""))
        return "\n".join(lines)


############################################################
#                          PARSING                         #
############################################################


def read_stackup_xml(filename: str | Path, variables: dict[str, float | str] | None = None) -> XMLStackup:
    """Reads a gds2palace stack-up XML file.

    Args:
        filename: Path to the XML file.
        variables: Overrides for values defined in the file's <Variables> section.
    """
    root = ET.parse(str(filename)).getroot()
    if root.tag != "Stackup":
        raise StackupXMLError(f"{filename} is not a stack-up file (root element <{root.tag}>).")
    schema = root.get("schemaVersion")
    if schema and tuple(map(int, schema.split("."))) > tuple(map(int, SUPPORTED_SCHEMA_VERSION.split("."))):
        logger.warning(f"Stack-up schemaVersion {schema} is newer than supported ({SUPPORTED_SCHEMA_VERSION}); "
                       "unknown attributes are ignored.")

    var = _resolve_variables(root, variables)
    materials = _read_materials(root, var)

    elayers = root.find("ELayers")
    if elayers is None:
        raise StackupXMLError("The stack-up file has no <ELayers> section.")
    unit_name = _Attrs(elayers, var).str("LengthUnit", "um")
    if unit_name not in _LENGTH_UNITS:
        raise StackupXMLError(f"Unknown LengthUnit '{unit_name}'. Known units: {sorted(_LENGTH_UNITS)}")
    unit = _LENGTH_UNITS[unit_name]

    dielectrics = _read_dielectrics(elayers, var, unit)
    layers = _read_layers(elayers, var, unit, dielectrics)
    derived = _read_derived(elayers, var, unit)

    for item in [*dielectrics, *layers]:
        if item.material != "PEC" and item.material not in materials:
            raise StackupXMLError(f"'{item.name}' uses unknown material '{item.material}'.")
    return XMLStackup(materials, dielectrics, layers, derived, var, unit, schema)


def _read_materials(root: ET.Element, var: dict) -> dict[str, XMLMaterial]:
    materials: dict[str, XMLMaterial] = {}
    section = root.find("Materials")
    for m in (section.findall("Material") if section is not None else []):
        a = _Attrs(m, var)
        mat = XMLMaterial(
            name=a.str("Name"), type=a.str("Type"),
            permittivity=a.float("Permittivity", 1.0), loss_tangent=a.float("DielectricLossTangent", 0.0),
            conductivity=a.float("Conductivity", 0.0), rs=a.float("Rs", 0.0), color=a.str("Color"),
        )
        materials[mat.name] = mat
    materials.setdefault("AIR", XMLMaterial("AIR", "Dielectric", color="d0d0d0"))
    return materials


def _edge(name: str, edge: str, dielectrics: dict, layers: dict | None = None) -> float | None:
    """z of the Top/Bottom edge of a resolved dielectric or layer, None if not resolved yet."""
    target = dielectrics.get(name) or (layers or {}).get(name)
    if target is None:
        raise StackupXMLError(f"Reference '{name}' does not exist.")
    z = target.zmax if edge == "top" else target.zmin
    return z


def _read_dielectrics(elayers: ET.Element, var: dict, unit: float) -> list[XMLDielectric]:
    section = elayers.find("Dielectrics")
    items: list[XMLDielectric] = []
    for d in (section.findall("Dielectric") if section is not None else []):
        a = _Attrs(d, var)
        zmin, zmax = a.float("Zmin"), a.float("Zmax")
        thickness = a.float("Thickness")
        item = XMLDielectric(
            name=a.str("Name"), material=a.str("Material"),
            thickness=None if thickness is None else thickness * unit,
            reference=a.str("Reference"), reference_edge=(a.str("ReferenceEdge") or "top").lower(),
            boundary_layer=a.int("Boundary"),
        )
        if item.reference is not None:
            item._zmin_offset = (zmin or 0.0) * unit
            if zmax is not None:
                item._zmax_offset = zmax * unit
            elif item.thickness is not None:
                item._zmax_offset = item._zmin_offset + item.thickness
            else:
                raise StackupXMLError(f"Dielectric '{item.name}' needs Thickness or Zmax.")
        elif zmin is not None and zmax is not None:
            item.zmin, item.zmax, item._absolute = zmin * unit, zmax * unit, True
        elif item.thickness is None:
            raise StackupXMLError(f"Dielectric '{item.name}' needs Thickness, or Zmin and Zmax.")
        items.append(item)

    # Implicit stacking: file order is top to bottom, stack the implicit ones from z = 0 upwards.
    z = 0.0
    for item in reversed(items):
        if item.reference is None and not item._absolute:
            item.zmin, z = z, z + item.thickness
            item.zmax = z

    by_name = {item.name: item for item in items}
    pending = [item for item in items if item.reference is not None]
    while pending:
        progress = False
        for item in list(pending):
            ref = by_name.get(item.reference)
            if ref is None:
                raise StackupXMLError(f"Dielectric '{item.name}' references unknown dielectric '{item.reference}'.")
            if ref.zmin is None:
                continue
            base = ref.zmax if item.reference_edge == "top" else ref.zmin
            item.zmin, item.zmax = base + item._zmin_offset, base + item._zmax_offset
            pending.remove(item)
            progress = True
        if not progress:
            raise StackupXMLError(f"Circular dielectric references: {[i.name for i in pending]}")
    return items


def _read_layers(elayers: ET.Element, var: dict, unit: float, dielectrics: list[XMLDielectric]) -> list[XMLLayer]:
    section = elayers.find("Layers")
    if section is None:
        return []
    substrate = section.find("Substrate")
    offset = (_Attrs(substrate, var).float("Offset", 0.0) if substrate is not None else 0.0) * unit

    layers: list[XMLLayer] = []
    offsets: dict[str, tuple[float, float]] = {}
    for el in section.findall("Layer"):
        a = _Attrs(el, var)
        lay = XMLLayer(
            name=a.str("Name"), type=(a.str("Type") or "conductor").lower(), material=a.str("Material"),
            gds_layer=a.int("Layer"), reference=a.str("Reference"),
            reference_edge=(a.str("ReferenceEdge") or "top").lower(),
        )
        zmin, zmax = a.float("Zmin"), a.float("Zmax")
        if zmin is None or zmax is None:
            raise StackupXMLError(f"Layer '{lay.name}' needs Zmin and Zmax.")
        if lay.reference is not None:
            offsets[lay.name] = (zmin * unit, zmax * unit)
            lay.zmin = lay.zmax = None
        else:
            lay.zmin, lay.zmax = zmin * unit + offset, zmax * unit + offset
        if zmin == zmax:
            lay.type = "sheet"
        layers.append(lay)

    if offsets and offset != 0.0:
        raise StackupXMLError("Reference-positioned layers cannot be combined with a nonzero <Substrate Offset>.")

    names = [lay.name for lay in layers]
    if offsets and len(set(names)) != len(names):
        raise StackupXMLError("Layer names must be unique when Reference is used.")
    dielectric_by_name = {d.name: d for d in dielectrics}
    clash = set(names) & set(dielectric_by_name)
    if offsets and clash:
        raise StackupXMLError(f"Names used for both a Dielectric and a Layer: {sorted(clash)}")

    layer_by_name = {lay.name: lay for lay in layers}
    pending = [lay for lay in layers if lay.reference is not None]
    while pending:
        progress = False
        for lay in list(pending):
            target = dielectric_by_name.get(lay.reference) or layer_by_name.get(lay.reference)
            if target is None:
                raise StackupXMLError(f"Layer '{lay.name}' references unknown '{lay.reference}'.")
            if target.zmin is None:
                continue
            base = target.zmax if lay.reference_edge == "top" else target.zmin
            dz0, dz1 = offsets[lay.name]
            lay.zmin, lay.zmax = base + dz0, base + dz1
            pending.remove(lay)
            progress = True
        if not progress:
            raise StackupXMLError(f"Circular layer references: {[lay.name for lay in pending]}")
    return layers


def _read_derived(elayers: ET.Element, var: dict, unit: float) -> list[XMLDerivedLayer]:
    section = elayers.find("DerivedLayers")
    derived: list[XMLDerivedLayer] = []
    for el in (section.findall("DerivedLayer") if section is not None else []):
        a = _Attrs(el, var)
        op = (a.str("Operation") or "").upper()
        operands = [_Attrs(o, var).int("Layer") for o in el.findall("Operand")]
        oversize = (a.float("Oversize", 0.0) or 0.0) * unit
        if op not in ("AND", "OR", "XOR", "NOT", "SIZE"):
            raise StackupXMLError(f"Derived layer '{a.str('Name')}': unknown Operation '{op}'.")
        if op == "SIZE" and (len(operands) != 1 or oversize == 0.0):
            raise StackupXMLError(f"Derived layer '{a.str('Name')}': SIZE needs exactly one operand and a nonzero Oversize.")
        if op != "SIZE" and len(operands) < 2:
            raise StackupXMLError(f"Derived layer '{a.str('Name')}': {op} needs at least two operands.")
        derived.append(XMLDerivedLayer(a.str("Name"), a.int("Layer"), op, operands, oversize))
    return derived
