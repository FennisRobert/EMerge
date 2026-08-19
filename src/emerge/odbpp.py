"""Bridge between `emcad`'s ODB++ parser / 2D polygon kernel and
`emerge`'s 3D CAD geometry. This is the only file in the `emcad`
package that imports `emerge` -- everything it pulls from `emcad`
itself (`Polygon`, `add_polygons`, `simplify_polyline`,
`via_wall_polygons`, `odbpp.*`) is already public API, by design, since
this file is slated to move into `emerge` itself so that `emcad` has no
dependency on it at all.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

import emcad as cad
import emerge as em
from .ext import GeoObject, GeoVolume, GeoSurface

from emcad.odbpp import PCBView
from emcad.odbpp.cache import save_pcb_cache, load_pcb_cache
from emcad import via_wall_polygons


def _parse_polygon(polygon: cad.Polygon, z: float, material: em.Material = em.lib.PEC) -> GeoObject:
    return em.geo.XYPolygon(polygon.xs, polygon.ys).geo(em.cs(origin=(0, 0, z))).set_material(material)


def parse_polygon(polygon: cad.Polygon, z: float, material: em.Material = em.lib.PEC,
                   extrusion: float | None = None) -> GeoObject:
    """Turn one `emcad.Polygon` -- outer boundary plus, recursively,
    every hole it carries at every nesting depth -- into a single
    `emerge` solid/face at height `z`, optionally extruded by
    `extrusion`.

    Recurses into `parse_polygon` (not the flat `_parse_polygon`) for
    each hole, so a hole that itself carries a nested hole (an island
    sitting inside it -- `Polygon.holes` supports arbitrary nesting
    depth, and both `add_polygons` and `via_wall_polygons` routinely
    produce it) is resolved correctly: subtracting a hole that's
    already had ITS OWN island subtracted-then-added-back nets out to
    re-adding that island automatically, no matter how deep the
    nesting goes -- exactly `Polygon.holes`' own even-odd convention,
    with no special-casing needed here.
    """
    main = _parse_polygon(polygon, z, material)
    if polygon.holes:
        hole_solids = [parse_polygon(hole, z, material) for hole in polygon.holes]
        tool = em.geo.unite(*hole_solids) if len(hole_solids) > 1 else hole_solids[0]
        main = em.geo.subtract(main, tool)
    if extrusion is not None:
        main = em.geo.extrude(main, dz=extrusion)
    return main


@dataclass
class ODBImportConfig:
    """Every tolerance, resolution, and refinement-size knob `ODBImport`
    uses, in one place. Construct one, tweak whatever fields matter for
    your board, and pass it in::

        config = ODBImportConfig()
        config.trace_post_simplify_delta = 20e-6   # coarser copper
        config.via_medium_segments = 16             # rounder medium vias
        odbfile = ODBImport(path, material, config=config)

    Every field also has a matching keyword argument on
    `generate_dielectric()`/`generate_traces()`/`generate_vias()`, which
    -- when passed explicitly -- overrides `self.config` for that one
    call only, without touching the config object itself.
    """

    # -- curve / arc tessellation (pads, symbols, board outline) -----------

    max_angle_deg: float = 10.0
    # Maximum angular sweep (degrees) any single tessellated arc segment
    # is allowed to cover -- governs every curved shape that isn't a via
    # circle (those use the tiered segment counts below instead): pad
    # symbols, standalone arc features, rounded-rectangle/oval corners,
    # board outline arcs. Smaller = smoother curves, more vertices, a
    # bigger boolean-union input. Forwarded to `ProductModel.from_path`.

    # -- board outline -------------------------------------------------

    board_outline_simplify_delta: float = 50e-6
    # RDP tolerance (meters) for the board outline polygon, used by
    # `generate_dielectric()`'s slab extrusion and `_generate_holes()`'s
    # cutout tool. Board outlines are usually large, simple polygons, so
    # this can be more aggressive than the trace tolerances below.

    board_outline_z_pad: float = 5e-3
    # Extra extrusion length (meters) added on each side of a board-
    # outline cutout / non-plated-hole tool, so it always fully punches
    # through the dielectric stack regardless of floating-point
    # roundoff at the exact top/bottom Z faces.

    # -- trace pre-union simplification --------------------------------

    trace_simplify_delta: float = 1e-6
    # RDP tolerance (meters) applied to each RAW feature polygon
    # independently, BEFORE the boolean union that fuses a layer's
    # pads/traces/pours together. Keep this SMALL: it exists only to
    # strip degenerate/tessellation-noise points and bound the union's
    # own input size, not to do real shape smoothing (that's
    # trace_post_simplify_delta / trace_dezigzag_max_kink_length below).
    # Setting it too large can nudge two originally-touching features
    # apart by more than merge_tol can re-snap, producing disconnected
    # fragments with hairline gaps instead of one fused polygon -- see
    # `odbpp.design.resolve_layer_polygons`'s docstring for the story.

    merge_tol: float | None = None
    # Exact-grid vertex-merge tolerance (meters) for the boolean union
    # itself. `None` uses the kernel's own default
    # (`kernel._constants.DEFAULT_MERGE_TOL`, currently 0.1um) -- only
    # worth overriding as a targeted escape hatch for a specific board
    # that still shows gaps with the tolerances above.

    # -- trace post-union smoothing --------------------------------------

    trace_dezigzag: bool = True
    # If True, clean up short zigzag/step artifacts a union of many
    # round-capped trace segments tends to leave behind -- a
    # tessellated circle's polygon approximation slightly missing the
    # tangent point where it should cleanly meet a straight segment.
    # Best-effort: can never break a valid design (see
    # `Polygon.dezigzag()`'s own docs).

    trace_dezigzag_max_kink_length: float = 10e-6
    # A segment shorter than this (meters) is a candidate "kink" to
    # collapse -- should stay well under your smallest real feature size.

    trace_dezigzag_max_angle_deg: float = 20.0
    # Maximum direction change (degrees) allowed between a kink's two
    # neighboring segments for it to still count as "basically straight".

    trace_dezigzag_min_neighbor_factor: float = 3.0
    # Each of a kink's two neighboring segments must be at least this
    # many times `trace_dezigzag_max_kink_length` long, so a run of
    # several genuinely small features isn't mistaken for one artifact.

    trace_post_simplify: bool = True
    # If True, run a second RDP pass on each layer's UNIONED output
    # (after dezigzag) -- this is where the real, aggressive point-count
    # reduction happens, safely, since it runs on already-fused geometry
    # instead of independently on each raw input feature (unlike
    # `trace_simplify_delta` above).

    trace_post_simplify_delta: float = 10e-6
    # RDP tolerance (meters) for that post-union pass.

    # -- via circle tessellation, tiered by drill diameter -----------------

    via_tiny_max_diameter: float = 0.3e-3
    # Vias with diameter <= this (meters) are "tiny" -- get
    # `via_tiny_segments`-sided circles.

    via_tiny_segments: int = 6
    # Circle tessellation segment count for tiny vias (e.g. signal vias).

    via_medium_max_diameter: float = 1.0e-3
    # Vias with diameter in (via_tiny_max_diameter, this] are "medium" --
    # get `via_medium_segments`-sided circles. Anything larger is "large".

    via_medium_segments: int = 12
    # Circle tessellation segment count for medium vias (e.g. power vias).

    via_large_segments: int = 24
    # Circle tessellation segment count for vias larger than
    # `via_medium_max_diameter` (e.g. mounting/mechanical holes).

    def segments_for_via(self, diameter: float) -> int:
        """Pick the tiered circle-tessellation segment count for a via
        of the given diameter (meters)."""
        if diameter <= self.via_tiny_max_diameter:
            return self.via_tiny_segments
        if diameter <= self.via_medium_max_diameter:
            return self.via_medium_segments
        return self.via_large_segments

    # -- via-wall assembly (generate_vias(autojoin_limit=...)) --------------

    via_post_simplify_delta: float = 1e-6
    # RDP tolerance (meters) applied to each assembled via-wall polygon
    # after `via_wall_polygons` builds it.

    via_min_hole_area_factor: float = 1.1
    # `generate_vias()`'s default `min_hole_area`, when not given
    # explicitly, is `autojoin_limit**2 * this factor` -- an enclosed
    # loop in the via network smaller than that is left solid instead
    # of carved into a hole.

    via_edge_width_factor: float = 1.0
    # `via_wall_polygons`' connecting-rectangle half-width as a fraction
    # of each via's own radius (1.0 = full radius, same thickness as the
    # via circles themselves; lower for a narrower connecting wall).

    via_angle_tol: float = 1e-3
    # `via_wall_polygons`' collinearity tolerance (radians) for merging
    # a straight run of vias into a single connecting segment.

    # -- mesh refinement (suggested sizes -- not auto-applied) --------------

    mesh_refinement_traces: float | None = None
    # Suggested boundary mesh-refinement size (meters) for copper
    # traces, e.g. `sim.mesher.set_boundary_size(em.select(*traces),
    # config.mesh_refinement_traces)`. Purely a place to keep this
    # number next to the geometry tolerances it's related to --
    # `ODBImport` never calls the mesher itself, since it doesn't own
    # your `Simulation` object. `None` means "no suggestion".

    mesh_refinement_vias: float | None = None
    # Same idea as `mesh_refinement_traces`, for via geometry.


class ODBImport:
    """Load an ODB++ PCB design and turn it into `emerge` CAD geometry.

    Quick start::

        odbfile = ODBImport(
            "example_files/RO4350B_array_PCB-odb",
            material,
            stack_thickness=[35 * um, 508 * um, 35 * um],
            reverse_stack=True,
        )
        diel = odbfile.generate_dielectric()
        traces = odbfile.generate_traces()
        vias = odbfile.generate_vias(autojoin_limit=0.001)

    `generate_dielectric()` builds the board's copper/dielectric slab
    stack, `generate_traces()` the per-layer copper (pads/traces/pours,
    already boolean-unified -- see `odbpp.design.resolve_layer_polygons`),
    and `generate_vias()` the plated drill holes, either as individual
    cylinders or, for dense via fences, thickened wall polygons (see
    `viaconnect.via_wall_polygons`). All three read from the same parsed
    `PCBView` (`self.pcbd`), built once in `__init__`.

    Every tolerance/resolution knob these three methods use is read from
    `self.config` (an `ODBImportConfig`) unless overridden per call --
    see that class for the full, centrally-documented list.
    """

    def __init__(self,
                 filename: str,
                 pcb_material: em.Material,
                 stack_thickness: list[float] | None = None,
                 reverse_stack: bool = False,
                 copper_placement: Literal['bottom', 'center', 'top'] = 'center',
                 centralize: bool = True,
                 thick_traces: bool = False,
                 cache_path: str | None = None,
                 plot: bool = False,
                 config: ODBImportConfig | None = None):
        """
        Args:
            filename: path to the ODB++ product model directory.
            pcb_material: `emerge` material for the dielectric slabs.
            stack_thickness: per-structural-layer thickness override,
                forwarded to `PCBView(thickness_stack=...)` -- see
                that class for the exact top-to-bottom ordering.
            reverse_stack: Z-flip the whole board (see
                `PCBView(reverse=...)`).
            copper_placement: where a copper layer's thickness sits
                relative to its neighboring dielectrics -- see
                `PCBView`'s own docs.
            centralize: shift the board so its outline's bounding-box
                center sits at the origin.
            thick_traces: give copper layers real Z-thickness in the
                stack instead of a flat, zero-thickness marker.
            cache_path: if given and that file already exists, load the
                resolved geometry straight from it (`odbpp.cache.
                load_pcb_cache`) instead of re-parsing `filename` and
                re-resolving every feature -- skips the two most
                expensive steps before the boolean kernel even starts,
                which matters for a script that re-runs against the
                same board repeatedly. If given but the file does NOT
                exist yet, parse normally and then save a cache there
                for next time. `None` (the default) never caches --
                always parses `filename` from scratch. `self.pm` (the
                raw parsed `ProductModel`) is `None` when loaded from
                cache, since a cache only stores already-resolved
                geometry.
            plot: if True, pop up one matplotlib figure per geometry
                layer (`PCBView.plot_layers()`) right after
                loading, as a visual sanity check. Off by default --
                each figure blocks on `plt.show()` until closed, which
                isn't what you want in an automated pipeline or test.
            config: an `ODBImportConfig` collecting every tolerance/
                resolution/refinement-size knob the `generate_*`
                methods use. `None` (the default) builds one with
                stock defaults -- see that class for the full list.
                Stored as `self.config`; mutate it any time before
                calling `generate_*`, or override individual values
                per call via that method's own keyword arguments.
        """
        self.filename = filename
        self.pcb_material: em.Material = pcb_material
        self.thick_traces: bool = thick_traces
        self.config: ODBImportConfig = config if config is not None else ODBImportConfig()
        self.zmin: float | None = None
        self.zmax: float | None = None

        if cache_path is not None and Path(cache_path).exists():
            logger.debug(f"ODBImport: loading cached geometry from '{cache_path}'")
            self.pcbd = load_pcb_cache(cache_path)
        else:
            logger.debug(f"ODBImport: parsing '{filename}'")
            # PCBView(filename, ...) parses AND resolves in one step --
            # see ProductModel.from_path, which it calls internally.
            self.pcbd = PCBView(filename,
                                include_types={"SIGNAL", "POWER_GROUND"},
                                reverse=reverse_stack,
                                copper_placement=copper_placement,
                                thick_traces=thick_traces,
                                thickness_stack=stack_thickness,
                                max_angle_deg=self.config.max_angle_deg)
            if cache_path is not None:
                save_pcb_cache(self.pcbd, cache_path)
        self.pm = self.pcbd.pm

        if plot:
            self.pcbd.plot_layers()
        if centralize:
            self.pcbd.centralize()

    def _generate_holes(self) -> GeoVolume | None:
        """Board-outline cutouts (`get_board_holes()`) AND non-plated
        drill holes, unioned into one tool for `generate_dielectric()`
        to subtract. Plated holes are deliberately NOT included here --
        those get copper via `generate_vias()` instead of a bare hole.

        Non-plated holes are drawn at their own [z1, z2] (a blind/buried
        hole doesn't span the whole board), not the board-outline
        holes' full-stack span.
        """
        z_pad = self.config.board_outline_z_pad

        holes = []
        for hxs, hys in self.pcbd.get_board_holes():
            xs_clean, ys_clean = cad.simplify_polyline(hxs, hys, self.config.board_outline_simplify_delta)
            holes.append(
                em.geo.XYPolygon(xs_clean, ys_clean).extrude(
                    self.zmax - self.zmin + z_pad,
                    em.cs(origin=(0, 0, self.zmin - z_pad / 2)),
                )
            )

        for hole in self.pcbd.iter_drill_holes():
            if hole.plated:
                continue
            depth = hole.z2 - hole.z1
            if depth <= 0:
                continue
            holes.append(
                em.geo.Cylinder(hole.diameter / 2, depth, em.cs(origin=(hole.x, hole.y, hole.z1)))
            )

        if not holes:
            return None
        return em.geo.unite(*holes)

    def generate_dielectric(self, merged: bool = True, outline_simplify_delta: float | None = None) -> GeoVolume | list[GeoVolume]:
        """Build the board's dielectric slab stack, with board-outline
        cutouts and non-plated drill holes already subtracted.

        Args:
            merged: if True (the default), union every dielectric slab
                into a single `GeoVolume`. If False, keep each slab
                separate (holes still subtracted from each) and return
                a list, one per dielectric layer top-to-bottom.
            outline_simplify_delta: RDP tolerance (meters) for the board
                outline. `None` (the default) uses
                `self.config.board_outline_simplify_delta`.
        """
        logger.debug("generate_dielectric: start")

        delta = outline_simplify_delta if outline_simplify_delta is not None else self.config.board_outline_simplify_delta
        xs, ys = self.pcbd.get_board_polygon()
        xs, ys = cad.simplify_polyline(xs, ys, delta)

        layers = []
        zs = []

        # Physical Z-stack (copper + dielectric slabs)
        for z1, z2, material in self.pcbd.iter_pcb_layers():
            if material != "DIELECTRIC":
                continue
            zs.extend([z1, z2])
            poly = em.geo.XYPolygon(xs, ys).extrude(z2 - z1, em.cs(origin=(0, 0, z1))).set_material(self.pcb_material)
            layers.append(poly)

        self.zmin = min(zs)
        self.zmax = max(zs)

        holes = self._generate_holes()
        logger.debug(f"generate_dielectric: {len(layers)} dielectric layer(s), holes={'yes' if holes is not None else 'no'}")

        if merged:
            layer = em.geo.unite(*layers).set_material(self.pcb_material)
            if holes is not None:
                layer = em.geo.subtract(layer, holes)
            return layer

        if holes is None:
            return layers

        new_layers = []
        for layer in layers:
            nl = em.geo.subtract(layer, holes, remove_tool=False)
            nl.properties = layer.properties
            new_layers.append(nl)
        return new_layers

    def generate_traces(
        self,
        dezigzag: bool | None = None,
        dezigzag_max_kink_length: float | None = None,
        dezigzag_max_angle_deg: float | None = None,
        dezigzag_min_neighbor_factor: float | None = None,
        post_simplify: bool | None = None,
        post_simplify_delta: float | None = None,
        simplify_delta: float | None = None,
        merge_tol: float | None = None,
    ) -> list[GeoVolume | GeoSurface]:
        """Build copper geometry (pads/traces/pours) for every signal
        layer, already boolean-unified per layer -- see
        `odbpp.design.resolve_layer_polygons`, which every parameter
        below is forwarded to. Every parameter defaults to `None`,
        meaning "use `self.config`" -- see `ODBImportConfig` for what
        each one does; passing a value here overrides that one call
        only, without touching `self.config`.
        """
        cfg = self.config
        simplify_delta = simplify_delta if simplify_delta is not None else cfg.trace_simplify_delta
        dezigzag = dezigzag if dezigzag is not None else cfg.trace_dezigzag
        dezigzag_max_kink_length = (
            dezigzag_max_kink_length if dezigzag_max_kink_length is not None else cfg.trace_dezigzag_max_kink_length
        )
        dezigzag_max_angle_deg = (
            dezigzag_max_angle_deg if dezigzag_max_angle_deg is not None else cfg.trace_dezigzag_max_angle_deg
        )
        dezigzag_min_neighbor_factor = (
            dezigzag_min_neighbor_factor if dezigzag_min_neighbor_factor is not None
            else cfg.trace_dezigzag_min_neighbor_factor
        )
        post_simplify = post_simplify if post_simplify is not None else cfg.trace_post_simplify
        post_simplify_delta = (
            post_simplify_delta if post_simplify_delta is not None else cfg.trace_post_simplify_delta
        )
        merge_tol = merge_tol if merge_tol is not None else cfg.merge_tol

        logger.debug("generate_traces: start")

        polygons = []
        for layer in self.pcbd.iter_geo_layers():
            # add-minus-remove is already resolved here, in emcad's own
            # 2D kernel -- see PCBView.resolve_layer_polygons. Same
            # method PCBView.plot_layers() uses, so what you see
            # there is exactly what ends up here.
            for poly in self.pcbd.resolve_layer_polygons(
                layer, simplify_delta=simplify_delta,
                dezigzag=dezigzag, dezigzag_max_kink_length=dezigzag_max_kink_length,
                dezigzag_max_angle_deg=dezigzag_max_angle_deg,
                dezigzag_min_neighbor_factor=dezigzag_min_neighbor_factor,
                post_simplify=post_simplify, post_simplify_delta=post_simplify_delta,
                merge_tol=merge_tol,
            ):
                polygons.append(parse_polygon(poly, layer.z1, material=em.lib.COPPER))

        logger.debug(f"generate_traces: {len(polygons)} solid(s)/surface(s) built")
        return polygons

    def generate_vias(
        self,
        segments: int | None = None,
        autojoin_limit: float | None = None,
        min_hole_area: float | None = None,
        view_process: bool = False,
    ) -> list[GeoVolume]:
        """Build copper geometry for every plated drill hole. Non-plated
        holes are cut from the dielectric instead -- see
        `generate_dielectric()`/`_generate_holes()`.

        Args:
            segments: circle tessellation resolution for every via.
                `None` (the default) picks a resolution PER VIA from its
                own drill diameter, via `self.config.segments_for_via()`
                (see `ODBImportConfig.via_tiny_segments` /
                `via_medium_segments` / `via_large_segments`) -- pass an
                explicit value here to use one flat resolution for every
                via instead.
            autojoin_limit: if given, vias closer than this (meters)
                are fused into a single thickened wall polygon per
                (Z plane, drill diameter) group instead of meshed as
                individual cylinders -- see `viaconnect.
                via_wall_polygons`. Each group's wall is drawn at that
                group's own actual drill diameter. `None` (the default)
                always uses plain individual cylinders.
            min_hole_area: only used with `autojoin_limit` -- drop any
                enclosed loop in the via network smaller than this
                (m^2) instead of carving it out. `None` (the default)
                picks `autojoin_limit**2 * self.config.via_min_hole_area_factor`.
            view_process: only used with `autojoin_limit` -- pop up a
                matplotlib figure illustrating the via-wall assembly
                pipeline. Off by default; see `via_wall_polygons` for
                what it shows.
        """
        cfg = self.config

        if autojoin_limit is None:
            vias = []
            for hole in self.pcbd.iter_drill_holes():
                if not hole.plated:
                    continue
                n = segments if segments is not None else cfg.segments_for_via(hole.diameter)
                via = em.geo.Cylinder(
                    hole.diameter / 2, hole.z2 - hole.z1, em.cs(origin=(hole.x, hole.y, hole.z1)), Nsections=n
                ).set_material(em.lib.COPPER)
                vias.append(via)
            logger.debug(f"generate_vias: {len(vias)} solid(s) built (plain-cylinder path)")
            return vias

        if min_hole_area is None:
            min_hole_area = autojoin_limit ** 2 * cfg.via_min_hole_area_factor

        # Grouped by (z1, z2, diameter) -- not just (z1, z2) -- since
        # via_wall_polygons takes one `thickness` per call and draws
        # every via in the batch at that same diameter. Vias with a
        # different drill size sharing the same Z-span still need their
        # own call so each gets its own actual diameter, not silently
        # merged and drawn at a size that doesn't match the source data.
        vias_by_group = defaultdict(list)
        for hole in self.pcbd.iter_drill_holes():
            if not hole.plated:
                continue
            vias_by_group[(hole.z1, hole.z2, hole.diameter)].append((hole.x, hole.y))
        logger.debug(f"generate_vias: {len(vias_by_group)} via (z-plane, diameter) group(s) to wall-join")

        via_solids = []
        for (z1, z2, diameter), xys in vias_by_group.items():
            xys = np.array(xys)
            n = segments if segments is not None else cfg.segments_for_via(diameter)
            logger.debug(
                f"generate_vias: z-plane [{z1 * 1e6:.1f}, {z2 * 1e6:.1f}]um "
                f"diameter={diameter * 1e6:.1f}um ({n} segments) -- {len(xys)} via(s)"
            )
            wall_polygons = via_wall_polygons(
                xys, thickness=diameter, max_dist=autojoin_limit, circle_segments=n,
                edge_width_factor=cfg.via_edge_width_factor, angle_tol=cfg.via_angle_tol,
                min_hole_area=min_hole_area, include_isolated_vias=True,
                _view_process=view_process,
            )
            for poly in wall_polygons:
                poly.simplify(cfg.via_post_simplify_delta)
                via_solids.append(
                    parse_polygon(poly, z1, material=em.lib.COPPER, extrusion=z2 - z1).set_material(em.lib.COPPER)
                )
        logger.debug(f"generate_vias: {len(via_solids)} solid(s) built (wall-joined path)")
        return via_solids
