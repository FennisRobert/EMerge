import sys
import types
import gmsh


class GmshTracer:
    forbidden: list[str] = [
        "bounding_box",
        "get_center_of_mass",
        "get_normal",
        "get_nodes",
        "getBoundary",
        "get_closest_point",
        "get_boundary",
        "get_entities",
        "get_mass",
        "getMass",
        "getEntities",
        "getNormal",
        "run",  # Suppresses fltk.run
        "get_elements",
        "getElements",
    ]

    def __init__(self, target, path="gmsh", stream=None):
        self._target = target
        self._path = path
        self._stream = stream or sys.stdout

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        full_path = f"{self._path}.{name}"

        # Recurse into module and class namespaces (e.g., gmsh.model, gmsh.model.occ)
        if isinstance(attr, (type, types.ModuleType)):
            return GmshTracer(attr, full_path, self._stream)

        # Wrap executable API calls
        if callable(attr):

            def wrapper(*args, **kwargs):
                # Omit forbidden read-only/GUI operations from the log
                if not any(item in name for item in self.forbidden):
                    args_str = ", ".join(repr(a) for a in args)
                    kwargs_str = ", ".join(
                        f"{k}={v}" for k, v in kwargs.items()
                    )
                    call_args = ", ".join(filter(None, [args_str, kwargs_str]))

                    self._stream.write(f"{full_path}({call_args})\n")
                    self._stream.flush()

                return attr(*args, **kwargs)

            return wrapper

        return attr


# Initialize call recorder prior to importing EMerge modules
repro_file = open("repro_gmsh_5_bug.py", "w")
repro_file.write("import gmsh\nimport numpy as np\n\n")
sys.modules["gmsh"] = GmshTracer(gmsh, stream=repro_file)
