# from .simstate import _GLOBAL_SIMSTATES, _GEOMANAGER
# import gc
# from .selection import SELECTOR_OBJ, _CALC_INTERFACE
# import gmsh
# from .solver import DEFAULT_ROUTINE
# from .logsettings import _GlobalHandler.active().logcontroller, _GlobalHandler.active().debugcollector, _LOG_BUFFER
# from .geometry import _GEOMANAGER, _GENERATOR
# from loguru import logger
# from .geo.pcb import _PCB_MANAGER


# def cleanup() -> None:
#     """Cleanup all global states used in EMerge."""
#     logger.info("Cleaning up EMerge global states")

#     _GEOMANAGER.clear()
#     _PCB_MANAGER.clear()
#     _CALC_INTERFACE.clear()
#     SELECTOR_OBJ.clear()
#     DEFAULT_ROUTINE.reset(hard=True)
#     _GENERATOR.reset()
#     _LOG_BUFFER.clear()
#     _GLOBAL_SIMSTATES.clear()

#     for _ in range(3):
#         gc.collect()

#     if gmsh.isInitialized():
#         gmsh.clear()
#         gmsh.finalize()

#     logger.info("Cleanup complete!")

#     _GlobalHandler.active().logcontroller.clear()
#     _GlobalHandler.active().debugcollector.clear()
