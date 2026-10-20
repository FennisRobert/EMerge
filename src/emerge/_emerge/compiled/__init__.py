from .baselib import CompiledLib

MATHLIB: CompiledLib = CompiledLib

# No longer use EMerge-IRON
# try:
#     import emerge_iron
#     from .iron import IRONLib
#     MATHLIB = IRONLib
#     logger.debug(f'Using EMerge-IRON as interpolation backend.')
# except ImportError:
#     logger.debug(f'Using EMR (Default) as interpolation backend.')
