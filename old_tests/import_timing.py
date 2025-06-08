# python -m cProfile -o import.prof import_timing.py
import fem
from fem.mth.optimized import gaus_quad_tri
import numpy as np

print(gaus_quad_tri(4).astype(np.float64))

"""
PYTHONPROFILEIMPORTTIME=1 python3 -X importtime -c "import fem" 2>&1 \
  | grep "import time" \
  | sort -nrk4 \
  | head -20
"""