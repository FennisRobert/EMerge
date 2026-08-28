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

# Last Cleanup: 2026-03-12
from __future__ import annotations
from typing import TypeVar, Generic, Any
from .physics.microwave.microwave_data import MWData
from .physics.heatconduction.heatconduction_data import HCData
from .simulation_data import DataContainer
from .file import Saveable

class SimulationDataset(Saveable):
    """This simple class contains the different kinds of data sets in the Simulation Model. It includes

    Attributes:
      self.mw: MWData              - The Microwave physics data
      self.globals: dict[str, Any] - Any globally defined data of choice in the Simulation
      self.sim: DataContainer      - Generic simulation data associated with different instantiation of your at parameter level.
    """
    
    def __init__(self):
        self.mw: MWData = MWData()
        self.hc: HCData = HCData()
        # .mw contains all Microwave Physics related data
        self.globals: dict[str, Any] = dict()
        # .globals contains all global variables. Simple dict
        self.sim: DataContainer = DataContainer()
        # .sim contains all generic data per variable parameter permutation
    
    @staticmethod
    def combine_sets(datasets: list[SimulationDataset]) -> SimulationDataset:
        """Generate one big container from a list of SimulationDataset

        Args:
            datasets (list[SimulationDataset]): The datasets to merge

        Returns:
            DataContainer: _description_
        """
        base, *others = datasets
        base.merge_with(*others)
        return base
    
    def merge_with(self, *others: SimulationDataset) -> SimulationDataset:
        """Combines this SimulationDataset with other SimulationDatasets

        Args:
            *other (SimulationDataset): The container to merge with

        Returns:
            DataContainer: _description_
        """
        for other in others:
          self.mw.merge_with(other.mw)
          self.globals.update(other.globals)
          self.sim.merge_with(other.sim)
        return self
    
    def remove_empty_datasets(self) -> None:
      """Cleans up datasets in .sim that are never written to.
      This primarily is applied to the default dataset that is initialized when starting a simulation
      if afterwards a Parameter sweep is called thus never causing a simulation to ever store something in the default one.
      """
      self.sim.remove_empty_datasets()
    
    def initialize(self, **params) -> None:
      """Initialize new datasets for the parameters **params: dict[str, float]
      """
      self.sim.new(**params)

    def reset(self):
      """Resets all SimulationDataset data
      """
      self.mw: MWData = MWData()
      self.hc: HCData = HCData()
      self.globals: dict[str, Any] = dict()
      self.sim: DataContainer = DataContainer() 
      self.initialize()
    
    def clean(self) -> None:
      """Clean the SimulationDataset object data
      """
      del self.mw
      del self.globals
      del self.sim
      