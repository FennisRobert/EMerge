from .physics.microwave.microwave_data import MWData

class DataAnalyzer:

    def __init__(self, mwdata: MWData | None = None):
        self.mwdata: MWData | None = mwdata


    