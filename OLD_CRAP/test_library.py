from heavi.component import BaseComponent

class ThroughHoleResistor(BaseComponent):

    def __init__(self, resistance: float, self_inductance: float, capacitance: float):
        super().__init__()
        self.resistance = resistance
        self.self_inductance = self_inductance
        self.capacitance = capacitance

    def __on_connect__(self):
        intermediate = self.network.node()
        self.network.resistor(self.node(1), intermediate, self.resistance)
        self.network.inductor(intermediate, self.node(2), self.self_inductance)
        self.network.capacitor(self.node(1), self.node(2), self.capacitance)
    