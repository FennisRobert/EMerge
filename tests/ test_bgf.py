import emerge as em
import numpy as np
from emerge._emerge.physics.microwave.bcs import background_field as bf


def test_field_orthogonality():
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 21)
    phis = np.linspace(-np.pi, np.pi, 21)
    T, P = np.meshgrid(thetas, phis)

    for t, p in zip(T.flatten(), P.flatten()):
        field_1 = bf.BackgroundField(100, t, p, 0, origin=(0, 0, 0), definition="EA")
        field_2 = bf.BackgroundField(
            100, t, p, np.pi / 2, origin=(0, 0, 0), definition="EA"
        )
        k = field_1.k(0, 0, 0)
        E = field_1.E(0, 0, 0)
        H = field_2.E(0, 0, 0)
        k = k / np.linalg.norm(k)
        E = E / np.linalg.norm(E)
        H = H / np.linalg.norm(H)
        assert abs(1 - np.dot(k, -np.cross(E, H))) < 1e-3


if __name__ == "__main__":
    test_field_orthogonality()
