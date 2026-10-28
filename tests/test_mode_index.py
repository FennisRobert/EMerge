# From asirinelli's EMerge 2.8.9 patch gist
"""Regression test for ambiguous multi-mode port indices (EMerge 2.8.9).

PortBC._gen_port_number builds float(f"{port}.{mode}"), so with >= 10 modes
mode 10 -> 1.10 == 1.1 (mode 1), mode 20 -> 1.2 (mode 2), ...  The S-matrix
port map then silently overwrites the entries of the colliding modes.
"""
from types import SimpleNamespace

from emerge._emerge.physics.microwave.bcs.port_bcs import PortBC


def keys(port, nmodes):
    p = SimpleNamespace(port_number=port, port_modes={i: None for i in range(1, nmodes + 1)})
    return [PortBC._gen_port_number(p, i) for i in range(1, nmodes + 1)]


def test_keys_unique_up_to_99_modes():
    for n in (2, 9, 10, 11, 23, 72, 99):
        k = keys(1, n) + keys(2, n)
        assert len(set(k)) == 2 * n, f"duplicate port indices for {n} modes"


def test_keys_unchanged_for_up_to_9_modes():
    assert keys(1, 9) == [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    assert keys(2, 3) == [2.1, 2.2, 2.3]


def test_single_mode_port_keeps_integer_index():
    assert keys(3, 1) == [3]


if __name__ == "__main__":
    test_keys_unique_up_to_99_modes()
    test_keys_unchanged_for_up_to_9_modes()
    test_single_mode_port_keeps_integer_index()
    print("PASS")
