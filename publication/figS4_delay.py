"""Supplementary figure 4 -- transmission delay.

Fully connected, w = 0.1, suprathreshold and low noise, with the delay
stepped from 1 to 5 ms. The delay does not move the stationary rate -- all
four runs sit at 664 Hz -- so what changes is purely the dynamics: the
feedback loop acquires a phase lag and the resonance moves down in frequency
as the delay grows.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figS_common as S                                     # noqa: E402

PANELS = [[("sup_delay_1", r"$d=1$ ms"), ("sup_delay_2", r"$d=2$ ms")],
          [("sup_delay_3", r"$d=3$ ms"), ("sup_delay_5", r"$d=5$ ms")]]


def main():
    S.grid("figS4_delay", PANELS)


if __name__ == "__main__":
    main()
