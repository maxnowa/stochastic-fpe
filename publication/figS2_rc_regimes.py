"""Supplementary figure 2 -- randomly connected recurrence, every regime.

Figure 4 tests recurrence in one regime. This repeats it across all four,
with the same coupling (w = 0.1) at a connection probability of 0.5. No analytical spectrum exists
for a coupled population, so the reference is the microscopic network.

Each run carries its own step size: the solver's assumed drive bound is
hard-coded for one operating point, and recurrent_plan computes what each
regime actually needs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figS_common as S                                     # noqa: E402

PANELS = [[("sup_rc_sub_low",  r"$\mu=0.8$,  $D=0.01$"),
           ("sup_rc_supra_low",  r"$\mu=1.2$,  $D=0.01$")],
          [("sup_rc_sub_high", r"$\mu=0.8$,  $D=0.1$"),
           ("sup_rc_supra_high", r"$\mu=1.2$,  $D=0.1$")]]


def main():
    S.grid("figS2_rc_regimes", PANELS,
           col_titles=["Subthreshold", "Suprathreshold"],
           row_labels=["Low noise", "High noise"])


if __name__ == "__main__":
    main()
