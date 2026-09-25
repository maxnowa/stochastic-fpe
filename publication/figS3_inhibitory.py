"""Supplementary figure 3 -- inhibitory recurrence.

A negative weight, w = -0.05, all-to-all, across all four regimes. The
solver's step-size heuristic has no branch for this: the expression
(w > 0) ? 600 : 300 hands a negative weight the 300 branch, which puts the
assumed extreme drive at mu - 15 and shrinks the step by an order of
magnitude for nothing. recurrent_plan replaces it with the bound the run
needs, which is roughly mu - 0.7 here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figS_common as S                                     # noqa: E402

PANELS = [[("sup_inh_sub_low",  r"$\mu=0.8$,  $D=0.01$"),
           ("sup_inh_supra_low",  r"$\mu=1.2$,  $D=0.01$")],
          [("sup_inh_sub_high", r"$\mu=0.8$,  $D=0.1$"),
           ("sup_inh_supra_high", r"$\mu=1.2$,  $D=0.1$")]]


def main():
    S.grid("figS3_inhibitory", PANELS,
           col_titles=["Subthreshold", "Suprathreshold"],
           row_labels=["Low noise", "High noise"])


if __name__ == "__main__":
    main()
