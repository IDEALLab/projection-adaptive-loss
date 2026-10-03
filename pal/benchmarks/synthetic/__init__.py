from pal.benchmarks.synthetic.rosenbrock_eq import RosenbrockEq
from pal.benchmarks.synthetic.s1_sphere_track import S1SphereTrack
from pal.benchmarks.synthetic.s2_active_set_switch import S2ActiveSetSwitch
from pal.benchmarks.synthetic.s3_illcond_tube import S3IllcondTube
from pal.benchmarks.synthetic.s4_qv_coupling import S4QvCoupling
from pal.benchmarks.synthetic.s5_overdetermined import S5Overdetermined
from pal.benchmarks.synthetic.s6_redundant_ineq import S6RedundantIneq
from pal.benchmarks.synthetic.curvature_hinge import (
    KAPPA_BY_VARIANT,
    R_BY_VARIANT,
    VARIANTS,
    CurvatureHinge,
)

from pal.benchmarks.synthetic.curvature_sine import (
    KAPPA_BY_VARIANT as CURVATURE_SINE_KAPPA_BY_VARIANT,
)
from pal.benchmarks.synthetic.curvature_sine import (
    OMEGA_BY_VARIANT,
    CurvatureSine,
)
from pal.benchmarks.synthetic.curvature_sine import (
    VARIANTS as CURVATURE_SINE_VARIANTS,
)

from pal.benchmarks.synthetic.curvature_warp import (
    KAPPA_BY_VARIANT as CURVATURE_WARP_KAPPA_BY_VARIANT,
)
from pal.benchmarks.synthetic.curvature_warp import (
    VARIANTS as CURVATURE_WARP_VARIANTS,
)
from pal.benchmarks.synthetic.curvature_warp import (
    CurvatureWarp,
)

__all__ = [
    "RosenbrockEq",
    "S1SphereTrack",
    "S2ActiveSetSwitch",
    "S3IllcondTube",
    "S4QvCoupling",
    "S5Overdetermined",
    "S6RedundantIneq",
    "CurvatureHinge",
    "CurvatureSine",
    "CURVATURE_SINE_KAPPA_BY_VARIANT",
    "CURVATURE_SINE_VARIANTS",
    "CurvatureWarp",
    "CURVATURE_WARP_KAPPA_BY_VARIANT",
    "CURVATURE_WARP_VARIANTS",
    "KAPPA_BY_VARIANT",
    "OMEGA_BY_VARIANT",
    "R_BY_VARIANT",
    "VARIANTS",
]
