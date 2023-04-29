import numpy as np
import matplotlib.pyplot as plt

# Import modules
from elastica.modules import BaseSystemCollection, Constraints, Forcing, Damping

# Import Cosserat Rod Class
from elastica.rod.cosserat_rod import CosseratRod

# Import Damping Class
from elastica.dissipation import AnalyticalLinearDamper

# Import Boundary Condition Classes
from elastica.boundary_conditions import FixedConstraint
from elastica.external_forces import GravityForces

# Import Timestepping Functions
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

class CatenaryCurveSimulator(BaseSystemCollection, Constraints, Forcing, Damping):
    pass


catenary_sim = CatenaryCurveSimulator()

# Options
PLOT_FIGURE = True
SAVE_FIGURE = False
SAVE_RESULTS = False

# setting up test params
n_elem = 100
density = 1000
nu = 2e-3
E = 1e6
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 5.4
base_radius = 0.05
base_area = np.pi * base_radius ** 2

shearable_rod= CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    0.0,     # internal damping constant, deprecated in v0.3.0
    E,
    shear_modulus=shear_modulus,
)
import matplotlib.pyplot as plt
plt.plot(
        shearable_rod.position_collection[2, :],
        shearable_rod.position_collection[1, :],
        "b-",
        label="n=" + str(shearable_rod.n_elems),
    )

catenary_sim.append(shearable_rod)

dl = base_length / n_elem
dt = 0.01 * dl

catenary_sim.dampen(shearable_rod).using(
    AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

catenary_sim.constrain(shearable_rod).using(
    FixedConstraint,
    constrained_position_idx=(0,-1)
)

gravitational_acc = -9.80665
catenary_sim.add_forcing_to(shearable_rod).using(
    GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
)

catenary_sim.finalize()

final_time = 500.0
total_steps = int(final_time / dt)

timestepper = PositionVerlet()

integrate(timestepper, catenary_sim, final_time, total_steps)
if PLOT_FIGURE:

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(
        shearable_rod.position_collection[2, :],
        shearable_rod.position_collection[1, :],
        "b-",
        label="n=" + str(shearable_rod.n_elems),
    )
    if SAVE_FIGURE:
        fig.savefig("catenary_curve.pdf")
    plt.show()
