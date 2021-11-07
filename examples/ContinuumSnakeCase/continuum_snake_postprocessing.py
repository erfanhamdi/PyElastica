import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from tqdm import tqdm


def plot_snake_velocity(
    plot_params: dict,
    period,
    filename="slithering_snake_velocity.png",
    SAVE_FIGURE=False,
):
    time_per_period = np.array(plot_params["time"]) / period
    avg_velocity = np.array(plot_params["avg_velocity"])

    [
        velocity_in_direction_of_rod,
        velocity_in_rod_roll_dir,
        _,
        _,
    ] = compute_projected_velocity(plot_params, period)

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.grid(b=True, which="minor", color="k", linestyle="--")
    ax.grid(b=True, which="major", color="k", linestyle="-")
    ax.plot(
        time_per_period[:], velocity_in_direction_of_rod[:, 2], "r-", label="forward"
    )
    ax.plot(
        time_per_period[:],
        velocity_in_rod_roll_dir[:, 0],
        c=to_rgb("xkcd:bluish"),
        label="lateral",
    )
    ax.plot(time_per_period[:], avg_velocity[:, 1], "k-", label="normal")
    fig.legend(prop={"size": 20})
    plt.show()

    if SAVE_FIGURE:
        fig.savefig(filename)


def plot_video(plot_params: dict, video_name="video.mp4", margin=0.2, fps=15):
    from matplotlib import pyplot as plt
    import matplotlib.animation as manimation

    t = np.array(plot_params["time"])
    positions_over_time = np.array(plot_params["position"])
    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    # First draw 2D plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with writer.saving(fig, video_name, dpi=300):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(1, len(t))):
                fig.clf()
                x = positions_over_time[time_idx][0]
                y = positions_over_time[time_idx][1]
                ax = plt.axes()
                ax.plot(x, y)
                ax.set_xlim(0, 4)
                ax.set_ylim(-1, 1)
                writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def compute_projected_velocity(plot_params: dict, period):

    time_per_period = np.array(plot_params["time"]) / period
    avg_velocity = np.array(plot_params["avg_velocity"])
    center_of_mass = np.array(plot_params["center_of_mass"])

    # Compute rod velocity in rod direction. We need to compute that because,
    # after snake starts to move it chooses an arbitrary direction, which does not
    # have to be initial tangent direction of the rod. Thus we need to project the
    # snake velocity with respect to its new tangent and roll direction, after that
    # we will get the correct forward and lateral speed. After this projection
    # lateral velocity of the snake has to be oscillating between + and - values with
    # zero mean.

    # Number of steps in one period.
    period_step = int(period / (time_per_period[-1] - time_per_period[-2])) + 1
    number_of_period = int(time_per_period.shape[0] / period_step)
    # Center of mass position averaged in one period
    center_of_mass_averaged_over_one_period = np.zeros((number_of_period - 2, 3))
    for i in range(1, number_of_period - 1):
        # position of center of mass averaged over one period
        center_of_mass_averaged_over_one_period[i - 1] = np.mean(
            center_of_mass[(i + 1) * period_step : (i + 2) * period_step]
            - center_of_mass[(i + 0) * period_step : (i + 1) * period_step],
            axis=0,
        )

    # Average the rod directions over multiple periods and get the direction of the rod.
    direction_of_rod = np.mean(center_of_mass_averaged_over_one_period, axis=0)
    direction_of_rod /= np.linalg.norm(direction_of_rod, ord=2)

    # Compute the projected rod velocity in the direction of the rod
    velocity_mag_in_direction_of_rod = np.einsum(
        "ji,i->j", avg_velocity, direction_of_rod
    )
    velocity_in_direction_of_rod = np.einsum(
        "j,i->ji", velocity_mag_in_direction_of_rod, direction_of_rod
    )

    # Get the lateral or roll velocity of the rod after subtracting its projected
    # velocity in the direction of rod
    velocity_in_rod_roll_dir = avg_velocity - velocity_in_direction_of_rod

    # Compute the average velocity over the simulation, this can be used for optimizing snake
    # for fastest forward velocity. We start after first period, because of the ramping up happens
    # in first period.
    average_velocity_over_simulation = np.mean(
        velocity_in_direction_of_rod[period_step * 2 :], axis=0
    )

    return (
        velocity_in_direction_of_rod,
        velocity_in_rod_roll_dir,
        average_velocity_over_simulation[2],
        average_velocity_over_simulation[0],
    )


def plot_curvature(
    plot_params: dict, rest_lengths, period, save_fig=False, filename="snake_curvature"
):
    s = np.cumsum(rest_lengths)
    L0 = s[-1]
    s = s / L0
    s = s[:-1].copy()
    x = np.linspace(0, 1, 100)
    curvature = np.array(plot_params["curvature"])
    time = np.array(plot_params["time"])
    peak_time = period * 0.125
    dt = time[1] - time[0]
    peak_idx = int(peak_time / (dt))
    fig = plt.figure(figsize=(5, 4), frameon=True, dpi=150)
    ax = fig.add_subplot(111, label="1")
    try:
        for i in range(peak_idx * 8, peak_idx * 8 * 2, peak_idx):
            ax.plot(s, curvature[i, 0, :] * L0, "k")
    except:
        print("Simulation time not long enough to plot curvature")
    ax.plot(x, 7 * np.cos(2 * np.pi * x - 0.80), "--")
    ax.set_ylabel(r"$\kappa$", fontsize=12)
    ax.set_xlabel("s", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-10, 10)
    plt.show()
    if save_fig:
        fig.savefig(filename)
