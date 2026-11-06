
"""Symmetric 2-DOF half-car suspension model.

The model has body bounce z and body pitch theta as its two coordinates.
Front and rear road inputs follow the same smooth bump, with the rear delayed
by the wheelbase divided by vehicle speed.

All quantities use SI units. k and c are the combined properties at each axle.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Editable model parameters (SI units)
# ---------------------------------------------------------------------------
MASS = 1200.0              # body mass represented by the model [kg]
PITCH_INERTIA = 1800.0     # pitch moment of inertia about the center of mass [kg m^2]
HALF_WHEELBASE = 1.4       # distance from center of mass to either axle, ell [m]

SPRING_STIFFNESS = 30_000.0  # stiffness at EACH axle, k [N/m]
DAMPING_COEFFICIENT = 3_000.0  # damping at EACH axle, c [N s/m]

BUMP_HEIGHT = 0.05         # cosine bump height [m]
BUMP_LENGTH = 1.0          # bump length along the road [m]
VEHICLE_SPEED = 5.0        # forward speed [m/s]

SETTLING_TIME_AFTER_REAR = 5.0  # simulate this long after rear clears bump [s]
NUMBER_OF_OUTPUT_POINTS = 2000

# ---------------------------------------------------------------------------
# Road input: front axle sees the bump first; rear axle follows after tau.
# ---------------------------------------------------------------------------
WHEELBASE = 2.0 * HALF_WHEELBASE
REAR_DELAY = WHEELBASE / VEHICLE_SPEED
BUMP_DURATION = BUMP_LENGTH / VEHICLE_SPEED


def bump_profile(time):
    """Return road displacement and its time derivative at `time`.

    The profile is a half-cosine bump from t=0 to t=BUMP_DURATION, and zero
    outside that interval. Its displacement and velocity are zero at both ends.
    """
    if 0.0 <= time <= BUMP_DURATION:
        phase = 2.0 * np.pi * VEHICLE_SPEED * time / BUMP_LENGTH
        road_height = 0.5 * BUMP_HEIGHT * (1.0 - np.cos(phase))
        road_velocity = (
            np.pi * BUMP_HEIGHT * VEHICLE_SPEED / BUMP_LENGTH * np.sin(phase)
        )
        return road_height, road_velocity

    return 0.0, 0.0


def road_inputs(time):
    """Return front/rear road heights and velocities: rf, rr, rf_dot, rr_dot."""
    rf, rf_dot = bump_profile(time)
    rr, rr_dot = bump_profile(time - REAR_DELAY)
    return rf, rr, rf_dot, rr_dot


def road_height_at_position(position):
    """Return the bump height at a location along the road."""
    if 0.0 <= position <= BUMP_LENGTH:
        phase = 2.0 * np.pi * position / BUMP_LENGTH
        return 0.5 * BUMP_HEIGHT * (1.0 - np.cos(phase))
    return 0.0


# ---------------------------------------------------------------------------
# Equations of motion, written as a first-order state derivative for solve_ivp.
# State order: [z, z_dot, theta, theta_dot]
# ---------------------------------------------------------------------------
def equations_of_motion(
    time, state, spring_stiffness=SPRING_STIFFNESS,
    damping_coefficient=DAMPING_COEFFICIENT,
):
    """Return [z_dot, z_ddot, theta_dot, theta_ddot] at the current state."""
    z, z_dot, theta, theta_dot = state
    rf, rr, rf_dot, rr_dot = road_inputs(time)

    # Bounce EOM:
    # M z_ddot + 2 c z_dot + 2 k z
    #     = k (rf + rr) + c (rf_dot + rr_dot)
    z_ddot = (
        spring_stiffness * (rf + rr)
        + damping_coefficient * (rf_dot + rr_dot)
        - 2.0 * damping_coefficient * z_dot
        - 2.0 * spring_stiffness * z
    ) / MASS

    # Pitch EOM:
    # I theta_ddot + 2 c ell^2 theta_dot + 2 k ell^2 theta
    #     = ell k (rf - rr) + ell c (rf_dot - rr_dot)
    theta_ddot = (
        HALF_WHEELBASE * spring_stiffness * (rf - rr)
        + HALF_WHEELBASE * damping_coefficient * (rf_dot - rr_dot)
        - 2.0 * damping_coefficient * HALF_WHEELBASE**2 * theta_dot
        - 2.0 * spring_stiffness * HALF_WHEELBASE**2 * theta
    ) / PITCH_INERTIA

    return np.array([z_dot, z_ddot, theta_dot, theta_ddot])


def main():
    """Integrate the model and animate the car with its bounce and pitch plots."""
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])
    end_time = REAR_DELAY + BUMP_DURATION + SETTLING_TIME_AFTER_REAR
    time_points = np.linspace(0.0, end_time, NUMBER_OF_OUTPUT_POINTS)

    def run_simulation(spring_stiffness, damping_coefficient):
        result = solve_ivp(
            fun=equations_of_motion,
            t_span=(0.0, end_time),
            y0=initial_state,
            t_eval=time_points,
            args=(spring_stiffness, damping_coefficient),
            rtol=1e-7,
            atol=1e-9,
        )
        if not result.success:
            raise RuntimeError(f"Integration failed: {result.message}")
        return result

    solution = run_simulation(SPRING_STIFFNESS, DAMPING_COEFFICIENT)
    z = solution.y[0]
    theta = solution.y[2]

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1.1, 1, 1]}
    )
    car_ax, z_ax, theta_ax = axes

    # The road coordinate is fixed to the bump: at t=0 the front axle is at x=0.
    # The camera follows the car so the bump stays in view during the pass.
    view_half_width = 2.4 * HALF_WHEELBASE
    road_x = np.linspace(-2.0 * HALF_WHEELBASE, BUMP_LENGTH + 2.0 * HALF_WHEELBASE, 500)
    road_y = np.array([road_height_at_position(x) for x in road_x])
    car_ax.plot(road_x, road_y, color="saddlebrown", linewidth=3)
    car_ax.axhline(0.0, color="0.35", linewidth=1)
    car_ax.set_xlim(-view_half_width, view_half_width)
    car_ax.set_ylim(-0.22, 0.85)
    car_ax.set_ylabel("Height [m]")
    car_ax.set_title("Car crossing the speed bump")
    car_ax.set_yticks([])
    car_ax.grid(True, axis="x", alpha=0.25)
    car_ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # Draw a simple rigid body and two wheels. Body motion is visually scaled
    # so the small modeled displacements remain clear beside the road bump.
    visual_scale = 3.0
    body_base = 0.52
    reference_body_line, = car_ax.plot(
        [], [], color="0.4", linestyle="--", linewidth=2, label="z = θ = 0 reference"
    )
    reference_points, = car_ax.plot(
        [], [], linestyle="None", marker="o", color="0.4", markersize=5
    )
    body_line, = car_ax.plot([], [], color="tab:blue", linewidth=7, solid_capstyle="round")
    current_points, = car_ax.plot(
        [], [], linestyle="None", marker="o", color="tab:orange", markersize=6
    )
    z_indicator, = car_ax.plot([], [], color="tab:green", linewidth=2)
    motion_label = car_ax.text(
        0, 0, "", ha="center", va="bottom", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
    )
    front_spring, = car_ax.plot([], [], color="tab:gray", linewidth=2)
    rear_spring, = car_ax.plot([], [], color="tab:gray", linewidth=2)
    front_wheel = plt.Circle((0, 0), 0.12, color="black")
    rear_wheel = plt.Circle((0, 0), 0.12, color="black")
    car_ax.add_patch(front_wheel)
    car_ax.add_patch(rear_wheel)
    time_label = car_ax.text(0.02, 0.92, "", transform=car_ax.transAxes)

    z_zero = z_ax.axhline(0.0, color="0.45", linestyle=":", linewidth=1.2)
    z_line, = z_ax.plot(solution.t, z, color="tab:blue")
    z_point, = z_ax.plot([], [], marker="o", color="tab:blue", markersize=7, zorder=4)
    z_point_label = z_ax.annotate(
        "", xy=(0, 0), xytext=(7, 8), textcoords="offset points",
        color="tab:blue", fontsize=9,
    )
    z_ax.set_ylabel("Body bounce z [m]")
    z_ax.grid(True)

    theta_zero = theta_ax.axhline(0.0, color="0.45", linestyle=":", linewidth=1.2)
    theta_line, = theta_ax.plot(solution.t, theta, color="tab:orange")
    theta_point, = theta_ax.plot([], [], marker="o", color="tab:orange", markersize=7, zorder=4)
    theta_point_label = theta_ax.annotate(
        "", xy=(0, 0), xytext=(7, 8), textcoords="offset points",
        color="tab:orange", fontsize=9,
    )
    theta_ax.set_ylabel("Body pitch θ [rad]")
    theta_ax.set_xlabel("Time [s]")
    theta_ax.grid(True)

    z_cursor = z_ax.axvline(0.0, color="black", linestyle="--", alpha=0.6)
    theta_cursor = theta_ax.axvline(0.0, color="black", linestyle="--", alpha=0.6)

    current_frame = 0

    def update(frame):
        nonlocal current_frame
        current_frame = frame
        time = solution.t[frame]
        center_x = VEHICLE_SPEED * time - HALF_WHEELBASE
        front_x = center_x + HALF_WHEELBASE
        rear_x = center_x - HALF_WHEELBASE
        center_height = body_base + visual_scale * z[frame]
        pitch = theta[frame]

        # Positive theta raises the front and lowers the rear in this model.
        front_body_y = center_height + visual_scale * HALF_WHEELBASE * pitch
        rear_body_y = center_height - visual_scale * HALF_WHEELBASE * pitch
        reference_body_line.set_data([rear_x, front_x], [body_base, body_base])
        reference_points.set_data([rear_x, front_x], [body_base, body_base])
        body_line.set_data([rear_x, front_x], [rear_body_y, front_body_y])
        current_points.set_data([rear_x, front_x], [rear_body_y, front_body_y])
        z_indicator.set_data([center_x, center_x], [body_base, center_height])
        motion_label.set_position((center_x, max(front_body_y, rear_body_y) + 0.08))
        motion_label.set_text(f"z = {z[frame]:+.3f} m   θ = {pitch:+.4f} rad")

        front_road_y = road_height_at_position(front_x)
        rear_road_y = road_height_at_position(rear_x)
        front_wheel.center = (front_x, front_road_y + 0.12)
        rear_wheel.center = (rear_x, rear_road_y + 0.12)
        front_spring.set_data([front_x, front_x], [front_road_y + 0.24, front_body_y - 0.04])
        rear_spring.set_data([rear_x, rear_x], [rear_road_y + 0.24, rear_body_y - 0.04])

        car_ax.set_xlim(center_x - view_half_width, center_x + view_half_width)
        time_label.set_text(f"t = {time:.2f} s   z = {z[frame]:.3f} m   θ = {theta[frame]:.3f} rad")
        z_cursor.set_xdata([time, time])
        theta_cursor.set_xdata([time, time])
        z_point.set_data([time], [z[frame]])
        z_point_label.xy = (time, z[frame])
        z_point_label.set_text(f"z = {z[frame]:+.3f} m")
        theta_point.set_data([time], [theta[frame]])
        theta_point_label.xy = (time, theta[frame])
        theta_point_label.set_text(f"θ = {theta[frame]:+.4f} rad")
        label_offset = (-68, 8) if time > 0.8 * end_time else (7, 8)
        z_point_label.set_position(label_offset)
        theta_point_label.set_position(label_offset)
        return (
            reference_body_line, reference_points, body_line, current_points,
            z_indicator, motion_label, front_spring, rear_spring, front_wheel,
            rear_wheel, time_label, z_cursor, theta_cursor, z_point,
            z_point_label, theta_point, theta_point_label,
        )

    # Reserve space beneath the plots for interactive model controls.
    fig.subplots_adjust(left=0.12, right=0.96, top=0.91, bottom=0.22, hspace=0.55)
    spring_slider_ax = fig.add_axes([0.22, 0.135, 0.52, 0.025])
    damping_slider_ax = fig.add_axes([0.22, 0.075, 0.52, 0.025])
    reset_button_ax = fig.add_axes([0.80, 0.065, 0.14, 0.09])

    spring_slider = Slider(
        spring_slider_ax,
        "Spring k [N/m]",
        5_000,
        80_000,
        valinit=SPRING_STIFFNESS,
        valstep=500,
        valfmt="%0.0f",
    )
    damping_slider = Slider(
        damping_slider_ax,
        "Damping c [N·s/m]",
        100,
        10_000,
        valinit=DAMPING_COEFFICIENT,
        valstep=100,
        valfmt="%0.0f",
    )
    reset_button = Button(reset_button_ax, "Reset animation")

    def update_parameters(_value):
        nonlocal solution, z, theta
        solution = run_simulation(spring_slider.val, damping_slider.val)
        z = solution.y[0]
        theta = solution.y[2]

        z_line.set_ydata(z)
        theta_line.set_ydata(theta)
        z_ax.relim()
        z_ax.autoscale_view(scalex=False, scaley=True)
        theta_ax.relim()
        theta_ax.autoscale_view(scalex=False, scaley=True)

        body_center_heights = body_base + visual_scale * z
        front_heights = body_center_heights + visual_scale * HALF_WHEELBASE * theta
        rear_heights = body_center_heights - visual_scale * HALF_WHEELBASE * theta
        lowest_body_point = min(np.min(front_heights), np.min(rear_heights))
        highest_body_point = max(np.max(front_heights), np.max(rear_heights))
        car_ax.set_ylim(min(-0.22, lowest_body_point - 0.2), max(0.85, highest_body_point + 0.2))

        update(min(current_frame, len(solution.t) - 1))
        fig.canvas.draw_idle()

    def reset_animation(_event):
        animation.event_source.stop()
        animation.frame_seq = animation.new_frame_seq()
        update(0)
        fig.canvas.draw_idle()
        animation.event_source.start()

    spring_slider.on_changed(update_parameters)
    damping_slider.on_changed(update_parameters)
    reset_button.on_clicked(reset_animation)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(solution.t),
        interval=20,
        blit=False,
        repeat=False,
    )
    # Keep a reference alive for interactive Matplotlib backends.
    fig._car_animation = animation
    fig._model_widgets = (spring_slider, damping_slider, reset_button)
    fig.suptitle("2-DOF Half-Car Suspension Response")
    plt.show()


if __name__ == "__main__":
    main()
