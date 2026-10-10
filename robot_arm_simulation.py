import sys
import tkinter as tk
from tkinter import ttk, messagebox
import roboticstoolbox as rtb
from spatialmath import SE3
from roboticstoolbox.backends.PyPlot import PyPlot
import numpy as np
import matplotlib.pyplot as plt

# --- 1. robot and trajectory kinematics ---
robot = rtb.DHRobot([
    rtb.RevoluteDH(d=0.5, a=0.0, alpha=np.pi/2),
    # Shoulder joint limit to prevent base collision and backward folding
    rtb.RevoluteDH(d=0.0, a=0.4, alpha=0.0, qlim=[-np.pi/3, np.pi/2]),
    # Elbow joint limit forced to negative to maintain an "elbow-up" natural stance
    rtb.RevoluteDH(d=0.0, a=0.3, alpha=0.0, qlim=[-np.pi, -0.01]) 
], name="3_Axis_Robot")

LETTERS = {
    'A': [(0.4, -0.15, 0.4), (0.4, 0.0, 0.8), (0.4, 0.15, 0.4), (0.4, 0.075, 0.6), (0.4, -0.075, 0.6)],
    'O': [(0.4, -0.15, 0.4), (0.4, 0.15, 0.4), (0.4, 0.15, 0.8), (0.4, -0.15, 0.8), (0.4, -0.15, 0.4)],
    'M': [(0.4, -0.15, 0.4), (0.4, -0.15, 0.8), (0.4, 0.0, 0.6), (0.4, 0.15, 0.8), (0.4, 0.15, 0.4)],
    'E': [(0.4, 0.15, 0.8), (0.4, -0.15, 0.8), (0.4, -0.15, 0.6), (0.4, 0.0, 0.6), (0.4, -0.15, 0.6), (0.4, -0.15, 0.4), (0.4, 0.15, 0.4)],
    'R': [(0.4, -0.15, 0.4), (0.4, -0.15, 0.8), (0.4, 0.15, 0.8), (0.4, 0.15, 0.6), (0.4, -0.15, 0.6), (0.4, 0.15, 0.4)]
}

def generate_trajectory(points, speed_level):
    num_steps = int(180 - (speed_level * 15))
    q_path = []
    # Initial pose updated to match negative elbow limit
    q0 = [0, 0, -0.5] 

    for i in range(len(points)-1):
        T1 = SE3(points[i][0], points[i][1], points[i][2])
        T2 = SE3(points[i+1][0], points[i+1][1], points[i+1][2])
        T_transition = rtb.ctraj(T1, T2, num_steps)

        solution = robot.ikine_LM(T_transition, q0=q0, mask=[1, 1, 1, 0, 0, 0])
        q_path.extend(solution.q)
        q0 = solution.q[-1] 

    return np.array(q_path)

def generate_infinity_trajectory(speed_level):
    num_steps = int(600 - (speed_level * 50)) 
    t = np.linspace(0, 2 * np.pi, num_steps)
    
    # Dimensions and center point safely constrained within reachable workspace
    x_target = 0.35 + 0.12 * np.sin(t)      
    y_target = 0.15 * np.sin(2 * t)        
    z_target = np.full_like(t, 0.1)       

    q_path = []
    q0 = [0, 0, -0.5]

    for xi, yi, zi in zip(x_target, y_target, z_target):
        target = SE3(xi, yi, zi)
        solution = robot.ikine_LM(target, q0=q0, mask=[1, 1, 1, 0, 0, 0])
        q_path.append(solution.q)
        q0 = solution.q

    return np.array(q_path)

# --- 2. asynchronous drawing loop ---
last_drawn_path = None
stop_flag = False  

def play_animation(q_path, selection_name):
    global stop_flag

    stop_flag = True
    root.update()
    stop_flag = False

    plt.close('all')
    env = PyPlot()
    env.launch()

    env.fig.set_size_inches(12, 6)
    env.fig.canvas.manager.set_window_title("Live Robot Simulation")

    env.ax.set_position([0.0, 0.1, 0.5, 0.8]) 
    env.ax.set_title("3-Axis Robot Animation")

    ax2d = env.fig.add_axes([0.55, 0.15, 0.4, 0.7])
    ax2d.set_title(f"{selection_name} - Live Trajectory Trace")
    
    ax2d.set_xlabel("Y Axis (m)")
    ax2d.set_ylabel("Z Axis (m)")

    if "Infinity" in selection_name:
        ax2d.set_xlim([-0.25, 0.25])
        ax2d.set_ylim([0.2, 0.5])
    else:
        ax2d.set_xlim([-0.3, 0.3])
        ax2d.set_ylim([0.0, 1.0])

    ax2d.grid(True)
    ax2d.set_aspect('equal', adjustable='box')

    env.add(robot)

    line2d, = ax2d.plot([], [], 'b-', linewidth=3, label='Past Trace')
    dot2d, = ax2d.plot([], [], 'ro', markersize=10, label='End-Effector (Current)')
    ax2d.legend()

    y_history, z_history = [], []

    lbl_status.config(text="Status: Drawing Live...", fg="green")

    for q in q_path:
        if stop_flag or not plt.fignum_exists(env.fig.number):
            break

        robot.q = q
        env.step(0.001) 

        T = robot.fkine(q)
        y_current = T.t[1]
        
        # Mapping X axis to Z axis in 2D plot for planar tracking
        if "Infinity" in selection_name:
            z_current = T.t[0] 
        else:
            z_current = T.t[2]

        y_history.append(y_current)
        z_history.append(z_current)
        line2d.set_data(y_history, z_history)
        dot2d.set_data([y_current], [z_current])

        plt.pause(0.01)
        root.update()

    if not stop_flag:
        lbl_status.config(text="Status: Waiting...", fg="black")

# --- 3. gui button commands ---
def start_drawing():
    global last_drawn_path
    selected = combo_selection.get()
    speed = slider_speed.get()

    lbl_status.config(text="Status: Calculating Matrices...", fg="blue")
    root.update()

    if selected == 'Infinity (On Ground)':
        q_path = generate_infinity_trajectory(speed)
    elif selected in LETTERS:
        q_path = generate_trajectory(LETTERS[selected], speed)
    else:
        messagebox.showwarning("Warning", "Please select a shape.")
        lbl_status.config(text="Status: Waiting...", fg="black")
        return

    last_drawn_path = q_path
    play_animation(q_path, selected)

def repeat_drawing():
    global last_drawn_path
    if last_drawn_path is not None:
        selected = combo_selection.get()
        play_animation(last_drawn_path, selected + " (Repeat)")
    else:
        messagebox.showinfo("Information", "No previous drawing in memory to repeat.")

def close_program():
    global stop_flag
    stop_flag = True
    plt.close('all')
    root.quit()
    root.destroy()
    sys.exit(0)

# --- 4. control panel design (GUI) ---
root = tk.Tk()
root.title("Robot Controller")
root.geometry("350x350")
root.protocol("WM_DELETE_WINDOW", close_program) 

tk.Label(root, text="Control Panel", font=("Arial", 12, "bold")).pack(pady=15)

tk.Label(root, text="1. Select Trajectory Shape:").pack()
options = list(LETTERS.keys()) + ['Infinity (On Ground)']
combo_selection = ttk.Combobox(root, values=options, state="readonly", width=20)
combo_selection.current(0)
combo_selection.pack(pady=5)

tk.Label(root, text="2. Adjust Drawing Speed:").pack(pady=(10,0))
tk.Label(root, text="(1: Industrial Slow  <-->  10: High Speed)", font=("Arial", 8)).pack()
slider_speed = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL, length=200)
slider_speed.set(4) 
slider_speed.pack(pady=5)

btn_draw = tk.Button(root, text="START DRAWING", font=("Arial", 10, "bold"), command=start_drawing, height=2)
btn_draw.pack(fill=tk.X, padx=40, pady=10)

btn_repeat = tk.Button(root, text="REPEAT LAST DRAWING", command=repeat_drawing)
btn_repeat.pack(fill=tk.X, padx=40, pady=5)

lbl_status = tk.Label(root, text="Status: Waiting...", font=("Arial", 9, "italic"))
lbl_status.pack(side=tk.BOTTOM, pady=10)

root.mainloop()