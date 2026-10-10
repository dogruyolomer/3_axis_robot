import sys
import tkinter as tk
from tkinter import ttk, messagebox
import roboticstoolbox as rtb
from spatialmath import SE3
import numpy as np
import swift
import spatialgeometry as sg

# --- 1. industrial robot model for visualization ---
# the mathematical system is constrained to 3 DOF, with the UR5 s
# additional joints locked to maintain a fixed orientation of the end effector.
try:
    robot = rtb.models.URDF.UR5()
except AttributeError:
    robot = rtb.models.UR5()

# standby position
q_home_ur5 = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
robot.q = q_home_ur5

# target coordinates scaled for the workspace reach
LETTERS = {
    'O': [(0.35, -0.15, 0.3), (0.35, 0.15, 0.3), (0.35, 0.15, 0.6), (0.35, -0.15, 0.6), (0.35, -0.15, 0.3)],
    'M': [(0.35, -0.15, 0.3), (0.35, -0.15, 0.6), (0.35, 0.0, 0.45), (0.35, 0.15, 0.6), (0.35, 0.15, 0.3)],
    'E': [(0.35, 0.15, 0.6), (0.35, -0.15, 0.6), (0.35, -0.15, 0.45), (0.35, 0.0, 0.45), (0.35, -0.15, 0.45), (0.35, -0.15, 0.3), (0.35, 0.15, 0.3)],
    'R': [(0.35, -0.15, 0.3), (0.35, -0.15, 0.6), (0.35, 0.15, 0.6), (0.35, 0.15, 0.45), (0.35, -0.15, 0.45), (0.35, 0.15, 0.3)]
}

# --- 2. trajectory planning & 3 axis reduction ---
def generate_trajectory(points, speed_level):
    step_count = int(400 - (speed_level * 35)) 
    path_data = [] 
    
    # restricting movement to x, y, z axes by locking orientation
    T_home = robot.fkine(q_home_ur5)
    R_fixed = T_home.R 
    
    home_path = rtb.jtraj(robot.q, q_home_ur5, 40)
    for q in home_path.q:
        path_data.append((q, False))
        
    T_start = SE3.Rt(R_fixed, [points[0][0], points[0][1], points[0][2]])
    q_start = robot.ikine_LM(T_start, q0=q_home_ur5).q
    
    start_path = rtb.jtraj(q_home_ur5, q_start, 40)
    for q in start_path.q:
        path_data.append((q, False))
        
    q0 = q_start
    for i in range(len(points)-1):
        T1 = SE3.Rt(R_fixed, [points[i][0], points[i][1], points[i][2]])
        T2 = SE3.Rt(R_fixed, [points[i+1][0], points[i+1][1], points[i+1][2]])
        T_line = rtb.ctraj(T1, T2, step_count)
        
        for T in T_line:
            solution = robot.ikine_LM(T, q0=q0) 
            path_data.append((solution.q, True))
            q0 = solution.q 
            
    return path_data

def trajectory_ground_infinity(speed_level):
    step_count = int(2000 - (speed_level * 150)) 
    t = np.linspace(0, 2 * np.pi, step_count)
    
    x_target = 0.35 + 0.10 * np.sin(t)      
    y_target = 0.15 * np.sin(2 * t)        
    z_target = np.full_like(t, 0.20)   
    
    path_data = []
    
    T_home = robot.fkine(q_home_ur5)
    R_fixed = T_home.R
    
    home_path = rtb.jtraj(robot.q, q_home_ur5, 40)
    for q in home_path.q:
        path_data.append((q, False))
        
    T_start = SE3.Rt(R_fixed, [x_target[0], y_target[0], z_target[0]])
    q_start = robot.ikine_LM(T_start, q0=q_home_ur5).q
    
    start_path = rtb.jtraj(q_home_ur5, q_start, 40)
    for q in start_path.q:
        path_data.append((q, False))
        
    q0 = q_start
    for xi, yi, zi in zip(x_target, y_target, z_target):
        target = SE3.Rt(R_fixed, [xi, yi, zi])
        solution = robot.ikine_LM(target, q0=q0)
        path_data.append((solution.q, True))
        q0 = solution.q
        
    return path_data

# --- 3. swift simulation and visual effects ---
env = swift.Swift()
env.launch(realtime=True) 
env.add(robot)

MAX_PAINT = 800
paint_pool = []
for _ in range(MAX_PAINT):
    point = sg.Sphere(radius=0.008, pose=SE3(0, 0, -5), color='#e74c3c')
    env.add(point)
    paint_pool.append(point)

last_drawn_path = None
is_drawing = False
stop_flag = False

def play_animation(path_data):
    global stop_flag, paint_pool, is_drawing
    
    stop_flag = False
    
    # hide all previous trajectory points
    for point in paint_pool:
        point.T = SE3(0, 0, -5).A
    
    lbl_status.config(text="System Status: EXECUTING TRAJECTORY...", fg="#f39c12")
    
    paint_frequency = max(1, len(path_data) // 350) 
    paint_counter = 0
    
    for i, (q, is_drawing_flag) in enumerate(path_data):
        if stop_flag or not root.winfo_exists():
            lbl_status.config(text="System Status: MOTION ABORTED", fg="#e74c3c")
            is_drawing = False
            return
            
        robot.q = q
        
        # render visual spheres during active trace
        if is_drawing_flag and i % paint_frequency == 0 and paint_counter < MAX_PAINT: 
            T_current = robot.fkine(q)
            paint_pool[paint_counter].T = T_current.A
            paint_counter += 1
        
        env.step(0.01) 
        root.update()
        
    lbl_status.config(text="System Status: IDLE", fg="#27ae60")
    is_drawing = False

# --- 4. gui button commands ---
def start_drawing():
    global last_drawn_path, is_drawing
    
    if is_drawing:
        return
        
    is_drawing = True
    selected = combo_selection.get()
    speed = slider_speed.get()
    
    lbl_status.config(text="System Status: CALCULATING KINEMATICS...", fg="#2980b9")
    root.update()
    
    if selected == 'Infinity (Ground)':
        path_data = trajectory_ground_infinity(speed)
    elif selected in LETTERS:
        path_data = generate_trajectory(LETTERS[selected], speed)
    else:
        messagebox.showwarning("Warning", "Please select a valid trajectory.")
        lbl_status.config(text="System Status: IDLE", fg="#27ae60")
        is_drawing = False
        return

    last_drawn_path = path_data
    play_animation(path_data)

def abort_motion():
    global stop_flag
    if is_drawing:
        stop_flag = True

def repeat_drawing():
    global last_drawn_path, is_drawing
    if is_drawing:
        return
        
    if last_drawn_path is not None:
        is_drawing = True
        play_animation(last_drawn_path)
    else:
        messagebox.showinfo("Information", "No previous trajectory found in memory.")

def close_program():
    global stop_flag
    stop_flag = True
    try:
        env.close() 
    except:
        pass
    root.quit()
    root.destroy()
    sys.exit(0)

# --- 5. control panel interface (GUI) ---
root = tk.Tk()
root.title("Robot Control Panel")
root.geometry("350x420")
root.protocol("WM_DELETE_WINDOW", close_program) 

tk.Label(root, text="Control Panel", font=("Arial", 12, "bold")).pack(pady=15)

tk.Label(root, text="1. Select Trajectory:").pack()
options = list(LETTERS.keys()) + ['Infinity (Ground)']
combo_selection = ttk.Combobox(root, values=options, state="readonly", width=20)
combo_selection.current(0)
combo_selection.pack(pady=5)

tk.Label(root, text="2. Set Execution Speed:").pack(pady=(10,0))
tk.Label(root, text="(1: Industrial Slow  <-->  10: High Speed)", font=("Arial", 8)).pack()
slider_speed = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL, length=200)
slider_speed.set(4) 
slider_speed.pack(pady=5)

btn_draw = tk.Button(root, text="START EXECUTION", font=("Arial", 10, "bold"), command=start_drawing, height=2)
btn_draw.pack(fill=tk.X, padx=40, pady=(10, 5))

btn_abort = tk.Button(root, text="ABORT MOTION", font=("Arial", 10, "bold"), fg="red", command=abort_motion, height=2)
btn_abort.pack(fill=tk.X, padx=40, pady=5)

btn_repeat = tk.Button(root, text="REPEAT LAST TRAJECTORY", command=repeat_drawing)
btn_repeat.pack(fill=tk.X, padx=40, pady=(5, 10))

lbl_status = tk.Label(root, text="System Status: IDLE", font=("Arial", 10, "bold"), fg="#27ae60")
lbl_status.pack(side=tk.BOTTOM, pady=15)

root.mainloop()