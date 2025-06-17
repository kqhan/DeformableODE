# **Deformable Object Simulation with Isaac Lab**

This project demonstrates high-fidelity simulation of multiple randomized deformable objects interacting within a rigid container using [NVIDIA Isaac Lab](https://github.com/NVIDIA-Omniverse/IsaacLab).
---

## 🧠 Overview

The simulation consists of:

* Five deformable primitives (sphere, cuboid, cylinder, capsule, cone) dropped into a bucket.
* Realistic physical interactions and bouncing under gravity.
* RGB camera capture from a top-angled fixed viewpoint.
* Export of vertex positions and object metadata in `.h5` format.
* Optional image frame dumps for visual inspection or ML pipelines.

---

## 🛠️ System Requirements

### ✅ Minimum

| Component | Requirement                                                                    |
| --------- |--------------------------------------------------------------------------------|
| GPU       | NVIDIA RTX 30xx or higher with >16 GB GPU memory (RTX A6000, L40S recommended) |
| CPU       | 8-core (Intel i7 or AMD Ryzen 7 equivalent)                                    |
| RAM       | 32 GB minimum                                                                  |
| OS        | Ubuntu 22.04+ or Windows 10+                                                   |
| Disk      | 20 GB free for simulation output                                               |
| Display   | Required for Isaac Sim GUI (unless using headless mode)                        |

---

## 🚀 Installation Instructions

### 1. Install Isaac Lab

Isaac Lab is built on top of NVIDIA Omniverse Isaac Sim. Follow official setup instructions:

* [Isaac Lab Setup Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

Ensure you follow the office [quickstart guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html):

* Set environment variables as needed.
* Run Isaac Sim once to initialize the cache and dependencies.

---

### 2. Configure Python Environment

Isaac Lab uses the Python bindings of Isaac Sim. You must use the **Python 3.10 environment bundled with Isaac Sim**,
and install the extra dependencies for data generation.

```bash
python.sh -m pip install h5py
python.sh -m pip install pillow
```

---

## 📦 Running the Simulation

To run the deformable simulation:

```bash
./isaaclab.sh -p path/to/data.py --save_camera --enable_cameras
```

* `--save_camera`: (optional) saves RGB frames during simulation.
* Output files (camera frames and HDF5 vertex data) will be stored in `./deformable_simulation_data`.

---

## 📂 Output Structure

```
deformable_simulation_data_/
├── camera_images/
│   ├── run_00/
│   │   ├── frame_000001.png
│   │   ├── ...
├── deformable_vertices_YYYYMMDD_HHMMSS.h5
├── deformable_vertices_YYYYMMDD_HHMMSS_run01.h5
└── ...
```

### HDF5 Structure:

Each `.h5` file includes:

* **metadata/**: object types, colors, and simulation creation timestamp.
* **simulation\_data/step\_xxxxxx/**: per-timestep vertex positions and root positions.

---

## 🎯 Use Cases

* Deformable physics benchmarking
* Learning-based simulation prediction
* Physics-based animation
* Dataset generation for robotics and vision tasks

---


## 🧩 Acknowledgments

This work builds upon:

* [Isaac Lab](https://github.com/NVIDIA-Omniverse/IsaacLab)
* [NVIDIA Omniverse Isaac Sim](https://developer.nvidia.com/isaac-sim)

---

## 🛡 License

This project is licensed under the **BSD 3-Clause License**.
