# Deformable Neural ODE

This repository contains the implementation and resources for the **Deformable Neural ODE** project.

---

## Repository Structure

```

/
├── assets
│   ├── images
│   │   └── output2.png
│   ├── videos
│   │   └── run\_combine.mp4
│   └── Deformable\_NeurlODE.pdf        # Project paper or documentation
├── src
│   ├── data\_generation
│   │   ├── data.py                    # Data generation scripts
│   │   └── README.md
│   └── ODE\_model
│       ├── ODE.py                    # Neural ODE model implementation
│       └── ReadMe.md
├── static
│   ├── css                          # Stylesheets (Bulma, FontAwesome, custom)
│   └── js                           # JavaScript files (Bulma carousel/slider, FontAwesome, custom)
├── index.html                      # Main HTML page
├── README.md                      # This file
└── .gitignore                     # Git ignore rules

````

---

## Overview

This project implements a deformable neural ordinary differential equation (Neural ODE) model to simulate and analyze dynamic systems with deformable components.

The repository contains:

- **Source code** for data generation and model training/testing.
- **Static assets** for the web interface (HTML, CSS, JS).
- **Documentation and media** including the project paper (`Deformable_NeurlODE.pdf`), images, and demo videos.


## Getting Started

1. Clone the repository:
```bash
   git clone <repo-url>
   cd DeformableODE-main
```


2. Explore the `src/data_generation` folder to generate datasets.

3. Train or test the Neural ODE model inside the `src/ODE_model` directory.

4. Open `index.html` in a browser to view the web interface for visualizing results.

---

## Dependencies

* Python 3.x
* Required Python libraries: (Specify libraries if known, e.g., numpy, torch, matplotlib)
* Web dependencies included under `static/css` and `static/js` (Bulma CSS framework, FontAwesome icons, etc.)

---

## Documentation

* The detailed project paper is located at `assets/Deformable_NeurlODE.pdf`.
* Additional documentation inside `src/data_generation/README.md` and `src/ODE_model/ReadMe.md`.

---

## Contact

For questions or contributions, please open an issue or contact the maintainer.

---


