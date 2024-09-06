### Hi, welcome to my numerical analysis projects.

This repository contains a collection of **Jupyter Notebook (`.ipynb`)** files focused on numerical analysis techniques. Each notebook explores various computational methods for solving mathematical problems, with interactive code, explanations, and visualizations.

## Table of Contents
- [Overview](#overview)
- [Projects](#projects)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Notebooks](#running-the-notebooks)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Numerical analysis focuses on developing algorithms to approximate solutions for problems in continuous mathematics. These Jupyter Notebooks explore a range of topics, including:
- Solving linear and nonlinear equations
- Numerical integration and differentiation
- Eigenvalue problems
- Approximation methods
- Differential equations and more

Each notebook provides code implementation, mathematical derivations, and visualizations to help users understand the underlying numerical methods.
---

## Projects

Below are some of the topics covered in this repository:
1. **Solving Linear Systems**
2. **Numerical Differentiation and Integration**
3. **Optimization and Root Finding**
4. **Sparse Matrices and Eigenvalue Problems**
5. **Approximation Methods**
6. **Numerical Solutions of Differential Equations**
---

## Requirements

To run both of these notebooks, the following Python packages are required:

```python
%matplotlib inline
import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy
import operator
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import plotly.offline as py
import scipy
import scipy.linalg as linalg
import sympy as sp
import time
import timeit
import warnings
from scipy.optimize import curve_fit
from functools import reduce
from IPython.display import Image, display
from sympy.solvers import solve
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from ipywidgets import interact, SelectionSlider
from plotly.subplots import make_subplots
```

These packages are used for:
- **Matplotlib**: Visualization of data and results.
- **NumPy**: Numerical computations.
- **Pandas**: Data manipulation and analysis.
- **Plotly**: Interactive visualizations.
- **SciPy**: Advanced scientific computations, including optimization, integration, and sparse matrix operations.
- **SymPy**: Symbolic mathematics.
- **IPython Widgets**: Interactive elements for notebook presentations.

---

## Installation

1. **Clone the repository (or download the notebooks)**:
   ```bash
   git clone https://github.com/kyracho/numerical-analysis-projects.git
   cd numerical-analysis-projects
   ```

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

Alternatively, you can install the required packages manually:
   ```bash
   pip install matplotlib numpy pandas plotly scipy sympy ipywidgets
   ```

---

## Running the Notebooks

To run the Jupyter Notebooks locally, follow these steps:

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the desired `.ipynb` file** from the list in the Jupyter interface.

3. **Interact with the code and visualizations**. Many notebooks contain interactive widgets and plots, so be sure to run each cell and explore the outputs.
