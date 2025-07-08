# BX242 Image Analysis Project

In this project, autogeneration tools are used in generate code, plotting, drafting, and modifying report.

## Directory Structure

```
bx242/
├── data/         # Butterfly image dataset  
├── module1/      # Image preprocessing and classification
├── module2/      # Image restoration and denoising
├── module3/      # Image quality assessment
└── report/       # Project documentation
```

## Source Code Structure

```
module1/
├── background_removal.py      # Advanced background removal with multiple algorithms
├── color_classification.py    # HUE-based butterfly classification
├── odd_detector_v1.py         # Anomaly detection using feature analysis
├── odd_detector_v2.py         # Enhanced odd butterfly detection
├── collection_display.py      # Visualization tools
└── module1.ipynb             # Jupyter notebook with experiments

module2/
├── denoiser.py               # Pre-trained U-Net denoiser wrapper
├── pnp_admm.py              # Plug-and-Play ADMM algorithm
├── pnp_red.py               # Plug-and-Play RED algorithm
├── blur_operators.py         # Gaussian and motion blur operators
├── inpainting_operators.py   # Image inpainting operators
├── conjugate_gradient.py     # Linear system solver
└── module2.ipynb           # Jupyter notebook with experiments

module3/
├── image_quality_assessment.py  # Comprehensive IQA metrics
└── module3.ipynb              # Jupyter notebook with experiments
```

## Running the Code

Run the Jupyter notebook inside each folder:

```bash
cd module1
jupyter notebook module1.ipynb
```

```bash
cd module2  
jupyter notebook module2.ipynb
```

```bash
cd module3
jupyter notebook module3.ipynb
```

## Data

The project uses butterfly images from the `data/` directory containing high-resolution photographs of various butterfly species for testing image processing algorithms.

## Dependencies

```bash
pip install opencv-python numpy torch torchvision scikit-image matplotlib scipy
``` 