**DL-based Pneumonia Detection**

Deep learning–based object detection system for identifying pneumonia opacities in chest X-ray images.

This project was developed as part of the RSNA Pneumonia Detection Challenge, using RetinaNet with a ResNet-101 backbone to detect and localize pneumonia regions in DICOM X-rays.

**1. Project Overview:**
Pneumonia is a serious respiratory infection and a leading cause of mortality worldwide.
Manual diagnosis using X-rays is time-intensive and prone to variability.
We built a RetinaNet object detector that automates pneumonia detection, reduces clinical workload, and enables faster, more consistent diagnosis.

**2. Methodology:**
**a) Model Architecture:**
Backbone: ResNet-101 pre-trained on ImageNet
Feature Pyramid Network (FPN): Multi-scale feature extraction
Anchors: 5 scales × 3 aspect ratios (0.5, 1.0, 2.0)
Loss: Focal Loss (classification) + Smooth L1 (bounding box regression)
Optimizer: AdamW (lr=5e-5, weight decay=1e-4)
Scheduler: StepLR (decay factor 0.5 every 10 epochs)
Early Stopping: Patience = 7 epochs

**b) Dataset:**
Source: RSNA Pneumonia Detection Challenge (Kaggle)
Training: 26,684 chest X-rays (6,012 with pneumonia annotations)
Split: 80% training, 20% validation
Image Size: Resized to 768×768 for GPU efficiency
Preprocessing: DICOM → grayscale, normalized to [0,1], converted to PyTorch tensors

**3. Results:**
mAP@0.5: 0.6205
Recall@100: 0.7614
Validation Loss: ~0.71 (best model)
Visualization: Bounding boxes drawn on predictions (green = model, red = ground truth)

The model demonstrated strong generalization on unseen X-rays, successfully detecting pneumonia-related opacities.
**Repository Structure:**
MyProject.py          # Model training script

test.ipynb            # Notebook for evaluation & visualization

logs/                 # Training logs

test_predictions/     # Sample predictions on test X-rays

README.md             # Project description

requirements.txt      # Dependencies

**Usage:**
**1. Clone the repo:** git clone https://github.com/manjusha0609/DL-based-Pneumonia-Detection.git
navigate- cd DL-based-Pneumonia-Detection

**2. Install dependencies:** You can install everything from requirements.txt:
pip install -r requirements.txt

**3. Train the mode:**  python MyProject.py

**4. Run evaluation:**
**Open and execute:** jupyter notebook test.ipynb
