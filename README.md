# Attention-Guided Jaw Bone Lesion Diagnosis in Panoramic Radiography
This framework trains a diagnosis model for panoramic radiograph data with
- a scale-invariant attention-guiding loss;
- a trapezoid augmentation method.
## Experiments
### Dataset
The function `load_dataset` in `dataset.py` can be replaced to one for custom datasets.
### Training
```
bash run_exp.sh
```