# 🎓 Bachelor Thesis: "Enhancing Graph Anomaly Detection Through Node Enrichment: Evaluating the Impact of Learned Representations and High-Level Semantic Features"

## 📘 About the Project

This project is part of my Bachelor's thesis at TU Dortmund University during the summer semester of 2025.  
The goal of this project is to implement and evaluate methods for **graph anomaly detection** using structural graph features and various anomaly detection models.

I explore techniques from recent graph learning research and compare the effectiveness of different models on publicly available datasets.

---

## 👨‍💻 Author

**Maksym Kravchenko**  
B.Sc. Informatiсs, Technical University Dortmund

---

## 🧪 Dataset Setup

Before running any experiments, make sure to extract the dataset files:

1. Go to the `graph_anomaly_detection/` directory.
2. Unzip the archive `datasets.zip` from the same directory containing the `.mat` files.

The expected structure after extraction:

```
graph_anomaly_detection/
└── datasets/
    ├── small/
    │   ├── Disney.mat
    │   ├── books.mat
    │   └── ...
    ├── medium/
    │   ├── Flickr.mat
    │   ├── Reddit.mat
    │   └── ...
    └── large/
        └── ...
```

---

### Dependencies

- `torch==2.2.0+cu121`
- `numpy==1.26.4`
- `scikit-learn==1.6.1`
- `scipy==1.15.2`
- `networkx==3.4.2`
- `matplotlib==3.10.1`
- `pyfglt==0.3.0`
- `pandas==2.2.3`
- `torch-geometric~=2.6.1`
- `seaborn~=0.13.2`

---

## 📜 Scripts

All training and evaluation scripts are located in the `scripts/` directory. These scripts are organized by model and dataset type.

```
scripts/
├── read_and_show_metrics.py
├── train_baseline.py
├── train_from_emd_baseline_with_alpha.py
├── ...
```

---

## 🏃 Run 

Run the ```check_train.py``` script to start calculating of all metrics for each datasets.
Run the ```read_and_show_metrics.py``` script to show and plot metrics for each dataset and
Use ```config.py``` to select the specific dataset

```
scripts/
├── check_train.py
```

---

## 🧠 Goal

The main objective is to detect anomalies in graph data using both traditional machine learning techniques and modern deep learning models.  
Structural node features will be extracted and used as input to anomaly detection algorithms.  
The performance of various models will be evaluated and compared across datasets of different sizes.

---

## 📚 References

The following literature and resources were used or referenced during the research:

1. PyGOD Documentation.
   https://pygod.org/
   
2. NetworkX Documentation.
   https://networkx.org/

3. Stanford CS224W Lecture Slides on Structural Features in Graphs (2021).  
   https://snap.stanford.edu/class/cs224w-2021/slides/02-tradition-ml.pdf

4. Dimitriadis, F., et al. (2021). PyFGLT: Python Library for Frequent Graph Pattern Features.  
   IEEE Paper: https://ieeexplore.ieee.org/abstract/document/9286205  
   Implementation: https://fcdimitr.github.io/pyfglt/

5. Datasets with labeled anomalies for Graph AD.
   https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection#Datasets

