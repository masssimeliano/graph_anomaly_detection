# Graph Anomaly Detection Project

## 📂 Dataset Setup

Before running any experiments, make sure to extract the dataset files:

1. Go to the `graph_anomaly_detection/` directory.
2. Unzip the archive `datasets.zip` from the same directory containing the `.mat` files.

Then extract it so the structure becomes:

```
graph_anomaly_detection/
└── 
   datasets/
   └── small/
       └── Disney.mat
           books.mat
           ... 
   └── medium/
       └── Flickr.mat
           Reddit.mat
           ... 
   └── large/
       └── ...
       ... (other .mat files)
```

## 🚀 Running Models

Make sure all Python dependencies are installed, including:
- `torch==2.2.0+cpu`
- `torch-geometric==2.6.1`
- `torch_scatter==2.1.2+pt22cpu`
- `torch_sparse==0.6.18+pt22cpu`
- `pygod==1.1.0`
- `numpy==1.26.4`
- `scikit-learn==1.6.1`
- `scipy==1.15.2`
- `networkx==3.4.2`
- `matplotlib==3.10.1`

## 📜 Scripts

All training and evaluation scripts are located in the `scripts/` directory. These scripts are organized by model and dataset type.

Structure:

```
scripts/
├── read_and_show_metrics.py
├── train_baseline.py
│──  ...
```
