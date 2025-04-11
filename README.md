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
   └── Disney.mat
       books.mat
       ... (other .mat files)
```

## 🚀 Running Models

Make sure all Python dependencies are installed, including:
- `torch~=2.5.1`
- `matplotlib~=3.10.1`
- `networkx~=3.4.2`
- `pygod~=1.1.0`
- `scikit-learn~=1.6.1`
- `torch-geometric~=2.6.1`
- `scipy~=1.15.2`
- `pyfglt~=0.3.0`
