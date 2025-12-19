DeepDPM

The intrinsically disordered regions (IDRs) of proteins play a crucial role in various biological
functions, with their embedded molecular recognition features (MoRFs) closely associated with
many important biological processes and diseases. Traditional MoRFs prediction methods are often
expensive and time-consuming, necessitating the development of a more precise and efficient
approach. To address this, we propose a novel deep learning model, DeepDPM, for MoRFs
prediction. DeepDPM combines the Prot-T5 and ESM-2 pre-trained models to extract sequence
information and structural information from proteins, respectively. It performs feature analysis by
integrating the BioWaveKAN module, mLSTM, optimized Transformer, and Mamba network, we
have designed an innovative dual-branch fusion module, BiLCrossAttention, which effectively
integrates temporal and spatial information using a weighted dual-channel cross-attention
mechanism. We have also innovatively optimized the Focal Loss function to tackle sample
imbalance, enhancing the model's ability to recognize key regions. On the newly constructed Test1
and Test2 datasets, DeepDPM outperformed five state-of-the-art methods. It achieved Matthew's
Correlation Coefficient (MCC) values of 77.39% and 69.08%, respectively, and Area Under the
Curve (AUC) values of 97.60% and 92.87%, respectively. These results demonstrate that DeepDPM
exhibits reliable and highly accurate performance in the Molecular Recognition Features (MoRFs)
prediction task.
<img width="480" height="543" alt="image" src="https://github.com/user-attachments/assets/a097bbc0-ec5e-4e12-85cc-d9078adf9521" />


# System requirement
- Python == 3.9.21
- NumPy == 1.26.4
- PyTorch (torch) == 2.6.0+cu124
- PyTorch Scatter (torch-scatter) == 2.1.2+pt26cu124
- PyTorch Cluster (torch-cluster) == 1.6.3+pt26cu124
- PyTorch Geometric (torch-geometric / pyg) == 2.6.1
- Biopython == 1.82
- fair-esm == 2.0.0
- sentencepiece == 0.2.0
- transformers == 4.49.0
- mamba-SSM == 2.2.4

# Description
The proposed DeepDPM method is implemented in Python based on the PyTorch framework for predicting molecular recognition features (MoRFs) in proteins.  
DeepDPM adopts a dual-branch network architecture and introduces a Bi-directional Local Cross Attention (BiLCrossAttention) module, which effectively fuses multi-source features and thereby significantly improves the accuracy of MoRF prediction.

# Datasets
DeepDPM provides several pre-split datasets in FASTA format for model training and evaluation, including:
- `Train.fasta`:  
  The main training set of DeepDPM, containing protein sequences used for model learning.
- `Training421.fasta`:  
  The classical TRAINING421 dataset, consisting of 421 protein sequences. It is one of the commonly used benchmark training sets in the MoRF prediction field.
- `Test1.fasta` and `Test2.fasta`:  
  Two newly constructed independent test sets, Test1 and Test2, used to systematically evaluate the generalization ability of DeepDPM under different data distributions. The MCC and AUC values reported in the paper are computed on these two datasets.
- `Test 419.fasta`:  
  The classical TEST419 dataset with 419 protein sequences, widely used for performance comparison in previous MoRF prediction methods. It serves as an important benchmark test set in this work.
- `Test 45.fasta` and `test49.fasta`:  
  Two additional independent test sets containing 45 and 49 protein sequences, respectively, which can be used to further verify the robustness of DeepDPM on diverse independent data.
All of the above are protein-sequence-level datasets used for training and testing DeepDPM on the MoRF prediction task.

# Feature
The `Extract_features` module contains two feature extraction scripts. By simply providing the path to the dataset, users can generate both ESM-2 and Prot-T5 features. All features generated in this study are stored in the `Feature` directory.

# Model
The model-related code is stored in the `model` directory, which contains three scripts corresponding to the first branch, the second branch, and the fusion module, respectively. The trained models are saved in the `save_model` directory.

# Test
The testing script is located in Test.py. Before use, you need to modify the model path in the script to point to the directory where the trained models are saved (save_model).
