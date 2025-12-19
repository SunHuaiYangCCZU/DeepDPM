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
