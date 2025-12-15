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
- python  3.9.21
- numpy  1.26.4 
- pytorch  1.13.1  
- pytorch-scatter  2.1.1  
- pytorch-cluster  1.6.1  
- pyg  2.3.0  
- biopython  1.81  
- fair-esm  2.0.0  
- dllogger  1.0.0  
- openfold  1.0.1  
- sentencepiece  0.1.99  
- transformers  4.30.1
