## EviCYP: in silico prediction of Cytochrome P450 substrates based on vector quantization and evidential deep learning
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.0-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)

## Environment Configuration

### (1) Feature Extraction
- **Python**: 3.10.18
- **Operating System**: Windows(CPU)
```text
torch==2.8.0
torchvision==0.23.0
torchtext==0.18.0
torch-geometric==2.6.1
torch-scatter==2.1.2+pt28cpu
pytorch-lightning==2.5.5
lightning-utilities==0.15.2
torchmetrics==1.8.2
pytorch-fast-transformers==0.4.0
x-transformers==2.7.6
vit-pytorch==1.11.7
transformers==4.48.1
rdkit==2025.3.6
chemprop==1.6.1
esm==3.2.1.post1
biopython==1.85
scikit-learn==1.7.2
```
### (2) Model Training
- **Python**: 3.9.20
- **Operating System**: Linux (with CUDA 12.4)
```text
numpy==1.26.4
torch==2.4.0
scikit-learn==1.6.1
```

## Acknowledgment of Open-Source Code Contributions
The code is based on the open-source repositories: [EviDTI](https://github.com/zhaoyanpeng208/EviDTI), [SVQDTI](https://github.com/jdcc2098/SVQDTI), [biomed-multi-view](https://github.com/BiomedSciAI/biomed-multi-view), [esm](https://github.com/evolutionaryscale/esm), many thanks to the authors!
