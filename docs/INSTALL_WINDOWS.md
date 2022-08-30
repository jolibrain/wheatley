# Environnement python 

télécharger et exécuter l' [installeur](https://docs.conda.io/en/latest/miniconda.html)
    

# Pytorch

## Avec GPU NVIDIA

- installer CUDA 11.6 depuis les [archives](https://developer.nvidia.com/cuda-toolkit-archive)
- installer pytorch
dans un shell anaconda:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
          
## Sans GPU NVIDIA

installer pytorch : 
dans un shell anaconda:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
        
# Installer GIT

installeur [ici](https://git-scm.com/downloads)

# Installer diverses dépendances:

dans le shell anaconda:
```
conda install -c conda-forge jsonpatch visdom networkx plotly
pip install ortools kaleido stable_baselines3 sb3_contrib
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
```

# Installer wheatley:
dans le shell anaconda:
```
git clone https://gitlab.com/jolibrain/wheatley.git
cd wheatley
```

# lancer visdom:
toujours dans le shell
```
python -m visdom.server
```
autoriser python à créer des sockets

ensuite lancer wheatley suivant docs/examples.md
