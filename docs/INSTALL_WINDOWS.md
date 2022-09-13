# Environnement python 

télécharger et exécuter [l'installeur](https://docs.conda.io/en/latest/miniconda.html). 
    
le mettre à jour :
dans un shell [ana|mini]conda: 
```
conda update --all -c conda-forge
```
    
# Environnement C++

télécharger et installer la [librairie sandard C++](https://aka.ms/vs/17/release/vc_redist.x86.exe)

# OpenCV

opencv pour python disponible sous [ana|mini]conda suppose que openCV a été compilé et installé depuis les sources, ce qui est assez compliqué à faire. 
Comme workaround, on propose la démarche suivante:

- installer opencv-python depuis [ana|mini]conda:
dans un shell [ana|mini]conda
```
conda install -c conda-forge opencv
```

- installer openCV pré-compilé pour windows :
  - télécharger depuis le [repo](https://sourceforge.net/projects/opencvlibrary/files/4.6.0/opencv-4.6.0-vc14_vc15.exe/download)
  - décompresser 

- recopier la librairie pour python dans l'environnement [ana|mini]conda
on appelle DIR_OPENCV l'endroit où est décompressé opencv
on appelle DIR_CONDA l'endroit où est installé [ana|mini]conda
depuis DIR_OPENCV\build\python\cv2\python-3.9
copier cv2.cp39-win_amd64.pyd vers DIR_CONDA\Lib\site-packages

- recopier la librairie opencv elle-même vers un endroit trouvable par la partie de la librairie pour python
depuis DIR_OPENCV\build\x64\vc15\bin
copier opencv_world460.dll et opencv_world460.pdb
vers DIR_CONDA


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
si besoin autoriser python à créer des sockets

ensuite lancer wheatley suivant docs/examples.md
