## Umgebung aus environment.yml erstellen
conda env create -f environment.yml
(falls das erste nicht geht, das vielleicht mal ausprobieren: conda-env create -n new_env -f=\path\to\base.yml)

## Umgebung

conda create -n aiarena python=3.6

conda install -c anaconda tensorflow-gpu 

conda install -c conda-forge keras 

conda install -c anaconda pyqt

conda install -c anaconda pyqtgraph

conda install -c conda-forge pynput oder pip install pynput

pip install ray

pip install tqdm


for tensorflow-gpu install latest Cuda drivers


### Genaue Reihenfolge:

1. Anaconda Community Version installieren
2. conda create -n aiarena python=3.6
3. pip install tensorflow==2.2.0
4. pip install keras==2.3.1
5. conda install -c anaconda pyqt
6. conda install -c anaconda pyqtgraph
7. pip install pynput
8. pip install ray
9. pip install tqdm


## Anaconda Umgebung yml File erstellen: 
conda env export > environment.yml
