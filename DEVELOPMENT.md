# Development

## Requirements
- Conda (I use 4.9.1, though it shouldn't matter too much)
- CUDA driver in the 11.0 series (or 11.3) (this would be important, else you have to create the conda env yourself) 

Check which CUDA driver you have by going to 

With conda activated, cd to this directory and run `conda create --name deep-learning --file specs.txt`

It should work now, but test!
```bash
python
import torch
torch.cuda.is_available()
```


### tqdm
Progress bars are nice, but require a bit of finicky stuff


```bash
conda install -c conda-forge tqdm   # conda
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
