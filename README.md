# Deep Learning Project Template
This template offers a lightweight yet functional project template for various deep learning projects. 
The template assumes [PyTorch](https://pytorch.org/) as the deep learning framework.
However, one can easily transfer and utilize the template to any project implemented with other frameworks.


## Table of Contents
- [Getting Started](#getting-started)
- [Template Layout](#template-layout)
- [Authors](#authors)
- [License](#license)


## Getting Started 


## Template Layout

```text
.
├── LICENSE.md
├── README.md
├── makefile            # makefile for various commands (install, train, pytest, mypy, lint, etc.) 
├── mypy.ini            # MyPy type checking configurations
├── pylint.rc           # Pylint code quality checking configurations
├── pyproject.toml      # Poetry project and environment configurations
│
├── data
│   ├── ...             # data reference files (index, readme, etc.)
│   ├── raw             # untreated data directly downloaded from source
│   ├── interim         # intermediate data processing results
│   └── processed       # processed data (features and targets) ready for learning
├── docs                # documentation files (*.txt, *.doc, *.jpeg, etc.)
├── logs                # logs for deep learning experiments
├── models              # saved models with optimizer states
├── notebooks           # Jupyter Notebooks (mostly for data processing and visualization)
│── src    
│   ├── data            # data processing classes, functions, and scripts
│   ├── evaluations     # evaluation classes and functions (metrics, visualization, etc.)
│   ├── experiments     # experiment configuration files
│   ├── modules         # activations, layers, modules, and networks (subclass of torch.nn.Module)
│   └── utilities       # other useful functions and classes
└── tests               # unit tests module for ./src
```


## Authors
* Xiaotian Duan (Email: xduan7 at gmail.com)


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for more details.

