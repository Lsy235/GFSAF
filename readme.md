# The process of running this project. (Source code of AAAI2026 paper 'Views Attention Fusion of Granular-ball Fuzzy Representations Split for Improving Multi-View Clustering')

## Main environment:

There are the main environment of this project:

    - numpy                      1.23.5
	- opencv-python              4.6.0.66
    - pandas                     2.0.3
	- timm                       1.0.11
	- torch                      2.4.1+cu124
	- torch-geometric            2.6.1
	- torch_scatter              2.1.2+pt24cu124
	- torch_sparse               0.6.18+pt24cu124
    - tqdm                       4.66.5
	- torchvision                0.19.1+cu124

## Steps:

There are more details of running this project:

    - Step 1. Download the databases into the data folder. (We provide the WebKB database for testing our code)
    - Step 2. run the run.sh script to train and evaluate the model. Or run the command `python train.py --dataset "WebKB" --epochs 100 --iteration 5` in the terminal.