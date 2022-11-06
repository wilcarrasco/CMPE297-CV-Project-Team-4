The Jupyter Notebook contains a view of the total inference pipeline: 2D_detection.ipynb

Training was done separately from the notebook given that the total training time was more than a week! But if you want to manually train you can follow the directions in the Notebook.

For convenience, the trained models are supplied: my_models/yolov5s.pt | my_models/yolov7.pt


Additionally, all required libraries are included to run inference on test images located here: test_images

And the outputs were saved here: inferrence_tryout/YoloV5 | inferrence_tryout/YoloV7


This Project assumes that an environment is setup on the SJSU HPC COE and any inference is ran on a GPU node.


&nbsp;
## Connecting to SJSU HPC:
Connect via VPN to the SJSU Network. For a complete guide pleaser reference: __[HPC Environment Setup Doc](https://docs.google.com/document/d/1GCI5YuP8vpEHGYMngUfdjhMLOrXFpfwJeocSSP7Cl6c/edit)__

SSH into the HPC head node, provided you have access, using a specific port that can be used to create a SSH tunnel for running Jupyter Notebooks:

```bash
 ssh -L [PORT]:localhost:[PORT] [SJSU ID]@coe-hpc1.sjsu.edu
```

 <b>Note:</b> Port is a port number that is not in use (e.x 16969)
 SJSU ID is your student ID

Once logged in check the available modules to load:

```bash
 [@coe-hpc1 ~] module avail

 ```
 Load the python3 and cuda10.1 module. First check current python version

 ```bash
[@coe-hpc1 ~]$ python -V
Python 2.7.5
[@coe-hpc1 ~]$ module load python3/3.7.0 cuda/10.1
[@coe-hpc1 ~]$ python -V
Python 3.7.0
```

Now that the cuda module and python module is loaded create virtual environment:

```bash
[@coe-hpc1 ~]$ virtualenv --system-site-packages -p python3.7 ./venvpy37cu10
```

 <b>Note:</b> If the virtualenv is not available, you can also use python3 -m venv ./xxx to create the virtual environment

Activate the python virtual environment
```bash
$ source ./venvpy37cu10/bin/activate
```

Now pip install the following packages:

```bash
pip install jupyterlab
pip install --user ipykernel
pip install tensorflow==2.3.0


```

Add a new kernel to the Jupyter lab:
```bash
python -m ipykernel install --user --name=venvpy37cu10
```

```bash
$ python -c 'import tensorflow as tf; print(tf.__version__); print("GPU Available: ", tf.test.is_gpu_available())'
2.3.0
GPU Available:  False (note: there is no GPU in the head node)
$pip show tensorflow
Name: tensorflow
Version: 2.3.0
```

Request a GPU node 

```bash
[@coe-hpc1 ~]$ srun -p gpu --gres=gpu --pty /bin/bash
```

<b>Note:</b> the GPU nodes do not have internet/external access. Download of source any external materials on the head node prior to requesting a GPU node.


Check for video card acces and cuda driver:

```bash
nvidia-smi
nvcc --version
```


2D Vision additional depenencies:

pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

<b>IMPORTANT:</b> Please review the HPC setup documentation to understand the how the HPC environment is set up! Training will take days so it is advise to setup a screen SSH session and reserver a node for at least 48Hrs. These steps may not be all inclusive so please ask if you are having issues