### Acknowledgement

This repo is heavily borrowed from [IntraDA](https://github.com/feipan664/IntraDA). We sincerely thank the authors for providing such a great framework.


## Pre-requsites
* Python 3.7
* Pytorch >= 0.4.1
* CUDA 9.0 or higher

## Installation
0. Clone the repo:
```bash
$ git clone https://github.com/LANMNG/RoadDA.git
$ cd RoadDA
```

1. Install OpenCV if you don't already have it:
```bash
$ conda install -c menpo opencv
```
if it doesn't work, please try to use conda pip
```bash
$ which pip    # should be $HOME/anaconda3/bin/pip, be sure to use conda pip
$ pip install opencv-python 
```

2. Install ADVENT submodule and the dependices using pip:
if you use 
```bash
$ pip install -e <root_dir/ADVENT>
```
With this, you can edit the ADVENT code on the fly and import function 
and classes of ADVENT in other project as well.

### Datasets
The format of the road data and the construction are suggested to be the same as the GTA5 and Cityscapes in [IntraDA](https://github.com/feipan664/IntraDA).


### Training
Our training environment is based on pytorch 0.4.1 and CUDA 9.0. To reach to the comparable performance you may need to train a few times.
By default, logs and snapshots are stored in ```<root_dir>/experiments``` with this structure:
```bash
<root_dir>/ADVENT/experiments/logs
<root_dir>/ADVENT/experiments/snapshots
```

**Step 1.** Conduct inter-domain adaptation by training [ADVENT](https://github.com/valeoai/ADVENT.git): 
```bash
$ cd <root_dir>/ADVENT/advent/scripts
$ python train.py --cfg ./config/advent.yml 
$ python train.py --cfg ./config/advent.yml --tensorboard % using tensorboard
```
After inter-domain training, it is needed to get best IoU iteration by runing:
```bash
$ cd <root_dir>/ADVENT/advent/scripts
$ python test.py --cfg ./config/advent.yml
```
The best IoU iteration ```BEST_ID``` will be a parameter to **step 2**. 

**Step 2.** Entropy-based ranking to split training set of target data into easy split and hard split: 
```bash
$ cd <root_dir>/entropy_rank
$ python entropy.py --best_iter BEST_ID --normalize False --lambda1 0.7 
```
You will see the pseudo labels generated in ```color_masks```, the easy split file names in ```easy_split.txt```, and the hard split file names in ```hard_split.txt```.

**Step 3.** Conduct intra-domain adaptation by runing:
```bash
$ cd <root_dir>/intrada
$ python train.py --cfg ./intrada.yml
$ python train.py --cfg ./intrada.yml --tensorboard % using tensorboard
```
After intra-domain training, it is needed to get best IoU iteration by runing:
```bash
$ cd <root_dir>/intrada
$ python test.py --cfg ./intrada.yml
```

**Step 4.** Conduct the self-training stage by repeating the step 2 and step 3 for several times.


## Testing
After the self-training stage, we get the performance by running:
```bash
$ cd <root_dir>/intrada
$ python test.py --cfg ./intrada.yml
```