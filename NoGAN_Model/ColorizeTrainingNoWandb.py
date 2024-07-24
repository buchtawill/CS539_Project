#!/usr/bin/env python
# coding: utf-8

# ## Stable Model Training with monitoring through Weights & Biases

# #### NOTES:  
# * This is "NoGAN" based training, described in the DeOldify readme.
# * This model prioritizes stable and reliable renderings.  It does particularly well on portraits and landscapes.  It's not as colorful as the artistic model.
# * Training with this notebook has been logged and monitored through [Weights & Biases](https://www.wandb.com/). Refer to [W&B Report](https://app.wandb.ai/borisd13/DeOldify/reports?view=borisd13%2FDeOldify).
# * It is **highly** recommended to use a 11 Go GPU to run this notebook. Anything lower will require to reduce the batch size (leading to moro instability) or use of "Large Model Support" from IBM WML-CE (not so easy to setup). An alternative is to rent ressources online.

# In[ ]:


# Install W&B Callback
#!pip install wandb

#NOTE:  This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

import os
import fastai
from fastai import *
from fastai.vision import *
from fastai.vision.gan import *
from deoldify.generators import *
from deoldify.critics import *
from deoldify.dataset import *
from deoldify.loss import *
from deoldify.save import *
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm
#import wandb
#from wandb.fastai import WandbCallback


# Setup

# Set up W&B: checks user can connect to W&B servers
# Note: set up API key the first time
#wandb.login()

path = Path('training_data')
path_hr = path/'color'
path_lr = path/'bandw'

proj_id = 'StableModel'

gen_name = proj_id + '_gen'
pre_gen_name = gen_name + '_0'
crit_name = proj_id + '_crit'

name_gen = proj_id + '_image_gen'
path_gen = path/name_gen

nf_factor = 2
pct_start = 1e-8


# Iterating through the dataset

# The dataset is very large and it would take a long time to iterate through all the samples at each epoch.
# We use custom samplers in order to limit epochs to subsets of data while still iterating slowly through the entire dataset (epoch after epoch). This let us run the validation loop more often where we log metrics as well as prediction samples on validation data.

# Reduce quantity of samples per training epoch
# Adapted from https://forums.fast.ai/t/epochs-of-arbitrary-length/27777/10

@classmethod
def create(cls, train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None, path:PathOrStr='.', bs:int=64,
            val_bs:int=None, num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None,
            device:torch.device=None, collate_fn:Callable=data_collate, no_check:bool=False, sampler=None, **dl_kwargs)->'DataBunch':
    "Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`. Passes `**dl_kwargs` to `DataLoader()`"
    datasets = cls._init_ds(train_ds, valid_ds, test_ds)
    val_bs = ifnone(val_bs, bs)
    if sampler is None: sampler = [RandomSampler] + 3*[SequentialSampler]
    dls = [DataLoader(d, b, sampler=sa(d), drop_last=sh, num_workers=num_workers, **dl_kwargs) for d,b,sh,sa in
            zip(datasets, (bs,val_bs,val_bs,val_bs), (True,False,False,False), sampler) if d is not None]
    return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

ImageDataBunch.create = create
ImageImageList._bunch = ImageDataBunch

class FixedLenRandomSampler(RandomSampler):
    def __init__(self, data_source, epoch_size):
        super().__init__(data_source)
        self.epoch_size = epoch_size
        self.not_sampled = np.array([True]*len(data_source))
    
    @property
    def reset_state(self): self.not_sampled[:] = True
        
    def __iter__(self):
        ns = sum(self.not_sampled)
        idx_last = []
        if ns >= len(self):
            idx = np.random.choice(np.where(self.not_sampled)[0], size=len(self), replace=False).tolist()
            if ns == len(self): self.reset_state
        else:
            idx_last = np.where(self.not_sampled)[0].tolist()
            self.reset_state
            idx = np.random.choice(np.where(self.not_sampled)[0], size=len(self)-len(idx_last), replace=False).tolist()
        self.not_sampled[idx] = False
        idx = [*idx_last, *idx]
        return iter(idx)
    
    def __len__(self):
        return self.epoch_size

def get_data(bs:int, sz:int, keep_pct=1.0, random_seed=None, valid_pct=0.2, epoch_size=1000):
    
    # Create samplers
    train_sampler = partial(FixedLenRandomSampler, epoch_size=epoch_size)
    samplers = [train_sampler, SequentialSampler, SequentialSampler, SequentialSampler]

    return get_colorize_data(sz=sz, bs=bs, crappy_path=path_lr, good_path=path_hr, random_seed=random_seed,
                             keep_pct=keep_pct, samplers=samplers, valid_pct=valid_pct)

# Function modified to allow use of custom samplers
def get_colorize_data(sz:int, bs:int, crappy_path:Path, good_path:Path, random_seed:int=None,
        keep_pct:float=1.0, num_workers:int=8, samplers=None, valid_pct=0.2, xtra_tfms=[])->ImageDataBunch:
    src = (ImageImageList.from_folder(crappy_path, convert_mode='RGB')
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(valid_pct, seed=random_seed))
    data = (src.label_from_func(lambda x: good_path/x.relative_to(crappy_path))
        .transform(get_transforms(max_zoom=1.2, max_lighting=0.5, max_warp=0.25, xtra_tfms=xtra_tfms), size=sz, tfm_y=True)
        .databunch(bs=bs, num_workers=num_workers, sampler=samplers, no_check=True)
        .normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data

# Function to limit amount of data in critic
def filter_data(pct=1.0):
    def _f(fname):
        if 'test' in str(fname):
            if np.random.random_sample() > pct:
                return False
        return True
    return _f

def get_crit_data(classes, bs, sz, pct=1.0):
    src = ImageList.from_folder(path, include=classes, recurse=True).filter_by_func(filter_data(pct)).split_by_rand_pct(0.1)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=sz)
           .databunch(bs=bs).normalize(imagenet_stats))
    return data

def create_training_images(fn,i):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn).convert('LA').convert('RGB')
    img.save(dest)  
    
def save_preds(dl):
    i=0
    names = dl.dataset.items    
    for b in tqdm(dl):
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1
    
def save_gen_images(keep_pct):
    if path_gen.exists(): shutil.rmtree(path_gen)
    path_gen.mkdir(exist_ok=True)
    data_gen = get_data(bs=bs, sz=sz, keep_pct=keep_pct)
    save_preds(data_gen.fix_dl)


# Create black and white training images

# Only runs if the directory isn't already created.

# if not path_lr.exists():
#     il = ImageList.from_folder(path_hr)
#     parallel(create_training_images, il.items)

# Number of black & white images
data_size = len(list(path_lr.rglob('*.*')))
print('Number of black & white images:', data_size)


# Pre-train generator
# Most of the training takes place here in pretraining for NoGAN.  The goal here is to take the generator as far as possible with conventional training, as that is much easier to control and obtain glitch-free results compared to GAN training.

# 64px

# Init logging of a new run
#wandb.init(tags=['Pre-train Gen'])  # tags are optional

bs=10
sz=64

# Define target number of training/validation samples as well as number of epochs
epoch_train_size = 100 * bs
epoch_valid_size = 10 * bs
valid_pct = epoch_valid_size / data_size
number_epochs = (data_size - epoch_valid_size) // epoch_train_size

# Log hyper parameters
#wandb.config.update({"Step 1 - batch size": bs, "Step 1 - image size": sz,
#                     "Step 1 - epoch size": epoch_train_size, "Step 1 - number epochs": number_epochs})

data_gen = get_data(bs=bs, sz=sz, random_seed=123, valid_pct=valid_pct, epoch_size=100*bs)

learn_gen = gen_learner_wide(data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor)

# learn_gen.callback_fns.append(partial(WandbCallback,
#                                      input_type='images'))  # log prediction samples

learn_gen.fit_one_cycle(number_epochs, pct_start=0.8, max_lr=slice(1e-3))
learn_gen.save(pre_gen_name)
learn_gen.unfreeze()
learn_gen.fit_one_cycle(number_epochs, pct_start=pct_start,  max_lr=slice(3e-7, 3e-4))

learn_gen.save(pre_gen_name)

# 128px

bs=10
sz=128

# Define target number of training/validation samples as well as number of epochs
epoch_train_size = 100 * bs
epoch_valid_size = 10 * bs
valid_pct = epoch_valid_size / data_size
number_epochs = (data_size - epoch_valid_size) // epoch_train_size

# Log hyper parameters
#wandb.config.update({"Step 2 - batch size": bs, "Step 2 - image size": sz,
#                     "Step 2 - epoch size": epoch_train_size, "Step 2 - number epochs": number_epochs})

learn_gen.data = get_data(bs=bs, sz=sz, random_seed=123, valid_pct=valid_pct, epoch_size=100*bs)

learn_gen.unfreeze()

learn_gen.fit_one_cycle(number_epochs, pct_start=pct_start, max_lr=slice(1e-7,1e-4))
learn_gen.save(pre_gen_name)

# 192px

bs=10
sz=192

# Define target number of training/validation samples as well as number of epochs
epoch_train_size = 100 * bs
epoch_valid_size = 10 * bs
valid_pct = epoch_valid_size / data_size
number_epochs = (data_size - epoch_valid_size) // epoch_train_size // 2  # Training is long - we use half of data

# Log hyper parameters
#wandb.config.update({"Step 3 - batch size": bs, "Step 3 - image size": sz,
#                     "Step 3 - epoch size": epoch_train_size, "Step 3 - number epochs": number_epochs})

learn_gen.data = get_data(bs=bs, sz=sz, random_seed=123, valid_pct=valid_pct, epoch_size=100*bs)
learn_gen.unfreeze()
learn_gen.fit_one_cycle(number_epochs, pct_start=pct_start, max_lr=slice(5e-8,5e-5))
learn_gen.save(pre_gen_name)

# End logging of current session run
# Note: this is optional and would be automatically triggered when stopping the kernel
#wandb.join()


# Repeatable GAN Cycle
# Best results so far have been based on repeating the cycle below a few times (about 5-8?), until diminishing returns are hit (no improvement in image quality).  Each time you repeat the cycle, you want to increment that old_checkpoint_num by 1 so that new check points don't overwrite the old.  

old_checkpoint_num = 0
checkpoint_num = old_checkpoint_num + 1
gen_old_checkpoint_name = gen_name + '_' + str(old_checkpoint_num)
gen_new_checkpoint_name = gen_name + '_' + str(checkpoint_num)
crit_old_checkpoint_name = crit_name + '_' + str(old_checkpoint_num)
crit_new_checkpoint_name= crit_name + '_' + str(checkpoint_num)


# Save Generated Images

bs=10
sz=192

# Define target number of training/validation samples as well as number of epochs
epoch_train_size = 100 * bs
epoch_valid_size = 10 * bs
valid_pct = epoch_valid_size / data_size
number_epochs = (data_size - epoch_valid_size) // epoch_train_size

data_gen = get_data(bs=bs, sz=sz, random_seed=123, valid_pct=valid_pct, epoch_size=100*bs)
learn_gen = gen_learner_wide(data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor).load(gen_old_checkpoint_name, with_opt=False)
save_gen_images(0.1)

# Pretrain Critic
# Only need full pretraining of critic when starting from scratch.  Otherwise, just finetune!

if old_checkpoint_num == 0:
    
    # Init logging of a new run
    #wandb.init(tags=['Pre-train Crit'])  # tags are optional
    
    bs=64
    sz=128
    learn_gen=None
    
    # Log hyper parameters
    #wandb.config.update({"Step 1 - batch size": bs, "Step 1 - image size": sz})

    gc.collect()    
    data_crit = get_crit_data([name_gen, 'test'], bs=bs, sz=sz)
    data_crit.show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)
    learn_crit = colorize_crit_learner(data=data_crit, nf=256)
    #learn_crit.callback_fns.append(partial(WandbCallback))  # log prediction samples
    learn_crit.fit_one_cycle(6, 1e-3)
    learn_crit.save(crit_old_checkpoint_name)

bs=10
sz=192

# Log hyper parameters
#wandb.config.update({"Step 2 - batch size": bs, "Step 2 - image size": sz})

data_crit = get_crit_data([name_gen, 'test'], bs=bs, sz=sz)
data_crit.show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)
learn_crit = colorize_crit_learner(data=data_crit, nf=256).load(crit_old_checkpoint_name, with_opt=False)
learn_crit.fit_one_cycle(4, 1e-4)
learn_crit.save(crit_new_checkpoint_name)


# GAN

# free up memory
learn_crit=None
learn_gen=None
learn=None
gc.collect()

# Set old_checkpoint_num to last iteration
old_checkpoint_num = 0
save_checkpoints = False
batch_per_epoch = 200

checkpoint_num = old_checkpoint_num + 1
gen_old_checkpoint_name = gen_name + '_' + str(old_checkpoint_num)
gen_new_checkpoint_name = gen_name + '_' + str(checkpoint_num)
crit_old_checkpoint_name = crit_name + '_' + str(old_checkpoint_num)
crit_new_checkpoint_name= crit_name + '_' + str(checkpoint_num)   

if False:   # need only to do it once
        
    # Generate data
    print('Generating data…')
    bs=8
    sz=192
    epoch_train_size = batch_per_epoch * bs
    epoch_valid_size = batch_per_epoch * bs // 10
    valid_pct = epoch_valid_size / data_size
    data_gen = get_data(bs=bs, sz=sz, epoch_size=epoch_train_size, valid_pct=valid_pct)
    learn_gen = gen_learner_wide(data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor).load(gen_old_checkpoint_name, with_opt=False)
    save_gen_images(0.02)

    # Pre-train critic
    print('Pre-training critic…')
    bs=16
    sz=192

    len_test = len(list((path / 'test').rglob('*.*')))
    len_gen = len(list((path / name_gen).rglob('*.*')))
    keep_test_pct = len_gen / len_test * 2

    data_crit = get_crit_data([name_gen, 'test'], bs=bs, sz=sz, pct=keep_test_pct)
    learn_crit = colorize_crit_learner(data=data_crit, nf=256).load(crit_old_checkpoint_name, with_opt=False)
    learn_crit.fit_one_cycle(1, 1e-4)
    learn_crit.save(crit_new_checkpoint_name)

# Creating GAN
print('Creating GAN…')
sz=192
bs=8
lr_GAN=2e-5
epoch_train_size = batch_per_epoch * bs
epoch_valid_size = batch_per_epoch * bs // 10
valid_pct = epoch_valid_size / data_size
len_test = len(list((path / 'test').rglob('*.*')))
len_gen = len(list((path / name_gen).rglob('*.*')))
keep_test_pct = len_gen / len_test * 2

data_crit = get_crit_data([name_gen, 'test'], bs=bs, sz=sz, pct=keep_test_pct)
learn_crit = colorize_crit_learner(data=data_crit, nf=256).load(crit_new_checkpoint_name, with_opt=False)
data_gen = get_data(bs=bs, sz=sz, epoch_size=epoch_train_size, valid_pct=valid_pct)
learn_gen = gen_learner_wide(data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor).load(gen_old_checkpoint_name, with_opt=False)
switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.0,1.5), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.9)), wd=1e-3)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))
#learn.callback_fns.append(partial(WandbCallback, input_type='images', seed=None, save_model=False))
learn.data = get_data(bs=bs, sz=sz, epoch_size=epoch_train_size, valid_pct=valid_pct)

# Start logging to W&B
#wandb.init(tags=['GAN'])
#wandb.config.update({"learning rate": lr_GAN})  

# Run the loop until satisfied with the results
while True:

    checkpoint_num = old_checkpoint_num + 1
    gen_old_checkpoint_name = gen_name + '_' + str(old_checkpoint_num)
    gen_new_checkpoint_name = gen_name + '_' + str(checkpoint_num)
    crit_old_checkpoint_name = crit_name + '_' + str(old_checkpoint_num)
    crit_new_checkpoint_name= crit_name + '_' + str(checkpoint_num)      
    
    
    # GAN for 10 epochs between each checkpoint
    try:
        learn.fit(1, lr_GAN)
    except:
        # Sometimes we get an error for some unknown reason during callbacks
        learn.callback_fns[-1](learn).on_epoch_end(old_checkpoint_num, None, [])
        
    if save_checkpoints:
        learn_crit.save(crit_new_checkpoint_name)
        learn_gen.save(gen_new_checkpoint_name)
    old_checkpoint_num += 1

# End logging of current session run
# Note: this is optional and would be automatically triggered when stopping the kernel
#wandb.join()