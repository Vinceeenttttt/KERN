# Knowledge-Embedded Routing Network  for Scene Graph Generation
Tianshui Chen*, Weihao Yu*, Riquan Chen, and Liang Lin, “Knowledge-Embedded Routing Network for Scene Graph Generation”, CVPR, 2019. (* co-first authors) [[PDF](http://whyu.me/pdf/CVPR2019_KERN.pdf)]


This repository contains trained models and PyTorch version code for the above paper, If the paper significantly inspires you, we request that you cite our work:

### Bibtex

```
@inproceedings{chen2019knowledge,
  title={Knowledge-Embedded Routing Network for Scene Graph Generation},
  author={Chen, Tianshui and Yu, Weihao and Chen, Riquan and Lin, Liang},
  booktitle = "Conference on Computer Vision and Pattern Recognition",
  year={2019}
}
```
# Setup
In our paper, our model's strong baseline model is [SMN](https://arxiv.org/abs/1711.06640) (Stacked Motif Networks) introduced by [@rowanz](https://github.com/rowanz) et al. To compare these two models fairly, the PyTorch version code of our model is based on [@rowanz](https://github.com/rowanz)'s code [neural-motifs](https://github.com/rowanz/neural-motifs). Thank [@rowanz](https://github.com/rowanz) for sharing his nice code to research community.

0. Install python3.6 and pytorch 3. I recommend the [Anaconda distribution](https://repo.continuum.io/archive/). To install PyTorch if you haven't already, use
 ```conda install pytorch=0.3.0 torchvision=0.2.0 cuda90 -c pytorch```.
 We use [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) to observe the results of validation dataset. If you want to use it in PyTorch, you should install TensorFlow and [tensorboardX](https://github.com/lanpa/tensorboardX) first. If you don't want to use TensorBaord, just not use the command ```-tb_log_dir```.
 
1. Update the config file with the dataset paths. Specifically:
    - Visual Genome (the VG_100K folder, image_data.json, VG-SGG.h5, and VG-SGG-dicts.json). See data/stanford_filtered/README.md for the steps to download these.
    - You'll also need to fix your PYTHONPATH: `export PYTHONPATH=/home/yuweihao/exp/KERN`

2. Compile everything. Update your CUDA path in Makefile file and run `make` in the main directory: this compiles the Bilinear Interpolation operation for the RoIs.

3. Pretrain VG detection. To compare our model with [neural-motifs](https://github.com/rowanz/neural-motifs) fairly, we just use their pretrained VG detection. [You can download their pretrained detector checkpoint provided by @rowanz.](https://drive.google.com/open?id=11zKRr2OF5oclFL47kjFYBOxScotQzArX) 
You could also run ./scripts/pretrain_detector.sh to train detector by yourself. Note: You might have to modify the learning rate and batch size according to number and memory of GPU you have.

4. Generate knowledge matrices: ```python prior_matrices/generate_knowledge.py```, or download them from here: prior_matrices (<a href="https://drive.google.com/open?id=1Tg4CtK8Y1JkSsuaLWIqwzIrp6VXd11JP" target="_blank">Google Drive</a>, <a href="https://1drv.ms/f/s!ArFSFaZzVErwgUNM6nitaMleGkxW" target="_blank">OneDrive</a>).

5. Train our KERN model. There are three training phase. You need a GPU with 12G memory. 
    - Train VG relationship predicate classification: run ```CUDA_VISIBLE_DEVICES=YOUR_GPU_NUM ./scripts/train_kern_predcls.sh``` 
    This phase maybe last about 20-30 epochs. 
    - Train scene graph classification: run ```CUDA_VISIBLE_DEVICES=YOUR_GPU_NUM ./scripts/train_kern_sgcls.sh```. Before run this script, you need to modify the path name of best checkpoint you trained in precls phase: ```-ckpt checkpoints/kern_predcls/vgrel-YOUR_BEST_EPOCH_RNUM.tar```. It lasts about 8-13 epochs, then you can decrease the learning rate to 1e-6 to further improve the performance. Like neural-motifs, we use only one trained checkpoint for both predcls and sgcls tasks. You can also download our checkpoint here: kern_sgcls_predcls.tar (<a href="https://drive.google.com/open?id=1F2WBSGRHmJD9K1LT8ImkGOCuZraood21" target="_blank">Google Drive</a>, <a href="https://1drv.ms/f/s!ArFSFaZzVErwgUKVN85N17rMEXME" target="_blank">OneDrive</a>).
    - Refine for detection: run ```CUDA_VISIBLE_DEVICES=YOUR_GPU_NUM ./scripts/train_kern_sgdet.sh``` or download the checkpoint here: kern_sgdet.tar (<a href="https://drive.google.com/open?id=1hAx4MpMiwofABQi9H6_Jb0Qjp016JX7T" target="_blank">Google Drive</a>, <a href="https://1drv.ms/f/s!ArFSFaZzVErwgUKVN85N17rMEXME" target="_blank">OneDrive</a>). If you find the validation performance plateaus, you could also decrease learning rate to 1e-6 to improve performance. 

6. Evaluate: refer to the scripts ```CUDA_VISIBLE_DEVICES=YOUR_GPU_NUM ./scripts/eval_kern_[predcls/sgcls/sgdet].sh```. You can conveniently find all our checkpoints, evaluation caches and results in this folder KERN_Download (<a href="https://drive.google.com/open?id=1yCQfZRCt6UF-C-jSq78NaF9IUyNplLFx" target="_blank">Google Drive</a>, <a href="https://1drv.ms/f/s!ArFSFaZzVErwgT_SvqLZ3sv5XDu-" target="_blank">OneDrive</a>).







# Acknowledgement
Thank [@yuweihao](https://github.com/yuweihao) for his generously releasing nice code [KERN](https://github.com/yuweihao/KERN).






# Help

Feel free to open an issue if you encounter trouble getting it to work.




