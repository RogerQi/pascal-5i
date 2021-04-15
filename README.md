# pascal-5i
A working Python implementation to construct pascal-5i dataset.

PASCAL-5i is a dataset frequently used in few-shot segmentation literatures.
However, there doesn't seem to be a **working** implementation available
online:

- [The official implementation](https://github.com/lzzcd001/OSLSM/blob/master/OSLSM/code/ss_settings.py)
    from the [OSLSM paper](https://arxiv.org/pdf/1709.03410.pdf) is written in a rather confusing way (at least to me).
- [This GitHub repo](https://github.com/DeepTrial/pascal-5) was the only other open-source
    implementation that I found. However, the code is not working and there are several unaddressed GitHub issues.

Hence, I want to implement this tool for my own use and anyone who is interested in
using pascal-5i. This is effectively a PyTorch dataset reader. The *i*-folding is
done in the dataset initialization stage. Hence, it poses no overhead during training/inference.

## Usage

By default, you can run data reading example at,
```
python3 main.py
```

The [SBD Dataset](http://home.bharathh.info/pubs/codes/SBD/download.html) and the
[Pascal VOC2012 Dataset](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/) will
be automatically downloaded to /data/sbd and /data/VOCdevkit using torchvision utility functions.

If you have downloaded the SBD dataset and the pascal VOC2012 dataset, you can specify

```
python3 main.py --base_dir /home/some_user/data
```

An example hierarchy of base dir is given here

```
.
+-- /home/some_user/data
|   +-- sbd
|       +-- cls (these files are extracted from [this zip](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz))
|       +-- img
|       ...
|   +-- VOCdevkit
|       +-- VOC2012 (extracted from [this zip](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar))
|           ...
```


## Caveats

- As mentioned in https://arxiv.org/pdf/1411.4038.pdf, there are actually some overlapping
    examples in the SBD training set and the VOC2012 validation set. We follow practices in
    [PANet](https://arxiv.org/abs/1908.06391), which aggregates all images in the SBD dataset
    and the Pascal VOC2012 dataset. Then, it isolates all images in the VOC2012 validation set
    as the validation set for Pascal 5i; while the rest of the images (including images
    from the SBD validation set) are considered as training set.
