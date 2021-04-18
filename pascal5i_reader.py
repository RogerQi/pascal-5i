import os
import ntpath
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
import torchvision

class Pascal5iReader(torchvision.datasets.vision.VisionDataset):
    """
    pascal_5i dataset reader

    Parameters:
        - root:  root to data folder containing SBD and VOC2012 dataset. See README.md for more details
        - fold:  folding index as in OSLSM (https://arxiv.org/pdf/1709.03410.pdf)
        - train: a bool flag to indicate whether L_{train} or L_{test} should be used
    """

    def __init__(self, root, fold, train=True):
        super(Pascal5iReader, self).__init__(root, None, None, None)
        assert fold >= 0 and fold <= 3
        self.train = train
        
        # Define base to SBD and VOC2012
        sbd_base = os.path.join(root, 'sbd')
        voc_base = os.path.join(root, 'VOCdevkit', 'VOC2012')
        
        # Define path to relevant txt files
        sbd_train_list_path = os.path.join(root, 'sbd', 'train.txt')
        sbd_val_list_path = os.path.join(root, 'sbd', 'val.txt')
        voc_train_list_path = os.path.join(voc_base, 'ImageSets', 'Segmentation', 'train.txt')
        voc_val_list_path = os.path.join(voc_base, 'ImageSets', 'Segmentation', 'val.txt')
        
        # Use np.loadtxt to load all train/val sets
        sbd_train_list = list(np.loadtxt(sbd_train_list_path, dtype = "str"))
        sbd_val_list = list(np.loadtxt(sbd_val_list_path, dtype = "str"))
        voc_train_list = list(np.loadtxt(voc_train_list_path, dtype = "str"))
        voc_val_list = list(np.loadtxt(voc_val_list_path, dtype = "str"))
        
        # Following PANet, we use images in SBD validation for training
        sbd_train_list = sbd_train_list + sbd_val_list
        
        # Remove overlapping images in SBD/VOC2012 from SBD train
        sbd_train_list = [i for i in sbd_train_list if i not in voc_val_list]
        
        # Generate self.images and self.targets
        if self.train:
            # If an image occur in both SBD and VOC2012, use VOC2012 annotation
            sbd_train_list = [i for i in sbd_train_list if i not in voc_train_list]
            
            # Generate image/mask full paths for SBD dataset
            sbd_train_img_list = [os.path.join(sbd_base, 'img', i + '.jpg') for i in sbd_train_list]
            sbd_train_target_list = [os.path.join(sbd_base, 'cls', i + '.mat') for i in sbd_train_list]

            # Generate image/mask full paths for VOC2012 segmentation training task
            voc_train_img_list = [os.path.join(voc_base, 'JPEGImages', i + '.jpg') for i in voc_train_list]
            voc_train_target_list = [os.path.join(voc_base, "SegmentationClass", i + '.png') for i in voc_train_list]

            # FINAL: Merge these two datasets
            self.images = sbd_train_img_list + voc_train_img_list
            self.targets = sbd_train_target_list + voc_train_target_list
        else:
            # Generate image/mask full paths for VOC2012 semantation validation task
            # Following PANet, only VOC2012 validation set is used for validation
            self.images = [os.path.join(voc_base, 'JPEGImages', i + '.jpg') for i in voc_val_list]
            self.targets = [os.path.join(voc_base, "SegmentationClass", i + '.png') for i in voc_val_list]
        
        # Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
        # Given fold number, define L_{test}
        self.val_label_set = [i for i in range(fold * 5 + 1, fold * 5 + 6)]
        self.train_label_set = [i for i in range(1, 21) if i not in self.val_label_set]
        if self.train:
            self.label_set = self.train_label_set
        else:
            self.label_set = self.val_label_set
        
        assert len(self.images) == len(self.targets)
        self.to_tensor_func = torchvision.transforms.ToTensor()
        
        # Find subset of image. This is actually faster than hist
        folded_images = []
        folded_targets = []
        for i in range(len(self.images)):
            mask = self.load_seg_mask(self.targets[i])
            for x in self.label_set:
                if x in mask:
                    # contain at least one pixel in L_{train}
                    folded_images.append(self.images[i])
                    folded_targets.append(self.targets[i])
                    break

        self.images = folded_images
        self.targets = folded_targets
    
    def __len__(self):
        return len(self.images)
    
    def load_seg_mask(self, file_path):
        """
        Load seg_mask from file_path (supports .mat and .png).

        Target masks in SBD are stored as matlab .mat; while those in VOC2012 are .png

        Parameters:
            - file_path: path to the segmenation file
        
        Return: a numpy array of dtype long and element range(0, 21) containing segmentation mask
        """
        if file_path.endswith('.mat'):
            mat = loadmat(file_path)
            target = Image.fromarray(mat['GTcls'][0]['Segmentation'][0])
        else:
            target = Image.open(file_path)
        target_np = np.array(target, dtype=np.long)
        
        # Annotation in VOC contains 255
        target_np[target_np > 20] = 0
        return target_np

    def set_bg_pixel(self, target_np):
        """
        Following OSLSM, we mask pixels not in current label set as 0. e.g., when
        self.train = True, pixels whose labels are in L_{test} are masked as background

        Parameters:
            - target_np: segmentation mask (usually returned array from self.load_seg_mask)
        
        Return:
            - Offseted and masked segmentation mask
        """
        if self.train:
            for x in self.val_label_set:
                target_np[target_np == x] = 0
            max_val_label = max(self.val_label_set)
            target_np[target_np > max_val_label] = target_np[target_np > max_val_label] - 5
        else:
            label_mask_idx_map = []
            for x in self.val_label_set:
                label_mask_idx_map.append(target_np == x)
            target_np = np.zeros_like(target_np)
            for i in range(len(label_mask_idx_map)):
                target_np[label_mask_idx_map[i]] = i + 1
        return target_np
    
    def __getitem__(self, idx):
        # For both SBD and VOC2012, images are stored as .jpg
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.to_tensor_func(img)
        
        target_np = self.load_seg_mask(self.targets[idx])
        target_np = self.set_bg_pixel(target_np)
        
        return img, torch.tensor(target_np)
