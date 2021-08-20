import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class Unaligned8Dataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')  # create a path '/path/to/data/trainC'
        self.dir_D = os.path.join(opt.dataroot, opt.phase + 'D')  # create a path '/path/to/data/trainD'
        self.dir_E = os.path.join(opt.dataroot, opt.phase + 'E')  # create a path '/path/to/data/trainE'
        self.dir_F = os.path.join(opt.dataroot, opt.phase + 'F')  # create a path '/path/to/data/trainF'
        self.dir_G = os.path.join(opt.dataroot, opt.phase + 'G')  # create a path '/path/to/data/trainG'
        self.dir_H = os.path.join(opt.dataroot, opt.phase + 'H')  # create a path '/path/to/data/trainH'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")
            self.dir_C = os.path.join(opt.dataroot, "valC")
            self.dir_D = os.path.join(opt.dataroot, "valD")
            self.dir_E = os.path.join(opt.dataroot, "valE")
            self.dir_F = os.path.join(opt.dataroot, "valF")
            self.dir_G = os.path.join(opt.dataroot, "valG")
            self.dir_H = os.path.join(opt.dataroot, "valH")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))
        self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.E_paths = sorted(make_dataset(self.dir_E, opt.max_dataset_size))
        self.F_paths = sorted(make_dataset(self.dir_F, opt.max_dataset_size))
        self.G_paths = sorted(make_dataset(self.dir_G, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.H_paths = sorted(make_dataset(self.dir_H, opt.max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.C_size = len(self.C_paths)  # get the size of dataset C
        self.D_size = len(self.D_paths)  # get the size of dataset B
        self.E_size = len(self.E_paths)  # get the size of dataset C
        self.F_size = len(self.F_paths)  # get the size of dataset C
        self.G_size = len(self.G_paths)  # get the size of dataset B
        self.H_size = len(self.H_paths)  # get the size of dataset C

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
            index_C = index % self.C_size
            index_D = index % self.D_size
            index_E = index % self.E_size
            index_F = index % self.F_size
            index_G = index % self.G_size
            index_H = index % self.H_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
            index_C = random.randint(0, self.C_size - 1)
            index_D = random.randint(0, self.D_size - 1)
            index_E = random.randint(0, self.E_size - 1)
            index_F = random.randint(0, self.F_size - 1)
            index_G = random.randint(0, self.G_size - 1)
            index_H = random.randint(0, self.H_size - 1)

        B_path = self.B_paths[index_B]
        C_path = self.C_paths[index_C]
        D_path = self.D_paths[index_D]
        E_path = self.E_paths[index_E]
        F_path = self.F_paths[index_F]
        G_path = self.G_paths[index_G]
        H_path = self.H_paths[index_H]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')
        D_img = Image.open(D_path).convert('RGB')
        E_img = Image.open(E_path).convert('RGB')
        F_img = Image.open(F_path).convert('RGB')
        G_img = Image.open(G_path).convert('RGB')
        H_img = Image.open(H_path).convert('RGB')
        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)
        C = transform(C_img)
        D = transform(D_img)
        E = transform(E_img)
        F = transform(F_img)
        G = transform(G_img)
        H = transform(H_img)

        return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H,
                'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'D_paths': D_path, 'E_paths': E_path, 'F_paths': F_path, 'G_paths': G_path, 'H_paths': H_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size, self.C_size, self.D_size, self.E_size, self.F_size, self.G_size, self.H_size)
