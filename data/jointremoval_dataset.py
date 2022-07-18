import os.path
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class JointremovalDataset(BaseDataset):
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
        self.dir_A2 = os.path.join(opt.dataroot, opt.phase + 'A2')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_D1 = os.path.join(opt.dataroot, opt.phase + 'D1')
        self.dir_D2 = os.path.join(opt.dataroot, opt.phase + 'D2')


        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_A2 = os.path.join(opt.dataroot, "valA2")
            self.dir_B = os.path.join(opt.dataroot, "valB")
            self.dir_C = os.path.join(opt.dataroot, "valC")
            self.dir_D1 = os.path.join(opt.dataroot, "valD1")
            self.dir_D2 = os.path.join(opt.dataroot, "valD2")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.A2_paths = sorted(make_dataset(self.dir_A2, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))
        self.D1_paths = sorted(make_dataset(self.dir_D1, opt.max_dataset_size))
        self.D2_paths = sorted(make_dataset(self.dir_D2, opt.max_dataset_size))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D1_paths)

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
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_C = index % self.C_size
            index_D = index % self.D_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_C = random.randint(0, self.C_size - 1)
            index_D = random.randint(0, self.D_size - 1)

        B_path = self.B_paths[index_A]
        C_path = self.C_paths[index_C]
        D1_path = self.D1_paths[index_D]
        D2_path = self.D2_paths[index_D]
        A2_path = self.A2_paths[index_A]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')
        D1_img = Image.open(D1_path).convert('RGB')
        D2_img = Image.open(D2_path).convert('RGB')
        A2_img = Image.open(A2_path).convert('RGB')
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        transform_params = get_params(self.opt, A_img.size)
        fix_transform = get_transform(self.opt, transform_params)

        A = fix_transform(A_img)
        B = fix_transform(B_img)
        C = transform(C_img)
        D1 = fix_transform(D1_img)
        D2 = fix_transform(D2_img)
        A2 = fix_transform(A2_img)

        return {'A': A, 'B': B, 'C': C, 'D1': D1, 'A2': A2, 'D2': D2, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'D1_paths': D1_path, 'A2_paths': A2_path, 'D2_paths': D2_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size, self.C_size, self.D_size)
