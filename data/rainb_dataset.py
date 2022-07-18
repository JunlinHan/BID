import os.path
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class RainbDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        if opt.phase == "train":
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
            self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
            self.dir_E1 = os.path.join(opt.dataroot, opt.phase + 'E1')
            self.dir_E2 = os.path.join(opt.dataroot, opt.phase + 'E2')

            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
            self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))
            self.E1_paths = sorted(make_dataset(self.dir_E1, opt.max_dataset_size))
            self.E2_paths = sorted(make_dataset(self.dir_E2, opt.max_dataset_size))

            self.A_size = len(self.A_paths)  # get the size of dataset A
            self.B_size = len(self.B_paths)
            self.C_size = len(self.C_paths)
            self.E_size = len(self.E1_paths)

        if opt.phase == "test":
            if opt.test_input == "A":
                self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
            elif opt.test_input == "B":
                self.dir_A = os.path.join(opt.dataroot, opt.phase + 'B')
            elif opt.test_input == "C":
                self.dir_A = os.path.join(opt.dataroot, opt.phase + 'C')
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
            self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
            # E1/E2 are useless, masks are not used in Task II.B testing
            self.dir_E1 = os.path.join(opt.dataroot, opt.phase + 'E1')
            self.dir_E2 = os.path.join(opt.dataroot, opt.phase + 'E2')

            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
            self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))
            self.E1_paths = sorted(make_dataset(self.dir_E1, opt.max_dataset_size))
            self.E2_paths = sorted(make_dataset(self.dir_E2, opt.max_dataset_size))

            self.A_size = len(self.A_paths)
            self.B_size = len(self.B_paths)
            self.C_size = len(self.C_paths)
            self.E_size = len(self.E1_paths)

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
            index_B = index % self.B_size
            index_C = index % self.C_size
            index_E = index % self.E_size
        else:
            index_B = random.randint(0, self.B_size - 1)
            index_C = random.randint(0, self.C_size - 1)
            index_E = random.randint(0, self.E_size - 1)

        B_path = self.B_paths[index_B]
        C_path = self.C_paths[index_C]
        E1_path = self.E1_paths[index_E]
        E2_path = self.E2_paths[index_E]


        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')
        E1_img = Image.open(E1_path).convert('RGB')
        E2_img = Image.open(E2_path).convert('RGB')

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        transform_params = get_params(self.opt, A_img.size)
        fix_transform = get_transform(self.opt, transform_params)

        A = fix_transform(A_img)
        B = transform(B_img)
        C = transform(C_img)
        E1 = fix_transform(E1_img)
        E2 = fix_transform(E2_img)

        return {'A': A, 'B': B, 'C': C, 'E1': E1, 'E2': E2, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'E1_paths': E1_path, 'E2_paths': E2_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size, self.C_size, self.E_size)
