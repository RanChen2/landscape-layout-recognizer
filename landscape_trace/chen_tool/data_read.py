from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import random
def resize_image(img, max_size=1024):
    width, height = img.size
    if width > height:
        ratio = max_size / width
    else:
        ratio = max_size / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
    return resized_img


class data_read_for_miaotuji():
    def __init__(self, testset,num_threads,batch_size):
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.load_size = 512
        self.testset = testset
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM',
                               '.bmp', '.BMP','.tif', '.TIF', '.tiff', '.TIFF',]
        self.images = [os.path.join(root, fname)
                       for root, _, fnames in sorted(os.walk(self.testset))
                       for fname in fnames
                       if any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)]
        self.images = sorted(self.images)
        print(f"总共有{len(self.images)}张图")
        self.dataloader = DataLoader(self, batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=int(self.num_threads))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        test_path = self.images[index]
        test_name = test_path.split('\\')[-1].split('.')[0]
        test = Image.open(test_path).convert('RGB')
        test = resize_image(test)
        test_size = test.size
        # Transformation sequence
        transform_list = [
            transforms.Resize([self.load_size, self.load_size], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(transform_list)

        return {'test': transform(test),'test_path':test_path, 'test_name': test_name, 'test_size':test_size}

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


class data_read_for_general_training():
    def __init__(self, dataroot,num_threads,batch_size,load_size,crop_size,trainingmode):
        self.trainingmode = trainingmode
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.load_size = load_size
        self.crop_size = crop_size
        self.dir_A = os.path.join(dataroot, 'trainA')
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM',
                               '.bmp', '.BMP','.tif', '.TIF', '.tiff', '.TIFF',]
        self.images = [os.path.join(root, fname)
                       for root, _, fnames in sorted(os.walk(self.dir_A))
                       for fname in fnames
                       if any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)]
        self.images = sorted(self.images)
        print(f"总共有{len(self.images)}张图")
        self.dataloader = DataLoader(self, batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=int(self.num_threads))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        A_path = self.images[index]
        if self.trainingmode == 'pix2pix':
            index_B = index
        elif self.trainingmode == 'cyclegan':
            index_B = random.randint(0, len(self.images) - 1)
        B_path = self.images[index_B].replace('trainA','trainB')
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # Random transformations parameters
        x = random.randint(0, max(0, self.load_size - self.crop_size))
        y = random.randint(0, max(0, self.load_size - self.crop_size))
        flip = random.random() > 0.5

        # Transformation sequence
        transform_list = [
            transforms.Resize([self.load_size, self.load_size], InterpolationMode.BICUBIC),
            # transforms.Resize([700, 1001], Image.BICUBIC),
            transforms.Lambda(lambda img: img.crop((x, y, x + self.crop_size, y + self.crop_size))),
            transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(transform_list)

        return {'A': transform(A), 'B': transform(B), 'A_paths': A_path, 'B_paths': A_path}

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


class data_read():
    def __init__(self, dataroot,num_threads,batch_size,load_size,crop_size,trainingmode,A_name,B_name):
        self.trainingmode = trainingmode
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.load_size = load_size
        self.crop_size = crop_size
        self.A_name = A_name
        self.B_name = B_name
        self.dir_A = os.path.join(dataroot, 'trainA')
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM',
                               '.bmp', '.BMP','.tif', '.TIF', '.tiff', '.TIFF',]
        self.images = [os.path.join(root, fname)
                       for root, _, fnames in sorted(os.walk(self.dir_A))
                       for fname in fnames
                       if any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)]
        self.images = sorted(self.images)
        print(f"总共有{len(self.images)}张图")
        self.dataloader = DataLoader(self, batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=int(self.num_threads))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        A_path = self.images[index]
        if self.trainingmode == 'pix2pix':
            index_B = index
        elif self.trainingmode == 'cyclegan':
            index_B = random.randint(0, len(self.images) - 1)
        B_path = self.images[index_B].replace('trainA','trainB')
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # Random transformations parameters
        x = random.randint(0, max(0, self.load_size - self.crop_size))
        y = random.randint(0, max(0, self.load_size - self.crop_size))
        flip = random.random() > 0.5

        # Transformation sequence
        transform_list = [
            transforms.Resize([self.load_size, self.load_size], InterpolationMode.BICUBIC),
            # transforms.Resize([700, 1001], Image.BICUBIC),
            transforms.Lambda(lambda img: img.crop((x, y, x + self.crop_size, y + self.crop_size))),
            transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(transform_list)

        return {'A': transform(A), 'B': transform(B), 'A_paths': A_path, 'B_paths': A_path}

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


