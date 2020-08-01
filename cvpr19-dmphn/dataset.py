from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

my_transform = transforms.Compose([
                    # transforms.RandomCrop((256,256)),     
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 1),
                ])

def rand_crop(data, label, img_h, img_w):
    width1 = random.randint(0, data.size[0] - img_w)
    height1 = random.randint(0, data.size[1] - img_h)
    width2 = width1 + img_w
    height2 = height1 + img_h 
    data = data.crop((width1, height1, width2, height2))
    label = label.crop((width1, height1, width2, height2))
    return data,label

class GoProDataset(Dataset):
    def __init__(self, path, mode='train'):
        super(self).__init__()
        self.inputs = []
        self.targets = []
        self.transform = my_transform
        if mode == 'train':
            train_fs = os.listdir(path+'/train/')
            for f_dir in train_fs:
                tmp_data_path = path+'/train/'+f_dir+"/blur/"
                tmp_target_path = path+'/train/'+f_dir+"/sharp/"
                for data_name in os.listdir(tmp_data_path):
                    self.inputs.append(os.path.join(tmp_data_path, data_name))
                    self.targets.append(os.path.join(tmp_target_path, data_name))
        elif mode == 'test':
            train_fs = os.listdir(path+'/test/')
            for f_dir in train_fs:
                tmp_data_path = path+'/test/'+f_dir+"/blur/"
                tmp_target_path = path+'/test/'+f_dir+"/sharp/"
                for data_name in os.listdir(tmp_data_path):
                    self.inputs.append(os.path.join(tmp_data_path, data_name))
                    self.targets.append(os.path.join(tmp_target_path, data_name))
        else:
            print("Unknown mode('test'or'train')!")

    def __getitem__(self, index):
        input_img_path, target_img_path = self.inputs[index], self.targets[index]
        img = Image.open(input_img_path).convert('RGB')
        target = Image.open(target_img_path).convert('RGB')
        img, target = rand_crop(img, target, 256, 256)
        img = self.transform(img)
        target = self.transform(target)
        return img, target

    def __len__(self):
        return len(self.inputs)