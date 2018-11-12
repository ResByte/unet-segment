import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

class ADE20K(Dataset):
    """ADE20K dataset is from MIT scene parsing challenge. 
    This is becoming a benchmark for several CV papers. 
    Reference: http://groups.csail.mit.edu/vision/datasets/ADE20K/
    """
    def __init__(self,root = './data/ADEChallengeData2016', is_train=True, resize=224):
        self.root = root 
        self.prepare_data()
        self.is_train = is_train
        self.resize = resize
        self.img_transform = transforms.Compose([
                transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
                ])
        if self.is_train:
            print(len(self.train_imgs))
        else:
            print(len(self.val_imgs))
    
    def prepare_data(self):
        image_path = os.path.join(self.root, 'images')
        ann_path = os.path.join(self.root, 'annotations')
        self.train_anns, self.val_anns = self.get_file_lists(ann_path)
        self.train_imgs, self.val_imgs = self.get_file_lists(image_path)
        assert len(self.train_imgs) == len(self.train_anns)
        assert len(self.val_imgs) == len(self.val_anns)
        
    def get_file_lists(self, data_path):
        data_train_files = os.listdir(os.path.join(data_path,'training'))
        data_val_files = os.listdir(os.path.join(data_path,'validation'))
        train_files = [os.path.join(data_path, 'training',x) for x in data_train_files ]
        val_files = [os.path.join(data_path, 'validation',x) for x in data_val_files ]
        return sorted(train_files), sorted(val_files)
    
    def load_image(self, img_path, ann_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert colors 
        img = cv2.resize(img,(self.resize,self.resize)) # resize to a fixed size
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1) # (H, W, C) --> (C, H, W)
        img = self.img_transform(torch.from_numpy(img.copy()))
        
        ann = cv2.imread(ann_path)
        ann = ann[:,:,0] # since all the channels have same values. 
        ann = cv2.resize(ann,(self.resize,self.resize))
        # ann = ann[np.newaxis,:,:]
        return img, torch.from_numpy(ann.astype(np.int)).long()
    
    def __getitem__(self, index):
        if self.is_train:
            return self.load_image(self.train_imgs[index], self.train_anns[index])
        else:
            return self.load_image(self.val_imgs[index], self.val_anns[index])
    
    def __len__(self):
        if self.is_train:
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)

if __name__=='__main__':
    
    train_data = ADEData(is_train=True)
    val_data = ADEData(is_train=False)
    img, ann = val_data[0]
    img.shape, ann.shape