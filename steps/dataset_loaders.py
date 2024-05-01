import pandas as pd
import os
from skimage import io
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(255),
                                transforms.RandomResizedCrop(224),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(),
                                transforms.ToTensor()])

class Dataset_Raigs(Dataset):
    def __init__(self,csv_file,root_dir,transform=None,train=True):
        #print(os.listdir())
        img_list = os.listdir(root_dir)
        img_list = [i.replace(".JPG","") for i in img_list]
        df = pd.read_csv(csv_file,sep=';')
        df = df[df['Eye ID'].isin(img_list)].reset_index(drop=True)
        df.fillna(-1,inplace=True)
        train_df, test_df = train_test_split(df, test_size=0.2,stratify=df['Final Label'], random_state=41)
        
        if(train):
            self.annotations = train_df
            self.root_dir = root_dir
            self.transform = transform
        else:
            self.annotations = test_df
            self.root_dir = root_dir
            self.transform = transform
            
    def __len__(self):
        return len(self.annotations) # 25000
    
    def __getitem__(self,index):
        image_path = os.path.join(self.root_dir,self.annotations.iloc[index,0]+'.JPG')
        image = io.imread(image_path)
        y_label = self.annotations.iloc[index,1]
        int_label = 0 if y_label == "NRG" else 1
        
        features = torch.tensor(self.annotations.iloc[index,[28, 30, 18, 27, 34, 17, 35, 8, 7, 3]].values.astype(float))
        
        
        if self.transform:
            image = self.transform(image)
        
        return (image,int_label,features)

def get_dataloaders(batch_size):
    training_data = Dataset_Raigs(
            csv_file="/workspace/shrey/Glucoma_Major/JustRAIGS_Train_labels.csv",
            root_dir="/workspace/shrey/Glucoma_Major/data",
            transform=transform,
            train = True
        )

       
    test_data = Dataset_Raigs(
            csv_file="/workspace/shrey/Glucoma_Major/JustRAIGS_Train_labels.csv",
            root_dir="/workspace/shrey/Glucoma_Major/data",
            transform=transform,
            train = False
        )
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


