import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2

from param_parsar import parameter_parser
from tensorboardX import SummaryWriter
warnings.filterwarnings("ignore")


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
fix_all_seeds(2021)

torch.cuda.is_available()
cuda = torch.device('cuda')
BASE_PATH=os.getcwd()

SAMPLE_SUBMISSION  = os.path.join(BASE_PATH,'dataset/input/sartorius-cell-instance-segmentation/sample_submission.csv')
TRAIN_CSV =os.path.join(BASE_PATH,"dataset/input/sartorius-cell-instance-segmentation/train.csv")
TRAIN_PATH =os.path.join(BASE_PATH,"dataset/input/sartorius-cell-instance-segmentation/train")
TEST_PATH =os.path.join(BASE_PATH,"dataset/input/sartorius-cell-instance-segmentation/test")

IMAGE_RESIZE = (224, 224)
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

#args를 위해 drop
args=parameter_parser()
print(vars(args))
LEARNING_RATE = args.learning_rate
EPOCHS =args.epochs


def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


def build_masks(df_train, image_id, input_shape):
    height, width = input_shape
    labels = df_train[df_train["id"] == image_id]["annotation"].tolist()
    mask = np.zeros((height, width))
    for label in labels:
        mask += rle_decode(label, shape=(height, width))
    mask = mask.clip(0, 1)
    return mask


class CellDataset(Dataset):
    def __init__(self,df):
        self.df=df
        self.base_path=TRAIN_PATH
        self.transforms = Compose([Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]), 
                                    # Normalize(mean=RESNET_MEAN, std=RESNET_STD, p=1), 
                                    HorizontalFlip(p=0.5),
                                    VerticalFlip(p=0.5),
                                    ToTensorV2()])
        self.gb=self.df.groupby('id')
        self.image_ids=df.id.unique().tolist()
    
    def __getitem__(self,idx):
        image_id=self.image_ids[idx]
        df=self.gb.get_group(image_id)
        annotations=df['annotation'].tolist()
        image_path=os.path.join(self.base_path,image_id+".png")
        image=cv2.imread(image_path)
    #         image=np.asarray(image,dtype=np.uint8)
        mask=build_masks(df_train,image_id,input_shape=(520,704))
        mask=(mask>=1).astype('float32')
        augmented=self.transforms(image=image,mask=mask)
        image=augmented['image']
        mask=augmented['mask']
        return image, mask.reshape((1,IMAGE_RESIZE[0],IMAGE_RESIZE[1]))

        
    def __len__(self):
        return len(self.image_ids)


df_train=pd.read_csv(TRAIN_CSV)
ds_train = CellDataset(df_train)


# train_dataloader=DataLoader(ds_train,batch_size=48,
#                             shuffle=False,
#                             drop_last=True,
#                             pin_memory=True)

train_dataloader=DataLoader(ds_train,batch_size=args.batch_size,drop_last=args.drop_last,pin_memory=args.pin_memory,num_workers=args.num_workers)
# train_dataloader=DataLoader(ds_train,args)
#Loss 선언
def dice_loss(input,target,smooth=1e-5):
    input=torch.sigmoid(input)
    intersection=(input*target).sum(dim=(2,3))
    union=input.sum(dim=(2,3))+target.sum(dim=(2,3))
    
    dice=(2.0*(intersection+smooth))/(union+smooth)
    return dice

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target):
        loss=dice_loss(input,target)
        return loss.mean()
         

#Model 선언
import torch
import collections.abc as container_abcs
torch._six.container_abcs = container_abcs
import segmentation_models_pytorch as smp
model_name=args.model

model = smp.Unet(model_name, encoder_weights="imagenet", activation=None, in_channels=3, classes=1).cuda()

#optimizer
optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)

torch.set_default_tensor_type("torch.cuda.FloatTensor")
n_batches = len(train_dataloader)

model.train()

criterion = Loss()

# tensorboard에 출력을 위하여
experiment_name=str(args)
writer=SummaryWriter(log_dir=f"runs/"+model_name+experiment_name)

for epoch in range(1,EPOCHS+1):
    print(f"Starting epoch: {epoch} / {EPOCHS}")
    running_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_dataloader):
        #Predict
        images,masks=batch
        #그 전 코드를 한줄로 만들어줌
        #image = train_features[0].squeeze()
        #mask = train_masks[0]
        images=images.cuda()
        masks=masks.cuda()
        images = images.type('torch.cuda.FloatTensor')
        outputs=model(images)
        loss=criterion(outputs,masks)
        
        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        
    epoch_loss=running_loss/n_batches
    print(f"epoch: {epoch} -Train Loss {epoch_loss:.4f}")
    writer.add_scalar("loss/train",epoch_loss,epoch)

class TestCellDataset(Dataset):
    def __init__(self):
        self.test_path = TEST_PATH
        
        # I am not sure if they adapt the sample submission csv or only the test folder
        # I am using the test folders as the ground truth for the images to predict, which should be always right
        # The sample csv is ignored
        self.image_ids = [f[:-4]for f in os.listdir(self.test_path)]
        self.num_samples = len(self.image_ids)
        self.transform = Compose([Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]), ToTensorV2()])

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.test_path, image_id + ".png")
        image = cv2.imread(path)
        image = self.transform(image=image)['image']
        return {'image': image, 'id': image_id}

    def __len__(self):
        return self.num_samples

ds_test = TestCellDataset()
dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)

model.eval()
submission = []
for i, batch in enumerate(tqdm(dl_test)):
    preds = torch.sigmoid(model(batch['image'].type('torch.cuda.FloatTensor').cuda()))
    print(preds.shape)
    writer.add_image("image/test",preds[0],i)
    print(preds[0])
    preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
    # print(preds.shape)

    # plt.imshow(mask[0].detach().to("cpu"), alpha=0.3)
    # plt.imshow(ds_train[0][1].permute(1,2,0))

    # plt.show()
    
    # plt.imshow(preds[0])
    # plt.show()
    