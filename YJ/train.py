import os, random
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from transforms import *
from losses import *
from utils import post_process
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp

SAMPLE_SUBMISSION  = '../input/sartorius-cell-instance-segmentation/sample_submission.csv'
TRAIN_CSV = "./train.csv"
TRAIN_PATH = "./train"
TEST_PATH = "./test"

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    def __init__(self, df):
        self.df = df
        self.base_path = TRAIN_PATH
        self.transforms = Compose([Resize(224), ToTensor()])
        self.gb = self.df.groupby('id')
        self.image_ids = df.id.unique().tolist()

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)
        annotations = df['annotation'].tolist()
        image_path = os.path.join(self.base_path, image_id + ".png")
        image = cv2.imread(image_path)
        mask = build_masks(pd.read_csv(TRAIN_CSV), image_id, input_shape=(520, 704))
        mask = (mask >= 1).astype('float32')
        augmented = self.transforms((image, mask))
        image, mask = augmented
        return image, mask

    def __len__(self):
        return len(self.image_ids)

class TestCellDataset(Dataset):
    def __init__(self):
        self.test_path = TEST_PATH

        # I am not sure if they adapt the sample submission csv or only the test folder
        # I am using the test folders as the ground truth for the images to predict, which should be always right
        # The sample csv is ignored
        self.image_ids = [f[:-4]for f in os.listdir(self.test_path)]
        self.num_samples = len(self.image_ids)
        self.transform = Compose([Resize_test(224), ToTensor_test()])

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.test_path, image_id + ".png")
        image = cv2.imread(path)
        image = self.transform(image)
        return image, image_id

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    fix_all_seeds(42)

    df_train = pd.read_csv(TRAIN_CSV)
    ds_train = CellDataset(df_train)
    ds_test = TestCellDataset()

    # plt.imshow(img[0], cmap='bone')
    # plt.show()
    # plt.imshow(mask[0], alpha=0.3)
    # plt.show()

    batch_size = 32
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = smp.UnetPlusPlus(encoder_name='efficientnet-b2', encoder_weights='imagenet', activation='sigmoid')
    model = model.cuda()

    focal_loss = FocalLoss(2.0)
    mixed_loss = MixedLoss(20.0, 2.0)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    writer = SummaryWriter(log_dir=os.path.join('runs'))

    model.train()
    for epoch in range(1, 20):
        model.train()
        print(f"Starting epoch: {epoch} / 20")
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dl_train):
            # Predict
            images, masks = batch
            images, masks = images.cuda(),  masks.cuda()
            outputs = model(images)
            loss = focal_loss(outputs, masks)

            # Back prop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        epoch_loss = running_loss / batch_size
        writer.add_scalar('train/loss', epoch_loss, epoch)
        print(f"Epoch: {epoch} - Train Loss {epoch_loss:.4f}")

        model.eval()
        for i, batch in enumerate(tqdm(dl_test)):
            val_image, val_id = batch
            preds = torch.sigmoid(model(val_image.cuda()))
            writer.add_images('test/pred', preds, epoch)
            preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
            for image_id, probability_mask in zip(val_id, preds):
                try:
                    #if probability_mask.shape != IMAGE_RESIZE:
                    #    probability_mask = cv2.resize(probability_mask, dsize=IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
                    probability_mask = cv2.resize(probability_mask, dsize=(704, 520), interpolation=cv2.INTER_LINEAR)
                    predictions = post_process(probability_mask)
                    predictions = ToTensor(predictions)
                    writer.add_images('test/post_pred', predictions, epoch)

                except Exception as e:
                    print(f"Exception for img: {image_id}: {e}")
