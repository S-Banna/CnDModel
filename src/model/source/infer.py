import torch
import cv2
from dataset import DamageDataset
from model import MiniUNet

dataset = DamageDataset("../training_data")
model = MiniUNet()
model.load_state_dict(torch.load("mini_unet.pt"))
model.eval()

for i in range(19, 21):
    x, y = dataset[i]

    with torch.no_grad():
        pred = torch.sigmoid(model(x.unsqueeze(0)))[0,0].numpy()

    binary = (pred > 0.08).astype("uint8") * 255

    cv2.imshow("Binary Prediction", binary)
    cv2.imshow("Mask", y[0].numpy())
    cv2.waitKey(0)