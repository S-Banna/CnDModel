import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DamageDataset
from model import MiniUNet

dataset = DamageDataset("../training_data")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = MiniUNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(5):
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

torch.save(model.state_dict(), "mini_unet.pt")