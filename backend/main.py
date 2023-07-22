import torch
import cv2
from server.modelConfig import NNArch, classes

checkpoint = torch.load('./intelClass.pth', map_location=torch.device('cpu'))
model= NNArch()
model.load_state_dict(model)
model.eval()

img = './data/seg_pred/seg_pred/3.jpg'




