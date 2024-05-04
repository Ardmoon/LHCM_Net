import torch
from LHCM_Net import LHCM_Net_320

weight_root = r'your_weight_path'  # your weight root
device = torch.device('cuda')  # device 'cuda' or 'cpu'
model = LHCM_Net_320().to(device)

weight_path = weight_root + r'\last.pth'   # weight
checkpoint = torch.load(weight_path)  # dict, include:'model','optimizer','scaler'
model.load_state_dict(checkpoint['model'])  # load model


x = torch.randn([16, 3, 320, 320]).to(device)   # your test image (batch)
a= model(x)
print(a[0].shape)






