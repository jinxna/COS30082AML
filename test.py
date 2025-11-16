import torch; 

print('cuda available:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())
