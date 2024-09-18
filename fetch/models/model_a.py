''' In this model, freq time is processed through a DenseNet121 block and  dm time
    is processed through and Xception block '''

import torch
import torch.nn as nn

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')