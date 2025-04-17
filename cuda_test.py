import torch
print(torch.cuda.is_available())  # Should print True if CUDA is available
print(torch.__version__)

import numpy as np
print(np.__version__)  # This should print the version of NumPy you installed
