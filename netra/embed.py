"""Route B features: let a pre-trained image network READ the photo.

We use ResNet-50 that was trained on ImageNet (1.2 million everyday photos). We do NOT
train it. We cut off its last layer and keep the 2048 numbers it produces for each
photo (an "embedding"). Our own statistical-ML models (logistic regression, SVM, ...)
then learn the DR grade from those numbers.
"""
import os

import cv2
import numpy as np

EMBED_DIM = 2048
ARCH = "resnet50"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Embedder:
    def __init__(self, size=384, pretrained=True, threads=None):
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        self.torch = torch
        self.size = int(size)
        if threads or os.cpu_count():
            torch.set_num_threads(int(threads or os.cpu_count()))
        # NETRA_NO_PRETRAINED=1 is only for offline code tests (random weights, useless results)
        pretrained = pretrained and os.environ.get("NETRA_NO_PRETRAINED") != "1"
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)
        net.fc = torch.nn.Identity()  # drop the ImageNet classifier, keep the 2048 numbers
        self.net = net.eval()

    def _to_tensor(self, rgb_list):
        arr = []
        for rgb in rgb_list:
            if rgb.shape[0] != self.size or rgb.shape[1] != self.size:
                rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
            arr.append(((rgb.astype(np.float32) / 255.0) - _MEAN) / _STD)
        x = np.stack(arr).transpose(0, 3, 1, 2)  # N, 3, H, W
        return self.torch.from_numpy(np.ascontiguousarray(x))

    def __call__(self, rgb_list, batch_size=16):
        """rgb_list: list of uint8 RGB arrays. Returns float32 array (N, 2048)."""
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(rgb_list), batch_size):
                x = self._to_tensor(rgb_list[i:i + batch_size])
                out.append(self.net(x).numpy().astype(np.float32))
        return np.concatenate(out, axis=0)
