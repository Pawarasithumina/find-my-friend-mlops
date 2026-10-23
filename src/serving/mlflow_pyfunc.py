"""Portable MLflow pyfunc face-classifier wrapper for serving JPEG image_b64 values."""
import base64
import io
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps


class FaceOpsPythonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from src.models.architectures import create_model

        state = torch.load(Path(context.artifacts['checkpoint']), map_location='cpu', weights_only=True)
        self.classes = list(state['classes'])
        if self.classes != ['pawara','uchitha','dhamith','unknown']:
            raise ValueError('Unknown class order')
        self.architecture = state['architecture']
        self.model = create_model(
            self.architecture,
            num_classes=4,
            dropout=float(state['hyperparameters']['dropout']),
            pretrained=False,
            freeze_backbone=bool(state['freeze_backbone']),
        )
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.model.eval()
        self.mean = torch.tensor((0.5, 0.5, 0.5) if self.architecture == 'custom_cnn' else (0.485, 0.456, 0.406)).view(3,1,1)
        self.std = torch.tensor((0.5, 0.5, 0.5) if self.architecture == 'custom_cnn' else (0.229, 0.224, 0.225)).view(3,1,1)
        torch.set_num_threads(min(torch.get_num_threads(), 4))

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame) or 'image_b64' not in model_input.columns:
            raise ValueError('Expected a DataFrame with image_b64 column')
        tensors = []
        if not len(model_input):
            raise ValueError('At least one image is required')
        if len(model_input) > 32:
            raise ValueError('Maximum 32 images per request')
        for encoded in model_input['image_b64']:
            if not isinstance(encoded, str) or len(encoded) > 12_000_000:
                raise ValueError('Invalid or oversized image_b64 input')
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > 8_000_000:
                raise ValueError('Image payload is too large')
            with Image.open(io.BytesIO(raw)) as img:
                img = ImageOps.exif_transpose(img).convert('RGB')
                img = ImageOps.pad(img, (224,224), method=Image.Resampling.BILINEAR, color=(0,0,0))
                arr = np.array(img, dtype=np.uint8, copy=True)
            tensor = torch.from_numpy(arr).permute(2,0,1).float().div(255)
            tensors.append((tensor - self.mean) / self.std)
        with torch.inference_mode():
            probs = self.model(torch.stack(tensors)).softmax(dim=1).numpy()
        output = []
        for p in probs:
            idx = int(p.argmax())
            row = {'class_name': self.classes[idx], 'confidence': float(p[idx])}
            for i, name in enumerate(self.classes):
                row['prob_' + name] = float(p[i])
            output.append(row)
        return pd.DataFrame(output)
