from abc import ABC, abstractmethod

import einops
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Resize
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    ViTMAEForPreTraining,
)


def get_processor_vit(model_name):
    model_name = model_name.lower()

    if "clip" in model_name:
        if model_name == "laion/CLIP-ViT-B-16-laion2B-s34B-b88K".lower():
            model, _, processor = open_clip.create_model_and_transforms(
                model_name="ViT-B-16", pretrained="laion2b_s34b_b88k"
            )
            vit = model.visual
        else:
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            vit = model.vision_model
    elif "dinov2" in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        vit = AutoModel.from_pretrained(model_name)
    elif "siglip" in model_name:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        vit = model.vision_model
    elif "mae" in model_name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTMAEForPreTraining.from_pretrained(model_name)
        vit = model.vit
    else:
        raise ValueError(f"Unsupported model {model_name}.")

    return processor, vit


def get_tivit(
    model_name,
    model_layer,
    aggregation,
    stride,
    patch_size,
):
    processor, vit = get_processor_vit(model_name)

    if hasattr(vit, "transformer") and hasattr(vit.transformer, "resblocks"):
        TiViTClass = TiViT_OpenCLIP
    elif hasattr(vit, "encoder") and (
        hasattr(vit.encoder, "layers") or hasattr(vit.encoder, "layer")
    ):
        TiViTClass = TiViT_HF
    else:
        raise ValueError("Unsupported model structure.")

    tivit = TiViTClass(
        processor=processor,
        vit=vit,
        layer_idx=model_layer,
        aggregation=aggregation,
        patch_size=patch_size,
        stride=stride,
    )

    return tivit


class BaseTiViT(nn.Module, ABC):
    def __init__(self, processor, vit, layer_idx, aggregation, patch_size, stride):
        super().__init__()
        self.processor = processor
        self.vit = vit
        self.layer_idx = layer_idx
        self.aggregation = aggregation
        self.patch_size = patch_size
        self.stride = stride
        self.processor = processor
        self.truncate_layers()

    @abstractmethod
    def truncate_layers(self):
        """Truncate transformer layers"""
        pass

    @abstractmethod
    def forward_vit(self, inputs):
        """Forward pass through ViT to extract hidden representations"""
        pass

    def forward(self, inputs):
        inputs = self.ts2image_transformation(
            inputs, patch_size=self.patch_size, stride=self.stride
        )
        hidden = self.forward_vit(inputs)

        return self.aggregate_hidden_representations(
            hidden, aggregation=self.aggregation
        )

    def aggregate_hidden_representations(self, hidden_states, aggregation):
        if aggregation == "mean":
            pooled = hidden_states.mean(dim=1)
        elif aggregation == "cls_token":
            pooled = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unsupported aggregation {aggregation}")

        return pooled

    def ts2image_transformation(
        self,
        x,
        patch_size,
        stride,
        image_size=224,
    ):
        # x: B x T x D
        # Normalization using robust scaling
        median = x.median(1, keepdim=True)[0]
        q_tensor = torch.tensor([0.75, 0.25], device=x.device, dtype=x.dtype)
        q75, q25 = torch.quantile(x, q_tensor, dim=1, keepdim=True)
        x = x - median
        iqr = q75 - q25
        x = x / (iqr + 1e-5)

        x = einops.rearrange(x, "b t d -> b d t")
        T = x.shape[-1]

        if stride == 1:  # No overlapping patches
            pad_left = 0
            if T % patch_size != 0:
                pad_left = patch_size - T % patch_size
            x_pad = F.pad(x, (pad_left, 0), mode="replicate")
            x_2d = einops.rearrange(x_pad, "b d (p f) -> (b d) 1 f p", f=patch_size)
        elif stride > 0 and stride < 1:  # Overlapping patches
            pad_left = 0
            if int(patch_size * stride) == 0:
                stride_len = 1
            else:
                stride_len = int(patch_size * stride)
            remainder = (T - patch_size) % stride_len
            if remainder != 0:
                pad_left = stride_len - remainder
            x_pad = F.pad(x, (pad_left, 0), mode="replicate")
            x_2d = x_pad.unfold(dimension=2, size=patch_size, step=stride_len)
        else:
            raise ValueError(
                f"Stride is set to {stride}, but should be a fraction of the patch size, and thus lie between 0 and 1."
            )

        # Adjust contrast
        min_vals = x_2d.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_vals = x_2d.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x_2d = (x_2d - min_vals) / (max_vals - min_vals + 1e-5)
        x_2d = torch.pow(x_2d, 0.8)

        # Resize to ViT input resolution
        x_resized = Resize((image_size, image_size), interpolation=0, antialias=False)(
            x_2d
        )

        # Generate grayscale images
        image_input = einops.repeat(x_resized, "b 1 h w -> b c h w", c=3)

        return image_input


class TiViT_HF(BaseTiViT):
    def __init__(self, processor, vit, layer_idx, aggregation, patch_size, stride):
        super().__init__(processor, vit, layer_idx, aggregation, patch_size, stride)
        self.to_pil = T.ToPILImage()

    def truncate_layers(self):
        if self.layer_idx and self.layer_idx != -1:
            if hasattr(self.vit.encoder, "layers"):
                self.vit.encoder.layers = self.vit.encoder.layers[: self.layer_idx]
            elif hasattr(self.vit.encoder, "layer"):
                self.vit.encoder.layer = self.vit.encoder.layer[: self.layer_idx]
            else:
                raise ValueError("Unknown model architecture cannot be truncated.")

    def forward_vit(self, inputs):
        device = inputs.device
        inputs = [self.to_pil(im) for im in inputs]
        inputs = self.processor(images=inputs, return_tensors="pt").to(device)
        outputs = self.vit(
            **inputs,
            output_hidden_states=(self.layer_idx is None),
        )

        if self.layer_idx:
            return outputs.last_hidden_state
        else:
            return torch.stack(outputs.hidden_states, dim=-1)


class TiViT_OpenCLIP(BaseTiViT):
    def __init__(self, processor, vit, layer_idx, aggregation, patch_size, stride):
        self.hidden_representations = {}
        super().__init__(processor, vit, layer_idx, aggregation, patch_size, stride)
        self.processor.transforms = [self.processor.transforms[-1]]

    def truncate_layers(self):
        if self.layer_idx is not None and self.layer_idx != -1:
            self.vit.transformer.resblocks = self.vit.transformer.resblocks[
                : self.layer_idx
            ]

    def forward_vit(self, inputs):
        hidden_states = []

        inputs = self.processor(inputs)

        x = self.vit._embeds(inputs)
        hidden_states.append(x)

        for blk in self.vit.transformer.resblocks:
            x = blk(x)
            hidden_states.append(x)

        if self.layer_idx is not None:
            return x
        else:
            return torch.stack(hidden_states, dim=-1)
