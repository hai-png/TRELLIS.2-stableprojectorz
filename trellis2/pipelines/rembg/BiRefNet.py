# File: trellis2/pipelines/rembg/BiRefNet.py
# trellis2/pipelines/rembg/BiRefNet.py
from typing import *
from transformers import AutoModelForImageSegmentation, PreTrainedModel
import torch
from torchvision import transforms
from PIL import Image
import os # <--- Added


def _patch_all_tied_weights_keys():
    """Compatibility patch for transformers >= 4.49.

    In transformers 4.49+, ``_move_missing_keys_from_meta_to_device`` calls
    ``self.all_tied_weights_keys.keys()``.  Custom model classes loaded via
    ``trust_remote_code=True`` (e.g. ZhengPeng7/BiRefNet) may have been built
    against an older transformers version and lack this property.

    We ensure the property exists on ``PreTrainedModel`` **and** we wrap the
    internal method so that, even if a dynamically-loaded subclass somehow
    misses the property, we add it on-the-fly instead of crashing.
    """
    # --- 1. Ensure the property exists on PreTrainedModel itself ---
    if not isinstance(getattr(PreTrainedModel, 'all_tied_weights_keys', None), property):
        @property
        def all_tied_weights_keys(self):
            """Return a dict mapping every tied-weight key to its group."""
            tied_groups = getattr(self, '_tied_weights_keys', None) or []
            all_keys = {}
            for group in tied_groups:
                for key in group:
                    all_keys[key] = group
            return all_keys
        PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys

    # --- 2. Wrap _move_missing_keys_from_meta_to_device as a safety-net ---
    _orig = getattr(PreTrainedModel, '_move_missing_keys_from_meta_to_device', None)
    if _orig is not None and not getattr(_orig, '_patched_atkw', False):
        def _safe_move_missing(self, *args, **kwargs):
            try:
                return _orig(self, *args, **kwargs)
            except AttributeError as exc:
                if 'all_tied_weights_keys' in str(exc):
                    # Dynamically add the missing property to *this* model class
                    cls = type(self)
                    if not isinstance(getattr(cls, 'all_tied_weights_keys', None), property):
                        @property
                        def _atwk(inner_self):
                            tied_groups = getattr(inner_self, '_tied_weights_keys', None) or []
                            all_keys = {}
                            for group in tied_groups:
                                for key in group:
                                    all_keys[key] = group
                            return all_keys
                        cls.all_tied_weights_keys = _atwk
                    return _orig(self, *args, **kwargs)
                raise
        _safe_move_missing._patched_atkw = True
        PreTrainedModel._move_missing_keys_from_meta_to_device = _safe_move_missing


# Apply the patch once when the module is imported
_patch_all_tied_weights_keys()


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        
        # 1. Check relative to Current Working Directory
        cwd_path = os.path.join(os.getcwd(), "MODELS", "RMBG-2.0")
        
        # 2. Check relative to this script file
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "MODELS", "RMBG-2.0"))
        
        target_path = None
        use_local = False

        # We look for model.safetensors or config.json to confirm validity
        if os.path.exists(os.path.join(cwd_path, "config.json")):
            target_path = cwd_path
            use_local = True
        elif os.path.exists(os.path.join(script_path, "config.json")):
            target_path = script_path
            use_local = True
        else:
            # Fallback to whatever was passed in (likely briaai/RMBG-2.0 which will fail if not logged in)
            target_path = model_name
            use_local = False
            
        if use_local:
            print(f"[INFO] Local RMBG-2.0 found. Loading from: {target_path}")
        else:
            print(f"[INFO] Local RMBG-2.0 NOT found. Attempting download/access: {model_name}")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            target_path, 
            trust_remote_code=True,
            local_files_only=use_local # Forces offline mode if local files are found
        )
        
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image