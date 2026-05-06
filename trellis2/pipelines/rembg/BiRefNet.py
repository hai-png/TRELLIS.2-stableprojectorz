# File: trellis2/pipelines/rembg/BiRefNet.py
# trellis2/pipelines/rembg/BiRefNet.py
from typing import *
from transformers import AutoModelForImageSegmentation, PreTrainedModel
import torch
from torchvision import transforms
from PIL import Image
import os # <--- Added


def _patch_transformers_compat():
    """Compatibility patch for transformers >= 4.49.

    In transformers 4.49+, two new patterns appear:

    1. ``post_init()`` does ``self.all_tied_weights_keys = ...`` (SET)
    2. ``_move_missing_keys_from_meta_to_device()`` does
       ``self.all_tied_weights_keys.keys()`` (GET)

    Custom model classes loaded via ``trust_remote_code=True``
    (e.g. ZhengPeng7/BiRefNet) may have been built against an older
    transformers version and lack this attribute entirely.  Adding a
    read-only ``@property`` fixes the GET but breaks the SET ("property
    has no setter").  Instead we install a read-write property whose
    getter returns the instance-level value if present, or an empty dict
    as a safe default; the setter stores into ``__dict__`` so that
    ``post_init()`` works normally.
    """

    # --- 1. Ensure all_tied_weights_keys is a read-write property ---
    existing = getattr(PreTrainedModel, 'all_tied_weights_keys', None)
    needs_patch = (
        existing is None
        or (isinstance(existing, property) and existing.fset is None)
    )
    if needs_patch:
        @property
        def all_tied_weights_keys(self):
            """Return tied-weights dict; default to empty if not yet set."""
            val = self.__dict__.get('all_tied_weights_keys')
            if val is not None:
                return val
            # Fallback: compute from _tied_weights_keys if available
            tied_groups = getattr(self, '_tied_weights_keys', None) or []
            all_keys = {}
            for group in tied_groups:
                for key in group:
                    all_keys[key] = group
            return all_keys

        @all_tied_weights_keys.setter
        def all_tied_weights_keys(self, value):
            """Allow post_init() to set the attribute normally."""
            self.__dict__['all_tied_weights_keys'] = value

        PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys

    # --- 2. Ensure get_expanded_tied_weights_keys exists (added in 4.49) ---
    if not hasattr(PreTrainedModel, 'get_expanded_tied_weights_keys'):
        def get_expanded_tied_weights_keys(self, all_submodels=False):
            """Stub for older transformers that lack this method."""
            tied_groups = getattr(self, '_tied_weights_keys', None) or []
            all_keys = {}
            for group in tied_groups:
                for key in group:
                    all_keys[key] = group
            return all_keys
        PreTrainedModel.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys

    # --- 3. Wrap _move_missing_keys_from_meta_to_device as safety-net ---
    _orig = getattr(PreTrainedModel, '_move_missing_keys_from_meta_to_device', None)
    if _orig is not None and not getattr(_orig, '_patched_atkw', False):
        def _safe_move_missing(self, *args, **kwargs):
            try:
                return _orig(self, *args, **kwargs)
            except AttributeError as exc:
                msg = str(exc)
                if 'all_tied_weights_keys' in msg:
                    # Ensure this specific instance has the attribute
                    if 'all_tied_weights_keys' not in self.__dict__:
                        self.__dict__['all_tied_weights_keys'] = {}
                    return _orig(self, *args, **kwargs)
                raise
        _safe_move_missing._patched_atkw = True
        PreTrainedModel._move_missing_keys_from_meta_to_device = _safe_move_missing


# Apply the patch once when the module is imported
_patch_transformers_compat()


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