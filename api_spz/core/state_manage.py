# api_spz/core/state_manage.py
import logging
import torch
from pathlib import Path

logger = logging.getLogger("trellis2_api")


def _apply_patches():
    """Apply compatibility patches for flex_gemm.

    On Windows, flex_gemm lacks native triton support, so we inject a fallback.
    On Linux, triton is available natively, so we only apply patches if triton
    is genuinely missing.
    """
    import os
    is_linux = os.name != 'nt'

    try:
        import flex_gemm.ops.spconv as spconv
        from flex_gemm.ops.spconv import Algorithm
        try:
            import triton as _triton_check  # noqa: F401
            if is_linux:
                print("[INIT] flex_gemm: native triton available, using default algorithm.")
            else:
                spconv.ALGORITHM = Algorithm.EXPLICIT_GEMM
                print("[INIT] flex_gemm EXPLICIT_GEMM patch applied (Windows).")
        except ImportError:
            spconv.ALGORITHM = Algorithm.EXPLICIT_GEMM
            print("[INIT] flex_gemm EXPLICIT_GEMM patch applied (triton not available).")
    except ImportError:
        print("[INIT] Could not patch flex_gemm spconv.")
    except Exception as e:
        print(f"[INIT] flex_gemm spconv patch failed: {e}")

    try:
        import flex_gemm.kernels as _fgk
        if not hasattr(_fgk, 'triton'):
            try:
                import triton as _triton_check  # noqa: F401
                print("[INIT] flex_gemm.kernels has no 'triton' attr, but triton is installed. "
                      "This may indicate a build issue with flex_gemm.")
            except ImportError:
                class _TritonFallback:
                    @staticmethod
                    def indice_weighed_sum_fwd(feats, indices, weights):
                        N = feats.shape[0]
                        idx = indices.long().clamp(min=0, max=N - 1)  # [M, 8]
                        M_shape, K = idx.shape
                        C = feats.shape[-1]
                        out = torch.zeros((M_shape, C), dtype=feats.dtype, device=feats.device)
                        for i in range(K):
                            out += feats[idx[:, i]] * weights[:, i].unsqueeze(-1)
                        return out

                    @staticmethod
                    def indice_weighed_sum_bwd_input(grad_output, indices, weights, N):
                        M, C = grad_output.shape
                        idx = indices.long().clamp(min=0, max=N - 1)
                        weighted_grad = grad_output.unsqueeze(1) * weights.unsqueeze(-1)  # [M, 8, C]
                        grad_feats = torch.zeros(N, C, device=grad_output.device, dtype=grad_output.dtype)
                        grad_feats.scatter_add_(0, idx.reshape(-1, 1).expand(-1, C), weighted_grad.reshape(-1, C))
                        return grad_feats

                _fgk.triton = _TritonFallback()
                print("[INIT] flex_gemm Triton fallback patch applied (triton not installed).")
    except ImportError:
        print("[INIT] Could not patch flex_gemm triton fallback.")
    except Exception as e:
        print(f"[INIT] flex_gemm triton patch failed: {e}")


class TrellisState:
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.pipeline = None

    def cleanup(self):
        """Clean up resources on shutdown."""
        self.pipeline = None
        torch.cuda.empty_cache()
        logger.info("Pipeline resources cleaned up")

    def initialize_pipeline(self, device=None):
        """Load the Trellis 2 pipeline and move it to the target device."""
        import os
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        os.environ["PYTORCH_ALLOC_CONF"] = "garbage_collection_threshold:0.65"
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        os.environ.setdefault('SPARSE_DEBUG', '0')
        # SETUPTOOLS_USE_DISTUTILS=stdlib is a Windows-only workaround.
        # On Linux it can cause issues with distutils, so we skip it.
        if os.name == 'nt':
            os.environ.setdefault('SETUPTOOLS_USE_DISTUTILS', 'stdlib')

        _apply_patches()

        # Apply transformers compatibility patch BEFORE any model loading.
        # In transformers >= 4.49, post_init() SETS all_tied_weights_keys and
        # _move_missing_keys_from_meta_to_device() GETS it.  Custom models
        # loaded via trust_remote_code=True may lack it.  We need a read-WRITE
        # property (not read-only) so post_init() can set it.
        from transformers import PreTrainedModel
        existing = getattr(PreTrainedModel, 'all_tied_weights_keys', None)
        needs_patch = (
            existing is None
            or (isinstance(existing, property) and existing.fset is None)
        )
        if needs_patch:
            @property
            def all_tied_weights_keys(self):
                val = self.__dict__.get('all_tied_weights_keys')
                if val is not None:
                    return val
                tied_groups = getattr(self, '_tied_weights_keys', None) or []
                all_keys = {}
                for group in tied_groups:
                    for key in group:
                        all_keys[key] = group
                return all_keys

            @all_tied_weights_keys.setter
            def all_tied_weights_keys(self, value):
                self.__dict__['all_tied_weights_keys'] = value

            PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys

        if not hasattr(PreTrainedModel, 'get_expanded_tied_weights_keys'):
            def get_expanded_tied_weights_keys(self, all_submodels=False):
                tied_groups = getattr(self, '_tied_weights_keys', None) or []
                all_keys = {}
                for group in tied_groups:
                    for key in group:
                        all_keys[key] = group
                return all_keys
            PreTrainedModel.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys

        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        if device is None:
            device = "cuda"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. A CUDA-capable GPU with drivers is required.\n"
                f"  torch version: {torch.__version__}\n"
                f"  Run 'nvidia-smi' in a terminal to check GPU status."
            )
        self._device = device

        print("[INIT] Loading Trellis 2 pipeline, please wait...")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
        self.pipeline.to(device)
        print("[INIT] Trellis 2 pipeline ready.")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info(f"VRAM allocated at startup: {allocated:.1f}MB")
            print(f"VRAM allocated at startup: {allocated:.1f}MB")


# Global state instance
state = TrellisState()