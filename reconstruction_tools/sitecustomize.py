"""Runtime-only Transformers compatibility shim; attack code stays untouched."""
import torch
from transformers import AutoModelForTokenClassification

_original = AutoModelForTokenClassification.from_pretrained

def _from_pretrained(cls, *args, **kwargs):
    # ponytail: remove unsupported auto placement for BERT, then place the
    # unchanged model on the selected CUDA device; split placement is not needed.
    if kwargs.get("device_map") == "auto":
        kwargs.pop("device_map")
        model = _original(*args, **kwargs)
        return model.to("cuda" if torch.cuda.is_available() else "cpu")
    return _original(*args, **kwargs)

AutoModelForTokenClassification.from_pretrained = classmethod(_from_pretrained)
