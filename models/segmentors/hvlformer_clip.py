from mmseg.models.builder import SEGMENTORS

from .tqdm_clip import tqdm_CLIP


@SEGMENTORS.register_module()
class HVLFormer_CLIP(tqdm_CLIP):
    """CLIP-backed HVLFormer segmentor.

    The implementation keeps TQDM's training/inference plumbing and swaps the
    decode head to HVLFormerHead in config files.
    """

    pass
