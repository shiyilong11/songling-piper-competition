"""Isolated learned video-mask backends; import only in a model worker."""

import contextlib
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from piper_sim.calibration import ROOT

METHODS = {
    "sam2_small": (ROOT / ".local/sam2/source", "configs/sam2.1/sam2.1_hiera_s.yaml",
                   ROOT / "models/sam2/sam2.1_hiera_small.pt"),
    "edgetam": (ROOT / ".local/tracking-bench/edgetam", "configs/edgetam.yaml",
                ROOT / ".local/tracking-bench/edgetam/checkpoints/edgetam.pt"),
    "cutie": (ROOT / ".local/tracking-bench/cutie", None,
              ROOT / "models/cutie/cutie-base-mega.pth"),
}



def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def add_source(method):
    path = METHODS[method][0]
    sys.path.insert(0, str(path))
    return path


def build_sam(method, *, video=True):
    source = add_source(method)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("SAM namespace resolved to the wrong isolated source")
    if method == "edgetam":
        # A complete segmentation checkpoint is loaded strictly below; downloading
        # redundant ImageNet initialization is unnecessary and not an inference change.
        from sam2.modeling.backbones import timm as backbone
        original = backbone.create_model
        def create_without_pretrain(*args, **kwargs):
            kwargs["pretrained"] = False
            return original(*args, **kwargs)
        backbone.create_model = create_without_pretrain
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    build = build_sam2_video_predictor if video else build_sam2
    with contextlib.redirect_stdout(sys.stderr):
        model = build(METHODS[method][1], str(METHODS[method][2]),
                      device="cuda", apply_postprocessing=False)
    return model


class CausalFrameStore:
    """Bounded image buffer; raises if a predictor attempts to read any future frame.

    Replaces only native file loading. The model's memory and inference code are
    unmodified. Historical feature/output dictionaries retain native behavior.
    """
    def __init__(self, image_size):
        self.image_size = image_size
        self.current = -1
        self.frames = {}
        self.accesses = []

    def append(self, rgb):
        self.current += 1
        resized = np.array(Image.fromarray(rgb).resize((self.image_size, self.image_size)))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor(
            [.229, .224, .225])[:, None, None]
        self.frames[self.current] = tensor
        # Preserve the initial prompt frame and the current frame, no future video.
        for index in list(self.frames):
            if index not in (0, self.current):
                del self.frames[index]

    def __len__(self):
        return self.current + 1

    def __getitem__(self, index):
        self.accesses.append((self.current, int(index)))
        if index > self.current:
            raise RuntimeError("Future frame access is forbidden")
        if index not in self.frames:
            raise RuntimeError("Predictor requested an evicted RGB frame")
        return self.frames[index]


class SAMStream:
    def __init__(self, method):
        self.model = build_sam(method)
        import sam2.sam2_video_predictor as module
        self.module = module
        # A one-frame propagation call otherwise prints a progress bar each tick.
        module.tqdm = lambda iterable, **kwargs: iterable
        self.store = CausalFrameStore(self.model.image_size)
        self.state = None

    def initialize(self, rgb, mask):
        self.store.append(rgb)
        height, width = rgb.shape[:2]
        original_loader = self.module.load_video_frames
        self.module.load_video_frames = lambda **kwargs: (self.store, height, width)
        try:
            self.state = self.model.init_state("causal-benchmark://frames",
                offload_video_to_cpu=True, offload_state_to_cpu=False)
        finally:
            self.module.load_video_frames = original_loader
        self.model.add_new_mask(self.state, frame_idx=0, obj_id=1, mask=mask)
        outputs = list(self.model.propagate_in_video(self.state, start_frame_idx=0,
                                                     max_frame_num_to_track=0))
        if len(outputs) != 1 or outputs[0][0] != 0:
            raise RuntimeError("Unexpected initialization frame output")
        return (outputs[0][2][0, 0] > 0).cpu().numpy()

    def step(self, rgb, index):
        self.store.append(rgb)
        if self.store.current != index:
            raise RuntimeError("Frame counter mismatch")
        self.state["num_frames"] = len(self.store)
        outputs = list(self.model.propagate_in_video(self.state, start_frame_idx=index,
                                                     max_frame_num_to_track=0))
        if len(outputs) != 1 or outputs[0][0] != index:
            raise RuntimeError("Predictor did not return exactly the current frame")
        return (outputs[0][2][0, 0] > 0).cpu().numpy()


class CutieStream:
    def __init__(self):
        source = add_source("cutie")
        from cutie.model.utils import resnet
        from hydra import compose, initialize_config_dir
        # Strict full-checkpoint loading below replaces every initialized weight.
        for name in ("resnet18", "resnet50"):
            original = getattr(resnet, name)
            def no_pretrain(*args, _original=original, **kwargs):
                kwargs["pretrained"] = False
                return _original(*args, **kwargs)
            setattr(resnet, name, no_pretrain)
        from cutie.inference.inference_core import InferenceCore
        from cutie.inference.utils.args_utils import get_dataset_cfg
        from cutie.model.cutie import CUTIE
        with initialize_config_dir(version_base="1.3.2", config_dir=str(source / "cutie/config")):
            cfg = compose(config_name="eval_config", overrides=["use_long_term=false",
                "mem_every=5", "max_internal_size=480", "flip_aug=false"])
        get_dataset_cfg(cfg)
        self.model = CUTIE(cfg).cuda().eval()
        state = torch.load(METHODS["cutie"][2], map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.processor = InferenceCore(self.model, cfg=cfg)

    def image(self, rgb):
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).to("cuda").float() / 255.0

    def initialize(self, rgb, mask):
        probability = self.processor.step(self.image(rgb),
            torch.from_numpy(mask.astype(np.int64)).cuda(), objects=[1])
        return (probability.argmax(dim=0) == 1).cpu().numpy()

    def step(self, rgb, index):
        probability = self.processor.step(self.image(rgb))
        return (probability.argmax(dim=0) == 1).cpu().numpy()
