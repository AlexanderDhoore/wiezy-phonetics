from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer

cfg = OmegaConf.load("diar_infer_telephonic.yaml")
cfg.diarizer.manifest_filepath = "manifest.json"
cfg.diarizer.out_dir = "outputs"

model = ClusteringDiarizer(cfg=cfg)
model.diarize()

