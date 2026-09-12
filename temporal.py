import torch
from .core import correspondence


class TemporalState:
    def __init__(self, height, width, frames, seed, tau, radius, keep_fraction, memory):
        self.height, self.width, self.frames = height, width, frames
        self.tokens = height * width
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.tau, self.radius, self.keep_fraction = tau, radius, keep_fraction
        self.memory = memory
        self.phase, self.step, self.collect = "invert", 0, False
        self.maps = None
        self.velocity_outputs = []
        self.trajectory = None

    def extract(self, args, context):
        result = context["original_block"](args)
        if self.collect:
            features = result["img"][:, :self.tokens]
            if features.shape[0] != self.frames:
                raise ValueError("EditVid requires one conditioning batch containing all chunk frames.")
            anchor = self.memory.setdefault("source_anchor", features[0].detach().cpu().clone()).to(features)
            self.maps = [correspondence(anchor, frame, self.width, self.tau, self.radius,
                                        self.keep_fraction, self.generator) for frame in features]
            if self.memory.get("first_chunk", True):
                self.maps[0] = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        return result

    def attention(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        if self.phase != "edit":
            return {}
        if attn_mask is not None:
            raise ValueError("EditVid does not yet support masked attention.")
        if q.shape[0] != self.frames:
            raise ValueError("EditVid does not support CFG batching or split frame batches.")
        from comfy.ldm.flux.math import apply_rope
        # Rotate current Q/K first. Cached keys already carry their spatial RoPE;
        # reference-image and text positions are never appended as video memory.
        q, k = apply_rope(q, k, pe) if pe is not None else (q, k)
        start = extra_options["img_slice"][0]
        key = (self.step, extra_options["block_type"], extra_options["block_index"])
        current_k = k[:, :, start:start + self.tokens]
        current_v = v[:, :, start:start + self.tokens]
        previous = self.memory.setdefault("kv", {}).get(key)
        first_k = torch.zeros_like(current_k[:1]) if previous is None else previous[0].to(k)
        first_v = torch.zeros_like(current_v[:1]) if previous is None else previous[1].to(v)
        appended_k = torch.cat((first_k, current_k[:-1]), dim=0)
        appended_v = torch.cat((first_v, current_v[:-1]), dim=0)
        self.memory["kv"][key] = (current_k[-1:].detach().cpu().clone(), current_v[-1:].detach().cpu().clone())
        return {"q": q, "k": torch.cat((k, appended_k), dim=2),
                "v": torch.cat((v, appended_v), dim=2), "pe": None}

    def output(self, attention, options):
        if self.phase != "edit" or self.maps is None:
            return attention
        layer = options["block_index"] + (8 if options["block_type"] == "single" else 0)
        if layer >= 12:
            return attention
        start = options["img_slice"][0]
        key = (self.step, layer)
        anchors = self.memory.setdefault("output_anchors", {})
        anchor = anchors.setdefault(key, attention[0, start:start + self.tokens].detach().cpu().clone()).to(attention)
        result = attention.clone()
        for frame, (source, target) in enumerate(self.maps):
            result[frame, start + target.to(result.device)] = anchor[source.to(result.device)]
        return result
