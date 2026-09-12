"""Native ComfyUI model patches and sampler orchestration. No Diffusers dependency."""
import copy
import torch
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
from .core import integrate
from .temporal import TemporalState


class EditVidPass(comfy.samplers.Sampler):
    def __init__(self, state, invert, anchor_strength):
        self.state, self.invert, self.anchor_strength = state, invert, anchor_strength

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None:
            raise ValueError("EditVid does not support latent noise masks.")
        state = self.state
        state.phase = "invert" if self.invert else "edit"
        extraction_step = int((sigmas.flip(0)[:-1] - .25).abs().argmin())

        def velocity(x, sigma, step, midpoint):
            comfy.model_management.throw_exception_if_processing_interrupted()
            state.step = step
            state.collect = self.invert and not midpoint and step == extraction_step
            state.velocity_outputs.clear()
            model_wrap(x, sigma.expand(x.shape[0]), **extra_args)
            if len(state.velocity_outputs) != 1 or state.velocity_outputs[0].shape != x.shape:
                raise RuntimeError("EditVid requires a single full-frame model evaluation; regional/CFG batching is unsupported.")
            return state.velocity_outputs.pop()

        try:
            result, history = integrate(latent_image, sigmas, velocity, self.invert,
                                        self.anchor_strength, state.trajectory, callback)
            if self.invert:
                state.trajectory = history
            return result
        finally:
            state.collect = False
            state.velocity_outputs.clear()


def conditioning_for_frames(conditioning, frames, start, total):
    if len(conditioning) != 1:
        raise ValueError("Use a single plain Text Encode output for each EditVid prompt.")
    embedding, metadata = conditioning[0]
    if any(key in metadata for key in ("area", "mask", "control", "hooks", "start_percent", "end_percent")):
        raise ValueError("Regional, scheduled and ControlNet conditioning are not supported yet.")
    if embedding.shape[0] not in (1, total):
        raise ValueError("Prompt batch must be one or match the complete video frame count.")
    metadata = metadata.copy()
    for key, value in list(metadata.items()):
        if torch.is_tensor(value) and value.ndim and value.shape[0] == total:
            metadata[key] = value[start:start + len(frames)]
    # Each frame receives its own source image in Klein's first reference slot.
    # External image references are deliberately unsupported in this first version.
    if metadata.get("reference_latents"):
        raise ValueError("Pass plain text conditioning; EditVid supplies the source frame references.")
    metadata["reference_latents"] = [frames]
    metadata["reference_latents_method"] = "index"
    if embedding.shape[0] != 1:
        embedding = embedding[start:start + len(frames)]
    return [[embedding, metadata]]


class EditVidSampler:
    CATEGORY = "sampling/EditVid"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("edited_latents",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "source": ("CONDITIONING",), "edit": ("CONDITIONING",),
            "latent_video": ("LATENT",), "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 14, "min": 0, "max": 0xffffffffffffffff}),
            "chunk_size": ("INT", {"default": 4, "min": 1, "max": 64}),
            "anchor_strength": ("FLOAT", {"default": .6, "min": 0., "max": 1., "step": .05}),
            "correspondence_threshold": ("FLOAT", {"default": .4, "min": -1., "max": 1., "step": .05}),
            "cycle_radius": ("FLOAT", {"default": 1.5, "min": 0., "max": 16., "step": .1}),
            "token_keep_fraction": ("FLOAT", {"default": .5, "min": 0., "max": 1., "step": .05}),
        }}

    @staticmethod
    def validate_model(model):
        diffusion = model.get_model_object("diffusion_model")
        params = getattr(diffusion, "params", None)
        if (params is None or params.hidden_size != 4096 or len(diffusion.double_blocks) != 8
                or params.guidance_embed or not params.global_modulation):
            raise ValueError("This version supports FLUX.2 Klein 9B distilled only. Use its standard Comfy model loader.")
        if diffusion.patch_size != 1:
            raise ValueError("Unsupported Klein latent patch layout.")

    @torch.no_grad()
    def sample(self, model, source, edit, latent_video, sigmas, seed, chunk_size,
               anchor_strength, correspondence_threshold, cycle_radius, token_keep_fraction):
        self.validate_model(model)
        frames = latent_video["samples"]
        if frames.ndim != 4 or frames.shape[1] != 128 or len(frames) == 0:
            raise ValueError("VAE Encode the video with the Klein VAE: expected [frames, 128, height, width].")
        if "noise_mask" in latent_video:
            raise ValueError("Remove the latent noise mask before EditVid sampling.")
        if sigmas.ndim != 1 or len(sigmas) < 2 or not torch.isfinite(sigmas).all():
            raise ValueError("Connect a valid Flux2Scheduler SIGMAS output.")
        if not (torch.all(sigmas[:-1] > sigmas[1:]) and abs(float(sigmas[0]) - 1) < 1e-6
                and float(sigmas[-1]) == 0):
            raise ValueError("EditVid requires a strictly decreasing full sigma schedule from 1 to 0.")
        if model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit"):
            raise ValueError("Use an unpatched Klein model; existing block replacements conflict with feature extraction.")
        memory, results = {}, []
        steps = len(sigmas) - 1
        chunks = (len(frames) + chunk_size - 1) // chunk_size
        progress = comfy.utils.ProgressBar(chunks * 2 * steps)
        try:
            for chunk_index, start in enumerate(range(0, len(frames), chunk_size)):
                comfy.model_management.throw_exception_if_processing_interrupted()
                chunk = frames[start:start + chunk_size]
                memory["first_chunk"] = start == 0
                state = TemporalState(*chunk.shape[-2:], len(chunk), (seed + start) % 2**64,
                                      correspondence_threshold, cycle_radius, token_keep_fraction, memory)
                patched = model.clone()
                patched.set_model_attn1_patch(state.attention)
                patched.set_model_attn1_output_patch(state.output)
                patched.set_model_patch_replace(state.extract, "dit", "double_block", 6)
                # Capture raw flow velocity before Comfy converts it to x0. This
                # avoids (x-x0)/sigma and its singularity at the inversion endpoint.
                sampling = copy.copy(model.get_model_object("model_sampling"))
                original_denoised = sampling.calculate_denoised

                def capture(sigma, output, model_input, state=state, original=original_denoised):
                    state.velocity_outputs.append(output.float())
                    return original(sigma, output, model_input)

                sampling.calculate_denoised = capture
                patched.add_object_patch("model_sampling", sampling)
                current = chunk
                for invert, conditioning in ((True, source), (False, edit)):
                    cond = conditioning_for_frames(conditioning, chunk, start, len(frames))
                    offset = (chunk_index * 2 + (0 if invert else 1)) * steps

                    def callback(i, denoised, x, total, offset=offset):
                        progress.update_absolute(offset + i + 1)

                    current = comfy.sample.sample_custom(
                        patched, torch.zeros_like(chunk, device="cpu"), 1.0,
                        EditVidPass(state, invert, anchor_strength), sigmas, cond, cond,
                        current, callback=callback, seed=seed)
                results.append(current.cpu())
                state.trajectory = None
            return ({"samples": torch.cat(results)},)
        finally:
            memory.clear()
