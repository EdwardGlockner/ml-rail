"""CLIP zero-shot classification (Radford et al., 2021).

Encodes images and prompted class names and returns softmax-scaled cosine similarities. Also
provides multi-layer image features (`encode_multi`).
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipZeroShotClassifier:
    """Frozen HuggingFace CLIPModel (image tower, text tower and the joint projection): image
    embeddings, text embeddings, and zero-shot probabilities from CLIP's learned logit scale."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embed_dim: int = self.model.config.projection_dim

    def _preprocess_images(self, images: np.ndarray) -> torch.Tensor:
        """images: (N, H, W, 3) uint8, already resized to this model's expected input (224x224
        for both ViT-B/32 and ViT-B/16) -- resizing is the caller's job."""
        x = torch.from_numpy(images).to(self.device).float() / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def encode_images(self, images: np.ndarray, batch_size: int = 64) -> torch.Tensor:
        """Returns L2-normalized (N, D) embeddings in CLIP's shared image-text space (the
        projected `get_image_features` output)."""
        feats = []
        for i in range(0, len(images), batch_size):
            batch = self._preprocess_images(images[i : i + batch_size])
            out = self.model.get_image_features(pixel_values=batch)
            # this transformers version wraps the projected embedding in pooler_output
            # instead of returning a plain tensor -- confirmed directly against the installed
            # version rather than assumed, since this is a real breaking API change.
            embed = out.pooler_output if hasattr(out, "pooler_output") else out
            feats.append(embed.float().cpu())
        feats = torch.cat(feats, dim=0)
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def encode_images_and_patches(self, images: np.ndarray, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        """Like `encode_images`, plus each patch token mapped into the same shared image-text space
        (post-layernorm, then CLIP's visual projection -- the same two steps CLIP applies to its
        class token), L2-normalized. Returns ((N, D) global, (N, P, D) patches, float16 to halve
        memory). Used for text-guided patch attention (CALIP-style) and region re-ranking."""
        glob, patches = [], []
        vm, proj = self.model.vision_model, self.model.visual_projection
        for i in range(0, len(images), batch_size):
            out = vm(pixel_values=self._preprocess_images(images[i : i + batch_size]))
            g = proj(out.pooler_output).float()
            p = proj(vm.post_layernorm(out.last_hidden_state[:, 1:])).float()
            glob.append((g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)).cpu())
            patches.append((p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)).half().cpu())
        return torch.cat(glob), torch.cat(patches)

    @torch.no_grad()
    def encode_multi(self, images: np.ndarray, layers: tuple[int, ...] = (6, 9), batch_size: int = 64) -> dict[str, np.ndarray]:
        """Everything an earlier experiment needs from one forward pass: the global
        embedding, projected patch tokens (float16), and the class token of each intermediate
        layer in `layers` after CLIP's post-layernorm (not projected: those layers were never
        trained to live in the text space). All L2-normalized."""
        vm, proj = self.model.vision_model, self.model.visual_projection
        out = {"global": [], "patches": [], **{f"cls{l}": [] for l in layers}}

        def norm(x):
            return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        for i in range(0, len(images), batch_size):
            o = vm(pixel_values=self._preprocess_images(images[i : i + batch_size]), output_hidden_states=True)
            out["global"].append(norm(proj(o.pooler_output).float()).cpu())
            out["patches"].append(norm(proj(vm.post_layernorm(o.last_hidden_state[:, 1:])).float()).half().cpu())
            for l in layers:
                out[f"cls{l}"].append(norm(vm.post_layernorm(o.hidden_states[l][:, 0]).float()).cpu())
        return {k: torch.cat(v).numpy() for k, v in out.items()}

    @torch.no_grad()
    def encode_text(self, class_names: list[str], prompt_template: str = "a photo of a {}.") -> torch.Tensor:
        """Returns L2-normalized (C, D) text embeddings, one per class name, using the given
        prompt template (Zhu et al.'s Table S1: "a photo of a {label}." for CIFAR100/ImageNet/
        SUN397, task-specific templates otherwise)."""
        prompts = [prompt_template.format(name.replace("_", " ")) for name in class_names]
        tokens = self.processor(text=prompts, return_tensors="pt", padding=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        out = self.model.get_text_features(**tokens)
        embed = out.pooler_output if hasattr(out, "pooler_output") else out
        embed = embed.float().cpu()
        return embed / embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def zero_shot_probs(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> np.ndarray:
        """image_embeds: (N, D), text_embeds: (C, D), both L2-normalized. Returns (N, C)
        softmax probabilities using CLIP's own learned logit_scale as temperature (Radford et
        al., 2021) -- the model's own calibration, not a hand-picked value like
        evaluation/fusion.py:softmax_probs_from_centroids uses for this project's other
        classifiers, since CLIP was trained with this exact scale baked in."""
        logit_scale = self.model.logit_scale.exp().item()
        logits = logit_scale * (image_embeds @ text_embeds.T)
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = torch.softmax(logits, dim=1)
        return probs.numpy()
