"""Versioned MiniMax H3 proof/production render profiles.

The proof profile is intentionally non-deliverable.  It exists to validate the
prompt, visible action, camera path, identity references and native AV timing at
roughly one sixth of the formal render cost before a reviewer promotes the
exact prompt/reference hashes to production.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


H3_RENDER_PROFILE_CONTRACT = "ai-manga.h3-render-profile/v1"
PROOF_THEN_PRODUCTION = "proof_then_production"
DIRECT_PRODUCTION = "direct_production"
DEFAULT_PRODUCTION_STRATEGY = PROOF_THEN_PRODUCTION

# MiniMax H3 runs at 24 fps on the 17k+5 frame lattice.  124 frames is the
# first point in the official trained range (~5 seconds); 243 frames preserves
# the project's established ~10.125-second source material contract.
H3_RENDER_PROFILES: dict[str, dict[str, Any]] = {
    "proof": {
        "profile_id": "h3-proof-v1",
        "label": "低成本预演",
        "purpose": "验证提示词、动作、人物参考、镜头与原生音画；禁止交付",
        "frame_count": 124,
        "duration_seconds": 124 / 24,
        "megapixels": 0.4,
        "turbo_steps": 6,
        "ref_image_size": "match",
        "reference_fidelity": "fast",
        "delivery_eligible": False,
    },
    "production": {
        "profile_id": "h3-production-v1",
        "label": "正式生产",
        "purpose": "使用已批准的预演提示词与参考资产生成可进入发布 QA 的源素材",
        "frame_count": 243,
        "duration_seconds": 243 / 24,
        "megapixels": 0.9,
        "turbo_steps": 8,
        "ref_image_size": "max",
        "reference_fidelity": "identity",
        "delivery_eligible": True,
    },
    # T8-style FastH3 VSA 4-step preview. Requires T8 custom node installed
    # AND the FastVideo FastH3 VSA LoRA in models/loras/FastH3-VSA/.
    # Tradeoff: 4 NFE + VSA learned-gate sparse attention; in motion-heavy
    # long video types quality may drop vs proof (6 steps, dense). Default
    # OFF — use only when explicitly enabled.
    "proof_fast": {
        "profile_id": "h3-proof-fast-v1",
        "label": "FastH3 VSA 4 步极速预演",
        "purpose": "T8 FastH3 VSA 4 步 + 90% 稀疏 attention；最便宜的预演",
        "frame_count": 124,
        "duration_seconds": 124 / 24,
        "megapixels": 0.4,
        "turbo_steps": 4,
        "ref_image_size": "match",
        "reference_fidelity": "vsa_sparse",
        "delivery_eligible": False,
        "requires": "t8mars/comfyui-minimax-h3-audio-T8 + FastH3-VSA LoRA",
    },
}


def normalize_production_strategy(value: Any) -> str:
    strategy = str(value or DEFAULT_PRODUCTION_STRATEGY).strip().lower()
    if strategy not in {PROOF_THEN_PRODUCTION, DIRECT_PRODUCTION}:
        raise ValueError(
            "production_strategy must be 'proof_then_production' or 'direct_production'"
        )
    return strategy


def preview_is_promoted(metadata: Mapping[str, Any] | None) -> bool:
    promotion = (metadata or {}).get("preview_promotion")
    return bool(
        isinstance(promotion, Mapping)
        and promotion.get("status") == "approved"
        and promotion.get("artifact_sha256")
        and promotion.get("decoded_visual_sha256")
        and promotion.get("prompt_sha256")
        and promotion.get("reference_bundle_sha256")
    )


def resolve_render_profile(
    strategy: Any = DEFAULT_PRODUCTION_STRATEGY,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_production_strategy(strategy)
    stage = "production" if (
        normalized == DIRECT_PRODUCTION or preview_is_promoted(metadata)
    ) else "proof"
    profile = deepcopy(H3_RENDER_PROFILES[stage])
    profile.update({
        "contract_version": H3_RENDER_PROFILE_CONTRACT,
        "stage": stage,
        "production_strategy": normalized,
    })
    return profile


def apply_render_profile(
    settings: Mapping[str, Any] | None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(settings or {})
    profile = resolve_render_profile(
        merged.get("production_strategy"), metadata=metadata,
    )
    resolved_values = {
        "production_strategy": profile["production_strategy"],
        "render_profile_contract": profile["contract_version"],
        "render_profile": profile["stage"],
        "render_profile_id": profile["profile_id"],
        "render_profile_label": profile["label"],
        "render_profile_purpose": profile["purpose"],
        "duration_seconds": profile["duration_seconds"],
        "source_generation_duration_seconds": profile["duration_seconds"],
        "frame_count": profile["frame_count"],
        "megapixels": profile["megapixels"],
        "turbo_steps": profile["turbo_steps"],
        "ref_image_size": profile["ref_image_size"],
        "reference_fidelity": profile["reference_fidelity"],
        "delivery_eligible": profile["delivery_eligible"],
    }
    if profile["production_strategy"] == DIRECT_PRODUCTION:
        # Backward-compatible expert path: an explicit direct-production
        # contract may retain its chosen duration/resolution/fidelity/steps.
        # The Web default uses proof_then_production, whose two stages remain
        # fully locked and reproducible.
        for key, value in resolved_values.items():
            if key in {
                "duration_seconds", "source_generation_duration_seconds",
                "megapixels", "turbo_steps", "ref_image_size", "reference_fidelity",
            }:
                merged.setdefault(key, value)
            else:
                merged[key] = value
    else:
        merged.update(resolved_values)
    return merged


def profile_cost_summary() -> dict[str, Any]:
    proof = H3_RENDER_PROFILES["proof"]
    production = H3_RENDER_PROFILES["production"]

    def proxy(profile: Mapping[str, Any]) -> float:
        return (
            float(profile["megapixels"])
            * float(profile["duration_seconds"])
            * int(profile["turbo_steps"])
        )

    proof_proxy = proxy(proof)
    production_proxy = proxy(production)
    return {
        "contract_version": H3_RENDER_PROFILE_CONTRACT,
        "proof": deepcopy(proof),
        "production": deepcopy(production),
        "proof_relative_compute": proof_proxy / production_proxy,
        "method": "megapixels × duration_seconds × turbo_steps; planning proxy only",
    }
