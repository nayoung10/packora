"""CPU-side optional conditioning policy and model ID preparation."""

# ruff: noqa: F722,F821

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from einops import rearrange

from src.data.constants import (
    NUM_ATOM_CHIRALITY_TYPES,
    NUM_BOND_TYPES,
    NUM_STEREOCHEMISTRY_TYPES,
)
from src.utils.tensor_typing import Bool, Float, Int

CONDITIONING_GROUPS = ("template", "stereochemistry", "spacegroup")
CONDITIONING_MODES = ("on", "off", "stochastic")

DEFAULT_CONDITIONING_POLICY = {
    "dropout_probs": {
        "template": 0.50,
        "stereochemistry": 0.50,
        "spacegroup": 0.90,
    },
    "contexts": {
        "train_loss": {
            "template": "stochastic",
            "stereochemistry": "stochastic",
            "spacegroup": "stochastic",
        },
        "val_loss": {
            "template": "off",
            "stereochemistry": "off",
            "spacegroup": "off",
        },
        "train_generation": {
            "template": "off",
            "stereochemistry": "off",
            "spacegroup": "off",
        },
        "validation_generation": {
            "template": "off",
            "stereochemistry": "off",
            "spacegroup": "off",
        },
        "predict": {
            "template": "off",
            "stereochemistry": "off",
            "spacegroup": "off",
        },
    },
}

ATOM_CHIRALITY_MODEL_NULL_ID = NUM_ATOM_CHIRALITY_TYPES + 1
NUM_ATOM_CHIRALITY_MODEL_TYPES = NUM_ATOM_CHIRALITY_TYPES + 2

BOND_TYPE_MODEL_PAD_ID = 0
BOND_TYPE_MODEL_NO_BOND_ID = 1
NUM_BOND_TYPE_MODEL_TYPES = NUM_BOND_TYPES + 1

BOND_STEREOCHEMISTRY_MODEL_NULL_ID = NUM_STEREOCHEMISTRY_TYPES + 1
NUM_BOND_STEREOCHEMISTRY_MODEL_TYPES = NUM_STEREOCHEMISTRY_TYPES + 2

SPACEGROUP_MODEL_UNKNOWN_ID = 0
SPACEGROUP_MODEL_NULL_ID = 231
NUM_SPACEGROUP_MODEL_TYPES = SPACEGROUP_MODEL_NULL_ID + 1


def _to_plain_dict(value: Any) -> dict[str, Any]:
    """Convert mapping-like config objects into plain dictionaries."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        return dict(value)
    return {
        str(key): _to_plain_dict(item) if isinstance(item, Mapping) else item
        for key, item in value.items()
    }


def _normalize_mode(mode: Any) -> str:
    """Normalize YAML mode values into policy mode strings."""
    if isinstance(mode, bool):
        return "on" if mode else "off"
    return str(mode)


def _presence_key(group: str) -> str:
    """Return the conditioning availability field name for one group."""
    return f"{group}_present"


@dataclass(frozen=True)
class ConditioningPolicy:
    """Resolve per-context optional conditioning masks on CPU."""

    contexts: dict[str, dict[str, str]] = field(default_factory=dict)
    dropout_probs: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "ConditioningPolicy":
        """Build a policy from Hydra or plain Python config."""
        if isinstance(cfg, ConditioningPolicy):
            return cfg

        raw = _to_plain_dict(DEFAULT_CONDITIONING_POLICY)
        override = _to_plain_dict(cfg)
        raw["dropout_probs"].update(override.get("dropout_probs", {}))
        raw["contexts"].update(override.get("contexts", {}))
        policy = cls(
            contexts={
                str(context): {
                    str(group): _normalize_mode(mode)
                    for group, mode in group_modes.items()
                }
                for context, group_modes in raw["contexts"].items()
            },
            dropout_probs={
                str(group): float(prob) for group, prob in raw["dropout_probs"].items()
            },
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        """Validate contexts, groups, modes, and dropout probabilities."""
        for group, prob in self.dropout_probs.items():
            if group not in CONDITIONING_GROUPS:
                raise ValueError(f"Unknown conditioning group: {group}")
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"Dropout probability for {group} must be in [0, 1].")

        for context, group_modes in self.contexts.items():
            for group in CONDITIONING_GROUPS:
                if group not in group_modes:
                    raise ValueError(
                        f"Missing conditioning group '{group}' in context '{context}'."
                    )
                mode = group_modes[group]
                if mode not in CONDITIONING_MODES:
                    raise ValueError(
                        f"Unsupported conditioning mode '{mode}' for {context}.{group}."
                    )

    def masks_from_presence(
        self,
        conditioning: Mapping[str, Any],
        context: str,
    ) -> dict[str, Float["b"]]:
        """Sample CPU optional-conditioning masks from presence flags."""
        if context not in self.contexts:
            raise ValueError(f"Unknown conditioning context: {context}")

        first_presence = conditioning[_presence_key(CONDITIONING_GROUPS[0])]
        batch_size = int(first_presence.shape[0])
        device = first_presence.device
        masks: dict[str, Float["b"]] = {}
        for group in CONDITIONING_GROUPS:
            mode = self.contexts[context][group]
            if mode == "on":
                mask = torch.ones(batch_size, device=device, dtype=torch.float32)
            elif mode == "off":
                mask = torch.zeros(batch_size, device=device, dtype=torch.float32)
            else:
                dropout_prob = self.dropout_probs[group]
                keep = torch.rand(batch_size, device=device) >= dropout_prob
                mask = keep.to(dtype=torch.float32)
            presence = conditioning[_presence_key(group)].to(
                device=device,
                dtype=torch.float32,
            )
            if presence.shape != (batch_size,):
                raise ValueError(
                    f"{_presence_key(group)} must have shape ({batch_size},)."
                )
            masks[group] = mask * presence
        return masks


def atom_chirality_model_ids(
    raw_ids: Int["b n"],
    atom_mask: Bool["b n"],
    stereochemistry_mask: Float["b"],
) -> Int["b n"]:
    """Map raw atom chirality labels to padded/null-aware model IDs."""
    real_ids = raw_ids.clamp(min=0, max=NUM_ATOM_CHIRALITY_TYPES - 1) + 1
    null_ids = torch.full_like(raw_ids, ATOM_CHIRALITY_MODEL_NULL_ID)
    use_real = rearrange(stereochemistry_mask > 0.5, "b -> b 1")
    active_ids = torch.where(use_real, real_ids, null_ids)
    return torch.where(atom_mask, active_ids, torch.zeros_like(raw_ids))


def bond_type_model_ids(
    raw_ids: Int["b n n"],
    valid_mol_pair: Bool["b n n"],
) -> Int["b n n"]:
    """Map raw dense bond labels to model IDs for valid same-molecule pairs."""
    real_ids = raw_ids.clamp(min=0, max=NUM_BOND_TYPES - 1) + 1
    return torch.where(valid_mol_pair, real_ids, torch.zeros_like(raw_ids))


def bond_stereochemistry_model_ids(
    raw_ids: Int["b n n"],
    valid_bond_pair: Bool["b n n"],
    stereochemistry_mask: Float["b"],
) -> Int["b n n"]:
    """Map raw bond stereochemistry labels to model IDs on valid bonds only."""
    real_ids = raw_ids.clamp(min=0, max=NUM_STEREOCHEMISTRY_TYPES - 1) + 1
    null_ids = torch.full_like(raw_ids, BOND_STEREOCHEMISTRY_MODEL_NULL_ID)
    use_real = rearrange(stereochemistry_mask > 0.5, "b -> b 1 1")
    active_ids = torch.where(use_real, real_ids, null_ids)
    return torch.where(valid_bond_pair, active_ids, torch.zeros_like(raw_ids))


def spacegroup_model_ids(
    raw_ids: Int["b"],
    spacegroup_mask: Float["b"],
) -> Int["b"]:
    """Map raw space-group numbers to model IDs with UNKNOWN and NULL states."""
    valid = (raw_ids >= 1) & (raw_ids <= 230)
    real_ids = torch.where(valid, raw_ids, torch.zeros_like(raw_ids))
    null_ids = torch.full_like(raw_ids, SPACEGROUP_MODEL_NULL_ID)
    return torch.where(spacegroup_mask > 0.5, real_ids, null_ids)


def prepare_model_conditioning(
    conditioning: dict[str, Any],
    atom_mask: Bool["b n"],
    policy: ConditioningPolicy,
    context: str,
    drop_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Build prepared model conditioning from raw collated source facts."""
    dropped = set(drop_fields or set())
    masks = policy.masks_from_presence(conditioning, context)

    membership = conditioning["membership"]
    same_group = rearrange(membership, "b i -> b i 1") == rearrange(
        membership, "b j -> b 1 j"
    )
    real_pair = rearrange(atom_mask, "b i -> b i 1") & rearrange(
        atom_mask, "b j -> b 1 j"
    )
    valid_mol_pair = same_group & real_pair
    bond_adj = conditioning["bond_adj"]
    valid_bond_pair = bond_adj & valid_mol_pair

    prepared = {
        "atomic_numbers": conditioning["atomic_numbers"],
        "template_coords": conditioning["template_coords"],
        "template_mask": masks["template"],
        "formal_charges": conditioning["formal_charges"],
        "atom_chirality": atom_chirality_model_ids(
            conditioning["atom_chirality"],
            atom_mask,
            masks["stereochemistry"],
        ),
        "membership": membership,
        "bond_type_adj": bond_type_model_ids(
            conditioning["bond_type_adj"],
            valid_mol_pair,
        ),
        "bond_stereochemistry_adj": bond_stereochemistry_model_ids(
            conditioning["bond_stereochemistry_adj"],
            valid_bond_pair,
            masks["stereochemistry"],
        ),
        "stereochemistry_mask": masks["stereochemistry"],
        "spacegroup_number": spacegroup_model_ids(
            conditioning["spacegroup_number"],
            masks["spacegroup"],
        ),
        "spacegroup_mask": masks["spacegroup"],
    }
    if "conditioning.bond_adj" not in dropped:
        prepared["bond_adj"] = bond_adj
    return prepared
