"""Validated molecular inputs for single-structure Packora prediction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ComponentRequest(BaseModel):
    """Describe one molecular component and its formula-unit ratio."""

    model_config = ConfigDict(extra="forbid")

    smiles: str = Field(min_length=1, max_length=2000)
    ratio: int = Field(ge=1, le=512)

    @field_validator("smiles")
    @classmethod
    def strip_smiles(cls, value: str) -> str:
        """Strip surrounding whitespace from a submitted SMILES string."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("SMILES must not be empty.")
        return normalized


class PredictionRequest(BaseModel):
    """Describe one Packora structure-prediction request."""

    model_config = ConfigDict(extra="forbid")

    model: str = "packora-m"
    components: list[ComponentRequest] = Field(min_length=1, max_length=16)
    z: int | None = Field(default=None, ge=1, le=512)
