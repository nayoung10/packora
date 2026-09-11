"""CSD refcode-family helpers for preprocessing."""

from src.data.types import Material


def refcode_family(material: Material) -> str:
    """Return the uppercase CSD refcode family for a material."""
    refcode = (material.info or {}).get("csd_refcode")
    if refcode is None or str(refcode) == "":
        raise ValueError("Material is missing info['csd_refcode'].")
    return str(refcode)[:6].upper()
