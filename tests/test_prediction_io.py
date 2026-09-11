from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from src.models.callbacks.generation_writer import PredictionBundleWriter
from src.models.flow_module import MaterialFlowModule
from src.prediction.artifacts import collect_raw_payloads
from src.prediction.io import (
    PredictionBundle,
    build_prediction_bundle,
    build_sample_index,
    expand_prediction_rows,
    load_prediction_bundle,
    metadata_rows_to_columns,
    pad_and_cat,
    reorder_metadata_columns,
    reorder_tensor_dict,
    save_prediction_bundle,
)


class _TrainerStub:
    """Minimal trainer object for prediction writer tests."""

    global_rank = 0


class _ChunkWriterStub:
    """Record chunks streamed from predict_step."""

    def __init__(self) -> None:
        """Initialize the chunk writer recorder."""
        self.chunks: list[dict[str, object]] = []

    def write_prediction_chunk(
        self,
        *,
        trainer: object,
        prediction: dict[str, object],
        batch_indices: object,
        batch: dict[str, object],
        batch_idx: int,
    ) -> None:
        """Record one streamed prediction chunk."""
        self.chunks.append(
            {
                "trainer": trainer,
                "prediction": prediction,
                "batch_indices": batch_indices,
                "batch": batch,
                "batch_idx": int(batch_idx),
            }
        )


class _RecordingFlowMatching:
    """Record sample calls and return minimal generated tensors."""

    def __init__(self) -> None:
        """Initialize the call recorder."""
        self.calls: list[dict[str, object]] = []

    def sample(
        self,
        *,
        num_atoms: torch.Tensor,
        multiplicity: int,
        num_steps: int,
        method: str,
        sde_noise_scale: float,
        conditioning: dict[str, torch.Tensor],
        conditioning_context: str,
        steering_args: dict | None,
        sampler_args: dict | None,
        spacegroup_guidance_weight: float | None,
        time_epsilon: float,
        use_inference_cache: bool,
    ) -> dict[str, torch.Tensor]:
        """Return one generated payload with batch rows expanded by multiplicity."""
        self.calls.append(
            {
                "multiplicity": int(multiplicity),
                "num_steps": int(num_steps),
                "method": str(method),
                "conditioning_context": str(conditioning_context),
                "time_epsilon": float(time_epsilon),
                "steering_args": steering_args,
                "sampler_args": sampler_args,
                "spacegroup_guidance_weight": spacegroup_guidance_weight,
                "use_inference_cache": bool(use_inference_cache),
            }
        )
        batch_size = int(num_atoms.shape[0])
        n_atoms = int(conditioning["atomic_numbers"].shape[1])
        return _make_tensor_payload(
            batch=batch_size * int(multiplicity), n_atoms=n_atoms
        )


def _make_tensor_payload(batch: int, n_atoms: int) -> dict[str, torch.Tensor]:
    """Build a minimal prediction-like tensor payload."""
    return {
        "cart_coords": torch.zeros(batch, n_atoms, 3, dtype=torch.float32),
        "lattice": torch.zeros(batch, 6, dtype=torch.float32),
        "atomic_numbers": torch.ones(batch, n_atoms, dtype=torch.long),
        "atom_mask": torch.ones(batch, n_atoms, dtype=torch.bool),
    }


def test_pad_and_cat_pads_variable_atom_count() -> None:
    """Pad and concatenate variable-N batches on batch axis."""
    first = _make_tensor_payload(batch=2, n_atoms=3)
    second = _make_tensor_payload(batch=1, n_atoms=5)
    merged = pad_and_cat([first, second])

    assert merged["cart_coords"].shape == (3, 5, 3)
    assert merged["atomic_numbers"].shape == (3, 5)


def test_expand_prediction_rows_repeats_source_data_in_source_major_order() -> None:
    """Expand one raw batch payload to generated rows with sample indices."""
    pred = _make_tensor_payload(batch=6, n_atoms=2)
    ref = _make_tensor_payload(batch=2, n_atoms=2)
    dataset_index = torch.tensor([10, 20], dtype=torch.long)
    metadata_rows = [{"material_id": "src0"}, {"material_id": "src1"}]

    (
        expanded_pred,
        expanded_ref,
        expanded_dataset_index,
        sample_index,
        expanded_metadata,
    ) = expand_prediction_rows(
        pred=pred,
        ref=ref,
        dataset_index=dataset_index,
        metadata_rows=metadata_rows,
    )

    assert expanded_pred is pred
    assert expanded_ref["cart_coords"].shape[0] == 6
    assert expanded_dataset_index.tolist() == [10, 10, 10, 20, 20, 20]
    assert sample_index.tolist() == [0, 1, 2, 0, 1, 2]
    assert expanded_metadata == [
        {"material_id": "src0"},
        {"material_id": "src0"},
        {"material_id": "src0"},
        {"material_id": "src1"},
        {"material_id": "src1"},
        {"material_id": "src1"},
    ]


def test_build_sample_index_uses_source_major_order() -> None:
    """Construct sample indices in source-major row order."""
    sample_index = build_sample_index(batch_size=2, multiplicity=3)

    assert sample_index.tolist() == [0, 1, 2, 0, 1, 2]


def test_predict_step_streams_sample_chunks_without_merging() -> None:
    """Return one prediction payload per sample chunk."""
    module = MaterialFlowModule.__new__(MaterialFlowModule)
    flow_matching = _RecordingFlowMatching()
    module.flow_matching = flow_matching
    module.conditioning_contexts = {"predict": "predict"}
    module._gen_samples_per_datapoint = 5
    module._gen_sample_chunk_size = 2
    module._gen_num_steps = 7
    module._gen_method = "sde"
    module._gen_sde_noise_scale = 0.5
    module._gen_sampler_args = {"noise_scale": 0.5}
    module._gen_spacegroup_guidance_weight = 2.0
    module._gen_steering_args = None
    module._gen_time_epsilon = 0.002
    batch = {
        "num_atoms": torch.tensor([2, 2], dtype=torch.long),
        "conditioning": {
            "atomic_numbers": torch.ones(2, 3, dtype=torch.long),
        },
    }

    chunks = module.predict_step(batch=batch, batch_idx=0)

    assert [chunk["sample_offset"] for chunk in chunks] == [0, 2, 4]
    assert [chunk["chunk_size"] for chunk in chunks] == [2, 2, 1]
    assert [call["multiplicity"] for call in flow_matching.calls] == [2, 2, 1]
    assert [chunk["pred"]["cart_coords"].shape[0] for chunk in chunks] == [4, 4, 2]
    assert all(call["time_epsilon"] == 0.002 for call in flow_matching.calls)
    assert all(
        call["spacegroup_guidance_weight"] == 2.0 for call in flow_matching.calls
    )


def test_predict_step_writes_chunks_immediately_when_writer_is_attached() -> None:
    """Stream generated chunks through the attached prediction writer."""
    module = MaterialFlowModule.__new__(MaterialFlowModule)
    flow_matching = _RecordingFlowMatching()
    writer = _ChunkWriterStub()
    trainer = _TrainerStub()
    module.flow_matching = flow_matching
    module.conditioning_contexts = {"predict": "predict"}
    module.__dict__["_trainer"] = trainer
    module._gen_prediction_writer = writer
    module._gen_samples_per_datapoint = 3
    module._gen_sample_chunk_size = 1
    batch = {
        "num_atoms": torch.tensor([2], dtype=torch.long),
        "conditioning": {
            "atomic_numbers": torch.ones(1, 3, dtype=torch.long),
        },
    }

    result = module.predict_step(batch=batch, batch_idx=4)

    assert result is None
    assert [call["multiplicity"] for call in flow_matching.calls] == [1, 1, 1]
    assert [item["prediction"]["sample_offset"] for item in writer.chunks] == [0, 1, 2]
    assert all(item["batch_idx"] == 4 for item in writer.chunks)
    assert all(item["trainer"] is trainer for item in writer.chunks)


def test_prediction_writer_stores_sample_chunk_offsets(tmp_path: Path) -> None:
    """Write and collect raw prediction chunks with preserved sample offsets."""
    writer = PredictionBundleWriter(output_dir=tmp_path)
    batch = {
        "cart_coords": torch.zeros(2, 3, 3),
        "lattice": torch.zeros(2, 6),
        "cell": torch.zeros(2, 3, 3),
        "conditioning": {"atomic_numbers": torch.ones(2, 3, dtype=torch.long)},
        "atom_mask": torch.ones(2, 3, dtype=torch.bool),
        "num_atoms": torch.tensor([3, 3], dtype=torch.long),
        "dataset_index": torch.tensor([10, 20], dtype=torch.long),
        "metadata": [{"material_id": "a"}, {"material_id": "b"}],
    }
    prediction = [
        {
            "pred": _make_tensor_payload(batch=4, n_atoms=3),
            "sample_offset": 0,
            "chunk_size": 2,
        },
        {
            "pred": _make_tensor_payload(batch=2, n_atoms=3),
            "sample_offset": 2,
            "chunk_size": 1,
        },
    ]

    writer.write_on_batch_end(
        trainer=_TrainerStub(),
        pl_module=None,
        prediction=prediction,
        batch_indices=None,
        batch=batch,
        batch_idx=3,
        dataloader_idx=0,
    )
    (
        pred_batches,
        ref_batches,
        dataset_index_batches,
        sample_index_batches,
        metadata_rows,
    ) = collect_raw_payloads(tmp_path)

    assert [pred["cart_coords"].shape[0] for pred in pred_batches] == [4, 2]
    assert [ref["cart_coords"].shape[0] for ref in ref_batches] == [4, 2]
    assert [batch.tolist() for batch in dataset_index_batches] == [
        [10, 10, 20, 20],
        [10, 20],
    ]
    assert [batch.tolist() for batch in sample_index_batches] == [
        [0, 1, 0, 1],
        [2, 2],
    ]
    assert metadata_rows == [
        {"material_id": "a"},
        {"material_id": "a"},
        {"material_id": "b"},
        {"material_id": "b"},
        {"material_id": "a"},
        {"material_id": "b"},
    ]


def test_bundle_save_and_load_round_trip(tmp_path: Path) -> None:
    """Serialize and load prediction bundle without losing alignment fields."""
    pred = _make_tensor_payload(batch=3, n_atoms=4)
    ref = _make_tensor_payload(batch=3, n_atoms=4)
    bundle = PredictionBundle(
        pred=pred,
        ref=ref,
        dataset_indices=np.array([0, 1, 2], dtype=np.int64),
        sample_indices=np.array([0, 1, 0], dtype=np.int64),
        metadata={
            "material_id": ["a", "b", "c"],
            "canonical_smiles": ["C", "CC", "CCC"],
            "csd_refcode": ["AAA", "BBB", "CCC"],
            "genarris_step": ["relax", "relax", "press"],
            "xtal_id": ["xtal0", "xtal1", "xtal2"],
        },
    )

    path = tmp_path / "predictions.pt"
    save_prediction_bundle(path, bundle)
    loaded = load_prediction_bundle(path)

    assert loaded.dataset_indices.tolist() == [0, 1, 2]
    assert loaded.sample_indices.tolist() == [0, 1, 0]
    assert loaded.metadata["material_id"] == ["a", "b", "c"]
    assert loaded.metadata["csd_refcode"] == ["AAA", "BBB", "CCC"]
    assert "split" not in loaded.metadata


def test_metadata_helpers_follow_shared_row_order() -> None:
    """Reorder metadata columns with one common row permutation."""
    rows = [
        {"material_id": "row0", "value": 10},
        {"material_id": "row1", "value": 20},
        {"material_id": "row2", "value": 30},
    ]
    columns = metadata_rows_to_columns(rows)
    order = torch.tensor([2, 0, 1], dtype=torch.long)
    reordered = reorder_metadata_columns(columns, order)

    assert reordered["material_id"] == ["row2", "row0", "row1"]
    assert reordered["value"] == [30, 10, 20]


def test_reorder_tensor_dict_uses_batch_axis() -> None:
    """Reorder every tensor field along the first axis."""
    payload = {
        "cart_coords": rearrange(
            torch.arange(3 * 2 * 3, dtype=torch.float32),
            "(b n c) -> b n c",
            b=3,
            n=2,
            c=3,
        ),
        "lattice": rearrange(
            torch.arange(3 * 6, dtype=torch.float32),
            "(b d) -> b d",
            b=3,
            d=6,
        ),
    }
    order = torch.tensor([2, 0, 1], dtype=torch.long)
    reordered = reorder_tensor_dict(payload, order)

    assert torch.equal(reordered["lattice"][0], payload["lattice"][2])


def test_build_prediction_bundle_deduplicates_sorted_dataset_indices() -> None:
    """Drop duplicate composite row ids introduced by distributed sampler padding."""
    pred_batches = [_make_tensor_payload(batch=4, n_atoms=2)]
    ref_batches = [_make_tensor_payload(batch=4, n_atoms=2)]
    dataset_index_batches = [torch.tensor([1, 0, 1, 1], dtype=torch.long)]
    sample_index_batches = [torch.tensor([0, 0, 1, 1], dtype=torch.long)]
    metadata_rows = [
        {
            "material_id": "row_dataset1_sample0",
            "csd_refcode": "BBB",
            "genarris_step": "relax",
            "xtal_id": "xtal1_s0",
        },
        {
            "material_id": "row_dataset0_sample0",
            "csd_refcode": "AAA",
            "genarris_step": "relax",
            "xtal_id": "xtal0_s0",
        },
        {
            "material_id": "row_dataset1_sample1",
            "csd_refcode": "BBB",
            "genarris_step": "relax",
            "xtal_id": "xtal1_s1",
        },
        {
            "material_id": "row_dataset1_sample1_dup",
            "csd_refcode": "BBB",
            "genarris_step": "relax",
            "xtal_id": "xtal1_s1",
        },
    ]

    bundle = build_prediction_bundle(
        pred_batches=pred_batches,
        ref_batches=ref_batches,
        dataset_index_batches=dataset_index_batches,
        sample_index_batches=sample_index_batches,
        metadata_rows=metadata_rows,
    )

    assert bundle.dataset_indices.tolist() == [0, 1, 1]
    assert bundle.sample_indices.tolist() == [0, 0, 1]
    assert bundle.pred["cart_coords"].shape[0] == 3
    assert bundle.metadata["material_id"] == [
        "row_dataset0_sample0",
        "row_dataset1_sample0",
        "row_dataset1_sample1",
    ]
    assert bundle.metadata["csd_refcode"] == ["AAA", "BBB", "BBB"]


def test_load_prediction_bundle_defaults_missing_sample_indices(tmp_path: Path) -> None:
    """Load old bundle payloads by defaulting missing sample indices to zero."""
    path = tmp_path / "predictions.pt"
    torch.save(
        {
            "pred": _make_tensor_payload(batch=2, n_atoms=3),
            "ref": _make_tensor_payload(batch=2, n_atoms=3),
            "dataset_indices": np.array([4, 5], dtype=np.int64),
            "metadata": {"material_id": ["row4", "row5"]},
        },
        path,
    )

    bundle = load_prediction_bundle(path)

    assert bundle.dataset_indices.tolist() == [4, 5]
    assert bundle.sample_indices.tolist() == [0, 0]
