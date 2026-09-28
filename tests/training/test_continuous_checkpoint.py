"""The continuous entry loads the selected adapter and changes actual outputs."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sts1_llm_policy.workflows.continuous_run import (
    run,
)
from sts1_llm_policy.train.lora import LoraSpec, inject_lora, save_lora_checkpoint, inspect_lora_checkpoint


class ContinuousCheckpointTests(unittest.TestCase):
    def test_changing_checkpoint_changes_loaded_model_behavior(self):
        from sts1_llm_policy.policy import transformers_backend
        base = nn.Sequential(nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            base[0].weight.zero_()
        runtime = SimpleNamespace(
            model_id="toy", revision="revision", dtype="float32", device="cpu",
            attention_implementation="eager", do_sample=False, num_beams=1,
            max_new_tokens=8, use_cache=True, repetition_penalty=1.0,
            eos_token_ids=(1,), generation_pad_token_id=0,
        )
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
        observed = []
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime.snapshot_path = root / "base-snapshot"
            arms = [{"config": {"id": "base", "policy": "llm"}, "runtime": runtime}]
            for name, value in (("sft", 1.0), ("dpo", 2.0)):
                model = deepcopy(base)
                spec = LoraSpec(("0",), rank=1, alpha=1.0, dropout=0.0)
                replaced = inject_lora(model, spec)
                with torch.no_grad():
                    model[0].lora_a.weight.fill_(1.0)
                    model[0].lora_b.weight.fill_(value)
                checkpoint = root / name
                save_lora_checkpoint(model, checkpoint, spec=spec, replaced_modules=replaced,
                                     base_model_id="toy", base_revision="revision")
                metadata = inspect_lora_checkpoint(checkpoint, expected_base_model_id="toy", expected_base_revision="revision")
                arms.append({"config": {"id": name, "policy": "llm", "checkpoint": name},
                             "runtime": runtime, "checkpoint": checkpoint, "checkpoint_metadata": metadata})
            route = {"route_index": 0, "combat_seed_group_index": 0, "steps": []}
            validated = {
                "config": {"run_id": "toy", "route_count": 1, "combat_seed_groups_per_route": 1,
                           "observation_version": "observation_v7", "interaction_contract": "combat_card_selection_v1"},
                "routes": [route], "arms": arms, "output": root / "outputs", "identity": {},
                "paths": {"picker_database": root / "unused.sqlite3"},
                "execution": SimpleNamespace(describe=lambda: {}),
            }

            def consume_model(validated, arm, route, *, backend, **kwargs):
                observed.append(backend.model(torch.ones(1, 2)).detach().tolist())
                return {"route": route, "status": "completed", "death_combat_index": None,
                        "combats_started": 0, "combats": []}

            with (
                patch("sts1_llm_policy.workflows.continuous_run.prepare_run", return_value=validated),
                patch("sts1_llm_policy.workflows.continuous_run.load_completed_route", return_value=None),
                patch("sts1_llm_policy.workflows.continuous_run._run_route", side_effect=consume_model),
                patch("sts1_llm_policy.data.card_pick_metrics.SqliteCardRateBackend"),
                patch("sts1_llm_policy.data.card_pick_metrics.MassNormalizedCardRewardPicker"),
                patch("sts1_llm_policy.eval.counterfactual_reward_picker_v2.StrategicRateTable"),
                patch.object(transformers_backend, "configure_local_huggingface_environment"),
                patch.object(transformers_backend, "verify_runtime_dependencies"),
                patch.object(transformers_backend, "verify_torch_hardware"),
                patch.object(transformers_backend, "verify_model_weights"),
                patch.object(transformers_backend, "load_verified_local_tokenizer", return_value=tokenizer),
                patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **kw: deepcopy(base)),
            ):
                run(root / "unused.json", project_root=root)
        self.assertEqual(observed, [[[0.0, 0.0]], [[2.0, 2.0]], [[4.0, 4.0]]])
