"""Small synthetic runtime contracts, independent of experiment assets and status."""
from copy import deepcopy
from dataclasses import replace

from sts1_llm_policy.model_runtime import load_base_model_runtime_config
from sts1_llm_policy.train.configured_adapters import PreparedTrainingData, SftAdapter
from sts1_llm_policy.train.configured_training import ConfiguredTrainingRun, _portable_config
from sts1_llm_policy.configuration import load_config_document

from tests.support import write_json

RUNTIME = {'schema_version': 'base_model_runtime_v2',
 'runtime_id': 'fixture',
 'experiment_protocol_id': 'fixture_protocol',
 'model': {'id': 'fixture/model',
           'revision': '1111111111111111111111111111111111111111',
           'snapshot_path': 'assets/1111111111111111111111111111111111111111',
           'architecture': 'Qwen2ForCausalLM',
           'config_sha256': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
           'generation_config_sha256': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
           'weight_assets_sha256': {'model.safetensors': 'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc'}},
 'tokenizer': {'loader': 'AutoTokenizer.from_pretrained',
               'local_files_only': True,
               'trust_remote_code': False,
               'use_fast': True,
               'implementation_class': 'Qwen2TokenizerFast',
               'vocab_size': 32,
               'total_size': 35,
               'model_max_length': 128,
               'chat_template': {'source': 'tokenizer.chat_template',
                                 'encoding': 'utf-8',
                                 'byte_length': 4,
                                 'sha256': 'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'},
               'special_tokens': {'bos_token': None,
                                  'bos_token_id': None,
                                  'eos_token': '<|im_end|>',
                                  'eos_token_id': 33,
                                  'pad_token': '<|endoftext|>',
                                  'pad_token_id': 32,
                                  'unk_token': None,
                                  'unk_token_id': None},
               'asset_sha256': {'tokenizer.json': 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'}},
 'prompt_rendering': {'message_roles': ['system', 'user'],
                      'tools': None,
                      'add_generation_prompt': True,
                      'tokenize': True,
                      'add_special_tokens': False},
 'model_loading': {'backend': 'transformers',
                   'loader': 'AutoModelForCausalLM.from_pretrained',
                   'local_files_only': True,
                   'trust_remote_code': False,
                   'device': 'cuda:0',
                   'dtype': 'bfloat16',
                   'device_map': None,
                   'quantization': None,
                   'attention_implementation': 'sdpa',
                   'eval_mode': True},
 'generation': {'batch_size': 1,
                'strategy': 'greedy',
                'temperature': 0.0,
                'do_sample': False,
                'num_beams': 1,
                'repetition_penalty': 1.0,
                'transformers_sampling_parameters': None,
                'max_new_tokens': 8,
                'use_cache': True,
                'eos_token_ids': [33, 32],
                'pad_token_id': 32,
                'constrained_decoding': False,
                'decode_skip_special_tokens': True,
                'decode_clean_up_tokenization_spaces': False,
                'independent_decisions': True,
                'chat_history': False}}

RECIPE = {'schema_version': 'training_recipe_v1',
 'recipe_id': 'fixture_recipe',
 'algorithm': 'sft',
 'lora': {'target_module_suffixes': ['q_proj', 'v_proj'], 'rank': 8, 'alpha': 16.0, 'dropout': 0.0},
 'tokenization': {'max_sequence_tokens': 3072,
                  'truncation': False,
                  'response_only': True,
                  'supervise_assistant_terminator': True},
 'optimizer': {'name': 'adamw', 'learning_rate': 0.0001, 'weight_decay': 0.0, 'max_grad_norm': 1.0},
 'training': {'epochs': 1,
              'micro_batch_units': 1,
              'gradient_accumulation_units': 2,
              'gradient_checkpointing': True,
              'resume_checkpoint_interval_steps': 1},
 'smoke': {'units': 2, 'optimizer_steps': 1, 'selection': 'longest_then_hash_spread'}}

EXECUTION = {'schema_version': 'training_execution_v1',
 'dependencies': {'python': 'test-version',
                  'torch': 'test-version',
                  'transformers': 'test-version',
                  'tokenizers': 'test-version',
                  'huggingface-hub': 'test-version',
                  'safetensors': 'test-version'},
 'hardware': {'platform': 'linux',
              'architecture': 'x86_64',
              'compute_capabilities': ['12.0'],
              'cuda_runtime': '13.2',
              'bf16_supported': True}}

def runtime_document(*, legacy=False):
    value = deepcopy(RUNTIME)
    if legacy:
        value.update(schema_version="base_model_runtime_v1", dependencies=deepcopy(EXECUTION["dependencies"]),
                     validated_hardware={"device_name": "fixture GPU", "compute_capability": "12.0",
                                         "cuda_runtime": "13.2", "bf16_supported": True},
                     validation_status={"tokenizer_local_only_verified": True, "model_weights_loaded": False,
                                        "generation_smoke_completed": False})
    return value


CONFIG = "run.json"
RUNNER = "sts1_llm_policy.train.configured_runner."


def fixture(root):
    root = root.resolve()
    raw = {"schema_version": "training_run_v2", "run_id": "fixture", "allowed_modes": ["preflight", "backward", "run"],
           "model_runtime": "runtime.json", "training_recipe": "recipe.json", "execution_profile": "execution.json",
           "dataset_manifest": "dataset.json", "seed": 7}
    config = _portable_config(raw)
    documents = {}
    for key, value in (("model_runtime", runtime_document()), ("training_recipe", RECIPE), ("execution_profile", EXECUTION)):
        write_json(root / raw[key], value)
        documents[key] = load_config_document(root, raw[key])
    write_json(root / CONFIG, raw)
    write_json(root / "legacy.json", runtime_document(legacy=True))
    dataset = {"splits": {"train": {"sha256": "a" * 64, "records": 1}}, "dataset_id": "test"}
    write_json(root / "dataset.json", dataset)
    runtime = load_base_model_runtime_config(
        root / config["model_runtime"], project_root=root, execution_profile=documents["execution_profile"].value,
    )
    return ConfiguredTrainingRun(
        root, replace(load_config_document(root, CONFIG), value=config), documents["training_recipe"], documents["model_runtime"],
        load_config_document(root, "dataset.json"), runtime, SftAdapter(),
        PreparedTrainingData("sft", ({"record_id": "long", "sequence_tokens_max": 5, "tokenized": {"input_ids": [1,2,3,4,5],
                                 "attention_mask": [1]*5, "labels": [-100,-100,-100,4,5], "prompt_token_count": 3}},),
                             {"truncated": False, "sequence_tokens_max": 5}),
        root / config["backward"]["output_dir"], root / config["backward"]["report"],
        None, "backward", {"implementation.py": "b" * 64}, documents["execution_profile"], None,
    )
