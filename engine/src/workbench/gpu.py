"""Downloaded-weight inference only. No private inference provider is supported."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path

from .corpus import ROOT, LOCAL, load_json, write_json


class LocalGenerator:
    def __init__(self, spec, revision=None):
        import torch
        import transformers
        from huggingface_hub import HfApi
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required for the five-model benchmark. Select a Colab GPU runtime.')
        self.spec = spec
        self.revision = revision or HfApi().model_info(spec['id']).sha
        info = HfApi().model_info(spec['id'], revision=self.revision)
        license_id = (info.card_data or {}).get('license')
        if license_id not in ('apache-2.0', 'mit') or license_id != spec['license']:
            raise ValueError(f"Unexpected model license for {spec['id']}: {license_id!r}")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_gb < (spec['recommended_gpu_gb'] - 2):
            raise RuntimeError(f"{spec['name']} targets {spec['recommended_gpu_gb']} GB GPU memory in this profile; detected {total_gb:.1f} GB. Use a larger runtime or run Spark first.")
        common = dict(revision=self.revision, device_map={'': 0}, trust_remote_code=spec['requires_remote_code'], attn_implementation='eager', dtype=torch.float16)
        tokenizer_cls = transformers.AutoTokenizer
        loader = transformers.AutoModelForCausalLM
        if spec['loader'] == 'multimodal':
            loader = getattr(transformers, 'AutoModelForImageTextToText', None) or getattr(transformers, 'AutoModelForMultimodalLM')
            tokenizer_cls = transformers.AutoProcessor
        if spec['loader'] == 'mistral':
            loader = transformers.Mistral3ForConditionalGeneration
            tokenizer_cls = transformers.MistralCommonBackend
        if spec['quantization'] == 'bnb4':
            common['quantization_config'] = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
        elif spec['quantization'] == 'fp16_dequantize_fp8':
            common['quantization_config'] = transformers.FineGrainedFP8Config(dequantize=True)
            common['dtype'] = torch.float16
        elif spec['quantization'] == 'native_mxfp4':
            # A100 fallback avoids assuming native FP4 kernels on older GPUs.
            capability = torch.cuda.get_device_capability(0)
            if capability[0] < 9:
                common['quantization_config'] = transformers.Mxfp4Config(dequantize=True)
                common['dtype'] = torch.bfloat16
                if total_gb < 60:
                    raise RuntimeError('gpt-oss MXFP4 dequantization targets an 80GB GPU on this runtime. Native-compatible hardware can use its compact checkpoint.')
        self.tokenizer = tokenizer_cls.from_pretrained(spec['id'], revision=self.revision, trust_remote_code=spec['requires_remote_code'])
        self.model = loader.from_pretrained(spec['id'], **common)
        if spec['quantization'] == 'fp16_dequantize_fp8':
            # FP8 dequantization can materialize BF16 submodules despite dtype.
            # Normalize the entire dequantized model for older T4 hardware.
            self.model = self.model.to(dtype=torch.float16)
        self.metadata = {'model_id': spec['id'], 'revision': self.revision, 'license': license_id, 'quantization': spec['quantization'], 'gpu': torch.cuda.get_device_name(0), 'gpu_memory_gb': round(total_gb, 2), 'torch': torch.__version__, 'transformers': transformers.__version__, 'python': platform.python_version(), 'max_new_tokens': 768, 'context_policy': 'same bounded evidence; text only', 'inference': 'self_hosted_downloaded_weights'}
        self.metadata['workflow_sha256'] = hashlib.sha256(b''.join((ROOT / 'engine/src/workbench' / name).read_bytes() for name in ('research.py', 'retrieval.py', 'contracts.py'))).hexdigest()
        self.metadata['inference_adapter_sha256'] = hashlib.sha256(b''.join((ROOT / 'engine/src/workbench' / name).read_bytes() for name in ('local_chat_model.py', 'gpu.py'))).hexdigest()
        self.measurements = []

    def __call__(self, prompt):
        from .local_chat_model import LocalWeightsChatModel
        if not hasattr(self, 'chat_model'):
            self.chat_model = LocalWeightsChatModel(backend=self, model_id=self.spec['id'], revision=self.revision)
        return self.chat_model.invoke(prompt).content

    def _generate_text(self, prompt):
        import torch
        messages = [{'role': 'user', 'content': prompt}]
        if self.spec['loader'] in ('mistral', 'multimodal'):
            messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
        kwargs = dict(tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True)
        if self.spec['family'] in ('Qwen', 'MiMo', 'Nanbeige', 'Spark'):
            kwargs['enable_thinking'] = False
        inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
        if hasattr(inputs, 'shape'):
            inputs = {'input_ids': inputs}
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        input_tokens = inputs['input_ids'].shape[-1]
        if input_tokens > 8000:
            raise RuntimeError('Evidence exceeds the common 8,000-token input cap; revise the task context rather than silently truncating it.')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=768)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        ids = outputs[0][input_tokens:]
        self.measurements.append({'input_tokens': input_tokens, 'output_tokens': len(ids), 'generation_seconds': round(duration, 3), 'tokens_per_second': round(len(ids) / duration, 2), 'peak_gpu_memory_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 3)})
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def close(self):
        import torch
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def run_benchmark(model_id, limit=None, langsmith=False, revision=None, case_set='showcase'):
    from .retrieval import CorpusIndex
    from .research import run_case
    spec = next(model for model in load_json(ROOT / 'models.json')['models'] if model['id'] == model_id)
    corpus = load_json(LOCAL / 'corpus.json')
    index = CorpusIndex(corpus['blocks'], corpus['documents'])
    if case_set not in ('showcase', 'smoke'):
        raise ValueError('Unknown benchmark case set')
    cases = load_json(ROOT / f'evals/{case_set}.json')[:limit]
    generator = LocalGenerator(spec, revision)
    generator.metadata['corpus_snapshot_sha256'] = hashlib.sha256(json.dumps([(b['id'], b.get('filing_date')) for b in corpus['blocks']], sort_keys=True).encode()).hexdigest()
    generator.metadata['corpus_block_count'] = len(corpus['blocks'])
    model_slug = model_id.replace('/', '--')
    write_json(LOCAL / 'models' / f'{model_slug}.json', generator.metadata)
    results, failures = [], []

    def target(inputs):
        previous_measurements = len(generator.measurements)
        try:
            run = run_case(index, inputs['case'], generate=generator, model=model_id, baseline=inputs.get('baseline', False))
        except Exception as error:
            failure = {'case_id': inputs['case']['id'], 'baseline': inputs.get('baseline', False), 'error': type(error).__name__, 'message': str(error)[:400]}
            if len(generator.measurements) > previous_measurements:
                failure['generation'] = generator.measurements[-1]
                failure['output_limit_reached'] = failure['generation']['output_tokens'] >= generator.metadata['max_new_tokens']
            failures.append(failure)
            raise
        run['hardware'] = generator.metadata
        run['generation'] = generator.measurements[-1]
        write_json(LOCAL / 'runs' / f"{run['id']}.json", run)
        results.append(run)
        print(f"Completed {model_id}: {run['case_id']} ({'baseline' if run['baseline'] else 'workflow'})", flush=True)
        return {'summary': run['output']['summary'], 'verification': run['verification'], 'run_id': run['id'], 'generation': run['generation']}

    try:
        if langsmith:
            if not os.environ.get('LANGSMITH_API_KEY'):
                raise RuntimeError('LANGSMITH_API_KEY is required for experiment recording, not for inference. Set it in Colab Secrets or run local-artifact mode.')
            from langsmith import Client
            from langsmith.evaluation import evaluate
            client = Client()
            fingerprint = hashlib.sha256(json.dumps({'cases': cases, 'corpus_snapshot_sha256': generator.metadata['corpus_snapshot_sha256']}, sort_keys=True).encode()).hexdigest()[:12]
            dataset_name = 'financial-research-workbench-' + fingerprint
            datasets = list(client.list_datasets(dataset_name=dataset_name))
            if datasets:
                dataset = datasets[0]
            else:
                dataset = client.create_dataset(dataset_name=dataset_name, description='Public financial-report research tasks. No gold quality label is inferred.')
                client.create_examples(inputs=[{'case': case, 'baseline': baseline} for case in cases for baseline in (True, False)], dataset_id=dataset.id)
            def reference_validity(run, example):
                value = (run.outputs or {}).get('verification', {}).get('citation_validity')
                return {'key': 'citation_reference_validity', 'score': value, 'comment': 'ID membership only; not entailment.'}
            def runtime_metrics(run, example):
                generation = (run.outputs or {}).get('generation', {})
                return {'results': [{'key': key, 'score': generation[key]} for key in ('tokens_per_second', 'peak_gpu_memory_gb', 'generation_seconds', 'input_tokens', 'output_tokens') if key in generation]}
            experiment = evaluate(target, data=dataset_name, evaluators=[reference_validity, runtime_metrics], experiment_prefix=model_slug, metadata=generator.metadata, max_concurrency=1)
            write_json(LOCAL / 'models' / f'{model_slug}-experiment.json', {'experiment_name': experiment.experiment_name, 'model_id': model_id, 'metadata': generator.metadata})
        else:
            for case in cases:
                for baseline in (True, False):
                    try:
                        target({'case': case, 'baseline': baseline})
                    except Exception as error:
                        print(f"Failed {case['id']}: {type(error).__name__}", flush=True)
    finally:
        generator.close()
        write_json(LOCAL / 'models' / f'{model_slug}-status.json', {'model_id': model_id, 'completed': len(results), 'failures': failures, 'metadata': generator.metadata})
    return results
