"""Export code provenance for the static evaluation UI. No inference or API calls."""
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'engine/src'))
from workbench.contracts import ResearchOutput

source = (ROOT / 'engine/src/workbench/research.py').read_text(encoding='utf-8')
tree = ast.parse(source)
prompt = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in n.targets))

def terms(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return terms(node.left) + terms(node.right)
    return [node]

variables = iter(['schema', 'question', 'scope', 'coverage_gaps', 'evidence'])
template = ''.join(n.value if isinstance(n, ast.Constant) else '{{' + next(variables) + '}}' for n in terms(prompt))
gpu_source = (ROOT / 'engine/src/workbench/gpu.py').read_text(encoding='utf-8')
gpu_tree = ast.parse(gpu_source)
evaluators = {n.name: ast.get_source_segment(gpu_source, n) for n in ast.walk(gpu_tree) if isinstance(n, ast.FunctionDef) and n.name in ('reference_validity', 'runtime_metrics')}
sessions = [
    ('Qwen/Qwen3.5-9B', '0d4137d6-9764-4e24-a501-e39c7d1e200d'),
    ('XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B', '2edc01e8-2e25-4912-81a0-708d37c70d47'),
    ('XHToken/Spark-X2.6-4B', 'f7ff327c-09bb-4f94-b9b6-b0df210e58d9'),
    ('mistralai/Ministral-3-3B-Instruct-2512', '56da2851-27f5-4194-aa0f-11494eccb295'),
    ('ibm-granite/granite-4.1-8b', 'ef1c0983-65f4-447d-847e-66bdec135b7a'),
]
experiments = []
for model_id, session_id in sessions:
    artifact = json.loads((ROOT / 'artifacts/local/models' / (model_id.replace('/', '--') + '-experiment.json')).read_text(encoding='utf-8'))
    experiments.append({'model_id': model_id, 'session_id': session_id, 'experiment_name': artifact['experiment_name']})
bundle = json.loads((ROOT / 'web/public/data/workbench.json').read_text(encoding='utf-8'))
workflow_hash = hashlib.sha256(b''.join((ROOT / 'engine/src/workbench' / n).read_bytes().replace(b'\r\n', b'\n') for n in ('research.py', 'retrieval.py', 'contracts.py'))).hexdigest()
if workflow_hash != bundle['comparison_cohort']['workflow_sha256']:
    raise RuntimeError('Source no longer matches recorded workflow (LF-normalized Colab source). Do not reconstruct historical prompts from this version.')
result = {
    'schema_version': 1,
    'provenance': 'Experiment session IDs verified in the signed-in LangSmith comparison on 2026-10-26. Prompt and evaluator snapshots exported from local source; not a native LangSmith dashboard or trace export.',
    'org_id': 'dd994a42-b5ce-44ee-97d5-c80a18801ff6',
    'dataset_id': 'dbd1ac79-81d2-4579-a99c-7d8f78879974',
    'dataset_name': 'financial-research-workbench-7a576e79b7a0',
    'experiments': experiments,
    'workflow_sha256': workflow_hash,
    'workflow_hash_basis': 'LF-normalized research.py + retrieval.py + contracts.py matches the recorded Colab workflow fingerprint.',
    'research_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'prompt_sha256': hashlib.sha256(template.encode()).hexdigest(),
    'prompt_template': template,
    'output_schema': ResearchOutput.model_json_schema(),
    'evaluators': evaluators,
    'native_spans_exported': False,
    'native_dashboards_exported': False,
    'prompt_hub_history_exported': False,
}
target = ROOT / 'web/src/data/evaluation-provenance.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('Exported five experiment identities, one prompt template and two evaluator definitions. No network or inference.')
