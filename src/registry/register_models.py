"""Package all three selected trial checkpoints as MLflow pyfunc registered models.

Give @production aliases to the two models frozen by VALIDATION scores before test.
This only registers/aliases models; it does not yet deploy a FastAPI process.
"""
import json
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import Schema, ColSpec

from src.serving.mlflow_pyfunc import FaceOpsPythonModel
from src.storage.checkpoint_fetch import download_trial_checkpoint

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports' / 'private_evaluation'
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI','http://127.0.0.1:5050')
REGISTRY_MANIFEST = OUTPUT / 'registry_versions.json'


def main():
    selection_file = OUTPUT / 'selection.json'
    eval_file = OUTPUT / 'heldout_results.json'
    if not selection_file.exists() or not eval_file.exists():
        raise RuntimeError('Selection and held-out evaluation must finish before registration')
    if REGISTRY_MANIFEST.exists():
        print('Already registered; not creating duplicate versions:', REGISTRY_MANIFEST)
        print(REGISTRY_MANIFEST.read_text())
        return
    selection = json.loads(selection_file.read_text())
    evaluated = json.loads(eval_file.read_text())
    assert selection['dataset_version'] == evaluated['dataset_version'] == 2
    approved = set(selection['selected_for_registry_production_alias'])
    assert len(approved) == 2
    assert set(evaluated['models']) == {rec['architecture'] for rec in selection['all_models']}
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment('FaceOps_Registered_Models')
    client = MlflowClient(tracking_uri=TRACKING_URI)
    manifest = {'dataset_version': 2, 'production_aliases': sorted(approved), 'models': {}}
    signature = ModelSignature(
        inputs=Schema([ColSpec('string', name='image_b64')]),
        outputs=Schema([ColSpec('string', name='class_name'), ColSpec('double', name='confidence')] +
                       [ColSpec('double', name='prob_'+name) for name in ('pawara','uchitha','dhamith','unknown')]),
    )
    for record in selection['all_models']:
        name = record['architecture']
        registry_name = f'faceops_{name}'
        source_id = record['source_run_id']
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = str(download_trial_checkpoint(source_id, temp))
            with mlflow.start_run(run_name=f'register_{name}') as run:
                mlflow.log_params({
                    'architecture':name, 'source_run_id':source_id,
                    'dataset_version':2, 'selected_from_validation':name in approved,
                })
                model_info = mlflow.pyfunc.log_model(
                    name='face_classifier',
                    python_model=FaceOpsPythonModel(),
                    artifacts={'checkpoint': checkpoint},
                    code_paths=[str(ROOT / 'src')],
                    signature=signature,
                    registered_model_name=registry_name,
                    pip_requirements=['mlflow-skinny==3.17.1', 'torch', 'torchvision', 'Pillow', 'numpy', 'pandas'],
                )
                version_number = model_info.registered_model_version
                if version_number is None:
                    # Older servers may not populate this field. Match the just-created run.
                    versions = client.search_model_versions(f"name='{registry_name}'")
                    belonging = [v for v in versions if v.run_id == run.info.run_id]
                    if not belonging:
                        raise RuntimeError(f'Cannot identify registered version for {registry_name}')
                    version_number = int(max(belonging, key=lambda v:int(v.version)).version)
                if name in approved:
                    client.set_registered_model_alias(registry_name, 'production', str(version_number))
                manifest['models'][name] = {
                    'registry_name': registry_name,
                    'version': str(version_number),
                    'registry_run_id': run.info.run_id,
                    'source_run_id': source_id,
                    'model_uri': model_info.model_uri,
                    'production_alias': name in approved,
                }
                print(f'{registry_name} v{version_number} registered' + (' — @production' if name in approved else ' — not production'))
    REGISTRY_MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Registry manifest:', REGISTRY_MANIFEST)
    print('Registered models are not yet served by FastAPI. Deployment is the next step.')


if __name__ == '__main__':
    main()
