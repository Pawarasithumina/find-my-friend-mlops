"""FaceOps: freeze validation-based selection, then evaluate held-out test once.

Use from repo root:
 python -m src.evaluation.finalize_models select
 python -m src.evaluation.finalize_models evaluate
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data.dataset_loader import FaceOpsDataset
from src.models.architectures import create_model
from src.storage.checkpoint_fetch import download_trial_checkpoint

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports' / 'private_evaluation'
SELECTION_PATH = OUTPUT / 'selection.json'
EVALUATION_PATH = OUTPUT / 'heldout_results.json'
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5050')
CLASSES = ['pawara', 'uchitha', 'dhamith', 'unknown']
DATASET_VERSION = 2

# The best MLflow trial IDs from the completed three-model Optuna run.
# Deliberately pinned: rerunning Optuna later cannot silently change the shortlist.
RUN_IDS = {
    'custom_cnn': '4af1ae38b417489b80a14c0086b12035',
    'mobilenet_v3': '419640ebb60644a9bf96c97c15254dce',
    'resnet18': '499828f2c005446ca3c7eb6adae41b85',
}


def get_client():
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient(tracking_uri=TRACKING_URI)


def select():
    if SELECTION_PATH.exists():
        print('Selection already frozen:', SELECTION_PATH)
        print(SELECTION_PATH.read_text())
        return

    client = get_client()
    ranked = []
    for name, run_id in RUN_IDS.items():
        run = client.get_run(run_id)
        params = run.data.params
        if params.get('model_name') != name or params.get('dataset_version') != '2':
            raise RuntimeError(f'Model/dataset mismatch for {name}: {run_id}')
        if params.get('train_mode') != 'production':
            raise RuntimeError(f'Not a production-mode trial: {name}')
        if run.info.status != 'FINISHED':
            raise RuntimeError(f'Incomplete MLflow run: {name}')
        if 'best_val_loss' not in run.data.metrics:
            raise RuntimeError(f'Missing validation loss: {name}')
        files = client.list_artifacts(run_id, 'checkpoint')
        if not any(x.path == 'checkpoint/best_state.pt' for x in files):
            raise RuntimeError(f'Checkpoint missing in MLflow: {name}')
        ranked.append({
            'architecture': name,
            'source_run_id': run_id,
            'best_val_loss': float(run.data.metrics['best_val_loss']),
        })

    ranked.sort(key=lambda entry: entry['best_val_loss'])
    frozen = {
        'dataset_version': DATASET_VERSION,
        'selection_rule': 'two lowest best validation cross-entropy losses',
        'selected_for_registry_production_alias': [m['architecture'] for m in ranked[:2]],
        'all_models': ranked,
        'test_set_used_for_selection': False,
        'known_limitation': 'Known-person images each come from one burst; test is not independent-session generalization.',
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(frozen, indent=2) + '\n')
    print('FROZEN VALIDATION-ONLY SELECTION:', SELECTION_PATH)
    print(json.dumps(frozen, indent=2))


def load_trial_model(client, model_record, temporary_dir):
    name = model_record['architecture']
    artifact = download_trial_checkpoint(
        model_record['source_run_id'], temporary_dir
    )
    # weights_only=True avoids arbitrary pickle execution for the known checkpoint format.
    checkpoint = torch.load(artifact, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != name or checkpoint.get('dataset_version') != DATASET_VERSION:
        raise RuntimeError('Checkpoint identity does not match the frozen selection')
    if list(checkpoint.get('classes', [])) != CLASSES:
        raise RuntimeError('Checkpoint class order mismatch')
    model = create_model(
        name, num_classes=4,
        dropout=float(checkpoint['hyperparameters']['dropout']),
        pretrained=False,  # All weights are restored; never download at evaluation.
        freeze_backbone=bool(checkpoint['freeze_backbone']),
    )
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def evaluate():
    if not SELECTION_PATH.exists():
        raise RuntimeError('Freeze selection first: python -m src.evaluation.finalize_models select')
    if EVALUATION_PATH.exists():
        print('Held-out evaluation already recorded. Not rerunning:', EVALUATION_PATH)
        print(EVALUATION_PATH.read_text())
        return

    frozen = json.loads(SELECTION_PATH.read_text())
    if frozen['dataset_version'] != DATASET_VERSION:
        raise RuntimeError('Wrong dataset version')
    if {x['architecture']: x['source_run_id'] for x in frozen['all_models']} != RUN_IDS:
        raise RuntimeError('Frozen selection differs from the pinned training runs')

    client = get_client()
    mlflow.set_experiment('FaceOps_Heldout_Evaluation')
    results = {}
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    for record in frozen['all_models']:
        architecture = record['architecture']
        with tempfile.TemporaryDirectory() as td:
            model, checkpoint = load_trial_model(client, record, Path(td))
            dataset = FaceOpsDataset('test', architecture, augment=False)
            if len(dataset) != 33:
                raise RuntimeError('Unexpected test split size — stop to avoid invalid evaluation')
            loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            true, pred = [], []
            total_loss = 0.0
            with torch.inference_mode():
                for images, labels in loader:
                    logits = model(images)
                    total_loss += float(nn.functional.cross_entropy(logits, labels, reduction='sum'))
                    pred.extend(logits.argmax(dim=1).cpu().tolist())
                    true.extend(labels.cpu().tolist())
            accuracy = float(accuracy_score(true, pred))
            macro_f1 = float(f1_score(true, pred, labels=[0,1,2,3], average='macro', zero_division=0))
            weighted_f1 = float(f1_score(true, pred, labels=[0,1,2,3], average='weighted', zero_division=0))
            details = classification_report(true, pred, labels=[0,1,2,3], target_names=CLASSES, output_dict=True, zero_division=0)
            cm = confusion_matrix(true, pred, labels=[0,1,2,3]).tolist()
            summary = {
                'source_run_id': record['source_run_id'],
                'best_val_loss': record['best_val_loss'],
                'best_epoch': int(checkpoint['best_epoch']),
                'test_images': len(dataset),
                'test_loss': total_loss / len(dataset),
                'test_accuracy': accuracy,
                'test_macro_f1': macro_f1,
                'test_weighted_f1': weighted_f1,
                'per_class': details,
                'confusion_matrix': cm,
            }
            with mlflow.start_run(run_name=f'heldout_{architecture}') as run:
                mlflow.log_params({
                    'architecture': architecture,
                    'source_trial_run_id': record['source_run_id'],
                    'dataset_version': DATASET_VERSION,
                    'preselected_from_validation': architecture in frozen['selected_for_registry_production_alias'],
                    'eval_split': 'test',
                })
                mlflow.log_metrics({k: summary[k] for k in ('test_loss','test_accuracy','test_macro_f1','test_weighted_f1')})
                mlflow.log_dict(summary, 'evaluation/metrics.json')
                fig, ax = plt.subplots(figsize=(6,5))
                ax.imshow(np.array(cm), cmap='Blues')
                ax.set_xticks(range(4), labels=CLASSES, rotation=35, ha='right')
                ax.set_yticks(range(4), labels=CLASSES)
                ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{architecture} — held-out test')
                for i in range(4):
                    for j in range(4):
                        ax.text(j, i, str(cm[i][j]), ha='center', va='center', color='red')
                fig.tight_layout()
                mlflow.log_figure(fig, 'evaluation/confusion_matrix.png')
                plt.close(fig)
                summary['evaluation_run_id'] = run.info.run_id
            results[architecture] = summary
            print(f'{architecture}: test accuracy={accuracy:.4f} macro-F1={macro_f1:.4f}')

    report = {
        'dataset_version': DATASET_VERSION,
        'selection_rule': frozen['selection_rule'],
        'selected_models_frozen_before_test': frozen['selected_for_registry_production_alias'],
        'test_only_evaluated_after_selection': True,
        'generalization_limit': frozen['known_limitation'],
        'models': results,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(report, indent=2) + '\n')
    print('Evaluation report:', EVALUATION_PATH)
    print('NOTE: Scores reflect same-session recording conditions for the known classes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['select', 'evaluate'])
    args = parser.parse_args()
    if args.stage == 'select': select()
    else: evaluate()
