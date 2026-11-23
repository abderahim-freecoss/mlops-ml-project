# Baseline de classification Iris

Ce projet entraîne une régression logistique sur le jeu Iris fourni par scikit-learn.
La configuration se trouve dans `config/train.yaml`. Elle réserve 20 % des données
au test et fixe la graine du découpage à 42.

## Installation

Prérequis : Python 3.10 ou supérieur. Exécuter les commandes depuis la racine du projet.

Sous Windows (PowerShell) :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Sous Linux ou macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Entraînement et évaluation

```text
python scripts/train.py
python scripts/evaluate.py
```

L'entraînement calcule l'accuracy et le F1 macro sur les 30 fleurs du jeu de test.
Le script d'évaluation produit un rapport sur les 150 fleurs, dont les 120 utilisées
pour l'entraînement. Pour estimer les performances sur des données inédites, utiliser
les métriques calculées sur le jeu de test.

## Tests

```text
python -m pytest -q
```

## Artefacts générés

Les scripts écrivent leurs sorties dans `artifacts/`, un dossier destiné à rester hors de Git.

- `model.joblib` contient le pipeline entraîné avec le prétraitement et le modèle.
- `metrics.json` contient l'accuracy et le F1 macro sur le jeu de test.
- `confusion_matrix.png` représente les prédictions sur le jeu de test.
- `report.json` contient le rapport de classification sur le jeu Iris complet.
