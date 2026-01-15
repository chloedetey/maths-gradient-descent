# Descente de Gradient - Projet de Mathématiques

Projet réalisé dans le cadre d'un TP de mathématiques sur l'optimisation numérique. L'objectif est d'explorer la descente de gradient et ses variantes à travers des implémentations et des visualisations.

## Objectifs

- Comprendre le fonctionnement de la descente de gradient
- Implémenter et comparer 4 algorithmes : Simple, Momentum, Nesterov et Adam
- Tester sur différentes fonctions de benchmark (Rosenbrock, Ackley, Himmelblau...)
- Analyser les cas d'échec : divergence, stagnation, minima locaux

## Structure du projet

```
├── src/                    # Code source
│   ├── functions.py        # Fonctions de test (quadratique, rosenbrock, ackley...)
│   ├── gradients.py        # Calcul du gradient (numérique + dual numbers)
│   ├── optimizers.py       # Les 4 algorithmes d'optimisation
│   └── visualization.py    # Fonctions de visualisation
│
├── notebooks/              # Expériences
│   ├── exp-draft.ipynb     # Notebook principal avec toutes les expériences
│   ├── notes.md            # Notes sur les corrections et observations
│   └── notes-resultats.md  # Résultats attendus pour chaque graphe
│
├── figures/                # Graphes générés
│   └── temp/               # Dernière génération
│
└── rapport/                # Rapport final
```

## Algorithmes implémentés

| Algorithme | Description |
|------------|-------------|
| **Simple** | Descente de gradient classique : x = x - α∇f(x) |
| **Momentum** | Ajoute une "inertie" pour accélérer la convergence |
| **Nesterov** | Variante de Momentum avec un lookahead |
| **Adam** | Learning rate adaptatif par dimension |

## Fonctions testées

- **Quadratique** : f(x,y) = x² + 2y² — fonction simple pour valider les implémentations
- **Rosenbrock** : vallée en banane, difficile à optimiser
- **Booth, Beale** : fonctions classiques de benchmark
- **Himmelblau** : 4 minima globaux équivalents
- **Ackley** : des centaines de minima locaux

## Utilisation

```bash
# Installation des dépendances
pip install -r requirements.txt

# Lancer le notebook
jupyter notebook notebooks/exp-draft.ipynb
```

Les graphes sont sauvegardés dans `figures/temp/`.

## Dépendances

- numpy
- matplotlib
- jupyter
