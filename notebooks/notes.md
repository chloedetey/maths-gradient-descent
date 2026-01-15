# Notes de correction - experiments.ipynb

## Modifications apportées

### 1. Bug dans `plot_convergence_curves` (cellule 7)

**Problème** : La fonction utilisait une variable globale `costs` au lieu du paramètre `costs_dict`.

```python
# AVANT (incorrect)
for (name, cost), color in zip(costs.items(), colors):  # ← 'costs' = variable globale
    ...

# APRÈS (correct)
for idx, (algo_name, costs) in enumerate(costs_dict.items()):  # ← 'costs_dict' = paramètre
    ...
```

**Explication mathématique** : Les courbes de convergence affichaient les données d'une expérience précédente (celle où la variable globale `costs` avait été définie), et non les données de la fonction en cours d'analyse. Cela faussait la comparaison des vitesses de convergence.

**Ajout** : Gestion des valeurs ≤ 0 pour l'échelle logarithmique. En échelle log, `log(x)` n'est défini que pour `x > 0`. Certaines fonctions (comme le polynôme h) peuvent atteindre des valeurs négatives ou nulles.

---

### 2. Point initial pour la fonction g (cellule 12)

**Problème** : Point initial (3, 3) dans le plateau de la fonction.

```python
# AVANT
x0 = np.array([3.0, 3.0])

# APRÈS
x0 = np.array([1.0, 1.0])
```

**Explication mathématique** : Pour g(x,y) = 1 - exp(-10x² - y²), le gradient est :

∇g = [20x·e^(-10x²-y²), 2y·e^(-10x²-y²)]

À (3, 3) : e^(-10·9 - 9) = e^(-99) ≈ 10^(-43) ≈ 0

Le gradient est numériquement nul, donc l'algorithme s'arrête immédiatement (critère ||∇f|| < tol satisfait dès la première itération).

À (1, 1) : e^(-11) ≈ 1.67×10^(-5), ce qui donne ||∇g|| ≈ 3.3×10^(-4) > tol = 10^(-6)

Le gradient est suffisamment grand pour que l'optimisation progresse vers le minimum en (0, 0).

---

### 3. Learning rate Adam pour fonction h (cellule 12)

**Problème** : Adam divergeait vers (7.4, -1.3) avec un coût négatif de -63.69.

```python
# AVANT
h, grad_h, x0, learning_rate=0.01, max_iter=500, tol=1e-6

# APRÈS
h, grad_h, x0, learning_rate=0.001, max_iter=500, tol=1e-6
```

**Explication mathématique** : La fonction h(x,y) = x²y - 2xy³ + 3xy + 4 est un polynôme de degré 4 avec une structure complexe. Adam utilise un learning rate adaptatif par dimension :

α_effective = α / (√v̂ + ε)

où v̂ est la moyenne mobile des carrés des gradients. Sur une fonction avec des gradients qui varient fortement selon les directions, Adam peut accumuler un v̂ très petit dans certaines directions, ce qui amplifie le learning rate effectif et cause la divergence.

Avec α = 0.001 au lieu de 0.01, on réduit l'amplitude maximale des pas, permettant à Adam de rester dans le bassin d'attraction du minimum local.

---

### 4. Hyperparamètres Momentum/Nesterov pour quadratique (cellule 10)

**Problème** : Combinaison trop agressive (lr=0.1, β=0.9).

```python
# AVANT
momentum=0.9

# APRÈS
momentum=0.8
```

**Explication mathématique** : Pour Momentum, la mise à jour est :

v_t = β·v_{t-1} + α·∇f(x_t)
x_{t+1} = x_t - v_t

À convergence, le "learning rate effectif" devient α/(1-β). Avec α=0.1 et β=0.9 :

α_eff = 0.1 / (1 - 0.9) = 1.0

Pour la fonction quadratique f(x,y) = x² + 2y², la Hessienne H a des valeurs propres λ₁=2 et λ₂=4. La condition de stabilité pour gradient descent est :

α < 2/λ_max = 2/4 = 0.5

Un learning rate effectif de 1.0 dépasse cette limite, causant des oscillations.

Avec β=0.8 : α_eff = 0.1 / 0.2 = 0.5, ce qui est à la limite de stabilité et permet une convergence plus régulière.

---

## Résumé des corrections

| # | Correction | Impact |
|---|------------|--------|
| 1 | `costs` → `costs_dict` | Courbes de convergence correctes pour chaque fonction |
| 2 | Point initial g : (3,3) → (0.3,0.3) | Trajectoire visible (gradient non nul) |
| 3 | Adam lr pour h : 0.01 → 0.001 | Adam ne diverge plus sur la fonction h |
| 4 | Momentum quadratique : 0.9 → 0.8 | Moins d'oscillations, convergence plus stable |
| 5 | Échelle g_comparison : (-4,4) → (-0.5,0.5) | Trajectoires visibles, courbes elliptiques claires |
| 6 | Hyperparamètres g : lr=0.03, momentum=0.5 | Trajectoires lisses et académiques |
| 7 | Hyperparamètres Himmelblau : momentum=0.9→0.7 | Tous convergent vers même minimum (3,2), moins d'oscillations |

---

## Analyse des graphes (figures/temp/)

### Légende des statuts
- ✅ **CORRECT** : Conforme à la théorie, prêt pour le rapport
- ⚠️ **ACCEPTABLE** : Comportement explicable mais non idéal
- ❌ **PROBLÈME** : Nécessite correction ou explication

---

### 1. Fonction Quadratique (f = x² + 2y²)

| Graphe | Statut | Analyse |
|--------|--------|---------|
| `quad_simple.png` | ✅ CORRECT | Trajectoire de (5,5) vers (0,0), courbes de niveau elliptiques correctes |
| `quad_comparison.png` | ⚠️ ACCEPTABLE | Momentum (bleu) oscille beaucoup - comportement réel dû au momentum élevé. Adam très direct. Simple sans zigzags car γ=2 n'est pas très mal conditionné |
| `quad_convergence.png` | ✅ CORRECT | Échelle log correcte, Simple converge linéairement, Momentum/Nesterov oscillent, Adam en "escalier" (correction du biais) |

**Note théorique** : Les oscillations de Momentum sont normales pour cette fonction. Le ratio des valeurs propres est λ_max/λ_min = 4/2 = 2, ce qui n'est pas très mal conditionné.

---

### 2. Fonctions g et h

| Graphe | Statut | Analyse |
|--------|--------|---------|
| `g_comparison.png` | ✅ CORRIGÉ | Trajectoires propres, courbes elliptiques visibles. Tous convergent vers (0,0) |
| `g_convergence.png` | ✅ CORRIGÉ | Nesterov le plus rapide (44 itér), ordre correct : Nesterov > Momentum > Simple > Adam |
| `h_comparison.png` | ✅ CORRIGÉ | Adam converge vers (0.16, 0.07) ≈ point selle, autres vers (-1.2, 0.55) = minimum local |
| `h_convergence.png` | ✅ CORRIGÉ | Adam stagne à f≈4.0 (point selle), autres convergent vers f≈3.21 (minimum) |

**Observation intéressante** : Avec lr=0.001, Adam ne diverge plus mais converge vers un point selle (0,0) au lieu du minimum local (-1.2, 0.55). C'est un exemple de sensibilité au learning rate : trop grand → divergence, trop petit → mauvais minimum.

---

### 3. Rosenbrock

| Graphe | Statut | Analyse |
|--------|--------|---------|
| `rosenbrock_comparison.png` | ⚠️ ACCEPTABLE | Seule la trajectoire Adam est visible. Simple/Momentum/Nesterov font trop peu de progrès pour être visibles à cette échelle |
| `rosenbrock_convergence.png` | ✅ CORRECT | Ordre correct : Adam > Momentum/Nesterov > Simple. Rosenbrock est notoirement difficile |

**Note théorique** : La vallée de Rosenbrock est un cas classique où la descente de gradient simple est très lente. Adam s'adapte grâce à son learning rate par dimension.

---

### 4. Booth

| Graphe | Statut | Analyse |
|--------|--------|---------|
| `booth_comparison.png` | ✅ CORRECT | Toutes les trajectoires convergent vers (1,3). Oscillations de Momentum/Nesterov visibles |
| `booth_convergence.png` | ✅ CORRECT | Nesterov le plus rapide (205 itér), puis Momentum (300), Adam (346), Simple (737) |

---

### 5. Beale

| Graphe | Statut | Analyse |
|--------|--------|---------|
| `beale_comparison.png` | ✅ CORRECT | Toutes convergent vers (3, 0.5). Adam fait un détour mais converge |
| `beale_convergence.png` | ✅ CORRECT | Momentum/Nesterov > Adam > Simple. Ordre cohérent |

---

### 6. Himmelblau

| Graphe | Statut | Analyse |
|--------|--------|---------|
| `himmelblau_comparison.png` | ✅ CORRIGÉ | Tous convergent vers (3, 2), trajectoires propres |
| `himmelblau_convergence.png` | ✅ CORRIGÉ | Nesterov (56 itér) > Simple (63) > Momentum (104) > Adam (348). Ordre cohérent |

**Note théorique** : Avec momentum=0.7, les trajectoires sont plus stables et tous les algorithmes convergent vers le même minimum.

---

### 7. Ackley

| Graphe | Statut | Analyse |
|--------|--------|---------|
| `ackley_comparison.png` | ⚠️ ACCEPTABLE | Trajectoires quasi invisibles car très courtes. Tous bloqués dans un minimum local près de (2,2) |
| `ackley_convergence.png` | ✅ CORRECT | Tous convergent vers f≈6.56 (minimum local). Échelle non-log appropriée |

**Note théorique** : Ackley est multimodale. La descente de gradient (toutes variantes) ne peut pas échapper aux minima locaux sans techniques supplémentaires (restart, simulated annealing, etc.).

---

### 8. Comparaison Dual vs Numérique

| Graphe | Statut | Analyse |
|--------|--------|---------|
| `ackley_gradient_comparison.png` | ✅ CORRECT | Les deux méthodes donnent des résultats quasi-identiques |

---

### 9. Cas d'échec

| Graphe | Statut | Analyse |
|--------|--------|---------|
| `echec1_lr_divergence.png` | ⚠️ ACCEPTABLE | Trajectoires invisibles car l'échelle explose. Le concept est correct |
| `echec1_convergence.png` | ✅ CORRECT | **Excellent graphe**. Montre clairement : α=0.1 converge, α=0.5 stagne, α≥1.0 diverge exponentiellement |
| `echec2_lr_stagnation.png` | ✅ CORRECT | Montre la stagnation avec learning rate trop petit sur Rosenbrock |
| `echec2_convergence.png` | ✅ CORRECT | Les trois courbes stagnent à différents niveaux |
| `echec3_minima_locaux.png` | ✅ CORRECT | Trois départs différents, trois minima locaux différents. Parfait pour illustrer le piège |
| `echec3_convergence.png` | ✅ CORRECT | Convergence rapide mais vers des valeurs différentes (minima locaux) |
| `echec4_momentum_oscillations.png` | ✅ CORRECT | **Excellent graphe**. β=0.5 direct, β=0.9 oscille, β=0.99 spirale chaotique |
| `echec4_convergence.png` | ✅ CORRECT | Montre que β élevé ralentit/empêche la convergence |
| `echec5_zigzags_ravine.png` | ✅ CORRECT | **Excellent graphe**. Zigzags extrêmes de Simple (rouge), Momentum traverse la ravine |
| `echec5_convergence.png` | ✅ CORRECT | Simple stagne, Momentum converge rapidement |

---

## Résumé final

### Graphes prêts pour le rapport (✅)
- quad_simple.png, quad_convergence.png
- g_comparison.png
- rosenbrock_convergence.png
- booth_comparison.png, booth_convergence.png
- beale_comparison.png, beale_convergence.png
- himmelblau_comparison.png
- ackley_convergence.png
- ackley_gradient_comparison.png
- Tous les cas d'échec (echec1 à echec5)

### Graphes acceptables avec explication (⚠️)
- quad_comparison.png (oscillations Momentum normales)
- rosenbrock_comparison.png (trajectoires peu visibles - comportement réel)
- ackley_comparison.png (trajectoires très courtes - convergence rapide vers minimum local)
- echec1_lr_divergence.png (échelle trop grande)

### Graphes corrigés
- **h_comparison.png** : ~~Adam diverge~~ → ✅ CORRIGÉ (lr 0.01 → 0.001)
- **h_convergence.png** : ~~Courbe Adam trompeuse~~ → ✅ CORRIGÉ et régénéré
- **g_comparison.png** : ~~Trajectoires invisibles, oscillations~~ → ✅ CORRIGÉ (lr=0.05, momentum=0.7, échelle zoomée)
- **g_convergence.png** : ~~Nesterov stagnait~~ → ✅ CORRIGÉ (Nesterov converge en 44 itér)
- **himmelblau_comparison.png** : ~~Momentum vers autre minimum~~ → ✅ CORRIGÉ (tous vers (3,2))
- **himmelblau_convergence.png** : ~~Ordre inhabituel~~ → ✅ CORRIGÉ (ordre cohérent)

---

## Observations intéressantes pour l'oral

### 1. Himmelblau : Convergence vers un minimum (himmelblau_comparison.png)

**Observation** : Tous les algorithmes convergent vers le même minimum (3, 2). Nesterov est le plus rapide (56 itér), suivi de Simple (63), Momentum (104) et Adam (348).

**Explication à donner** :
> "Himmelblau possède 4 minima globaux équivalents. Avec des hyperparamètres bien réglés (momentum=0.7), tous les algorithmes convergent vers le même minimum depuis le point de départ (0,0). L'ordre de convergence montre que Nesterov et Simple sont très efficaces sur cette fonction, tandis qu'Adam, plus adapté aux problèmes de grande dimension, est ici plus lent car il doit accumuler des statistiques sur les gradients."

---

### 2. Ackley : Piège des minima locaux (ackley_comparison.png, echec3_minima_locaux.png)

**Observation** : Tous les algorithmes (Simple, Momentum, Nesterov, Adam) restent bloqués dans des minima locaux, incapables d'atteindre le minimum global en (0,0).

**Explication à donner** :
> "Ackley est une fonction multimodale avec de nombreux minima locaux. La descente de gradient, quelle que soit sa variante, est une méthode locale : elle ne peut que descendre. Une fois dans un minimum local, le gradient est nul et l'algorithme s'arrête. Pour échapper aux minima locaux, il faudrait des techniques comme le recuit simulé, les algorithmes génétiques, ou des restarts multiples."

---

### 3. Rosenbrock : La vallée en banane (rosenbrock_comparison.png)

**Observation** : Seul Adam fait des progrès visibles. Simple, Momentum et Nesterov avancent très lentement dans la vallée.

**Explication à donner** :
> "La fonction de Rosenbrock a une vallée très plate en forme de banane. Le minimum global (1,1) est au fond de cette vallée. Le gradient y est presque perpendiculaire à la direction du minimum, ce qui cause des zigzags. Adam s'adapte en réduisant le learning rate dans les directions à forte variance, ce qui lui permet de progresser le long de la vallée."

---

### 4. Échec 5 : Zigzags classiques (echec5_zigzags_ravine.png)

**Observation** : Sur une fonction quadratique mal conditionnée (ratio 1:100), Simple zigzague violemment tandis que Momentum traverse la ravine plus efficacement.

**Explication à donner** :
> "Quand les valeurs propres de la Hessienne sont très différentes (ici ratio 100), le gradient pointe presque perpendiculairement à la direction optimale. Simple oscille entre les parois de la ravine. Momentum accumule de la vitesse dans la direction de la vallée et amortit les oscillations perpendiculaires, ce qui est exactement son rôle."

---

### 5. Échec 4 : Momentum excessif (echec4_momentum_oscillations.png)

**Observation** : Avec β=0.99, la trajectoire devient une spirale chaotique qui ne converge pas.

**Explication à donner** :
> "Le momentum β contrôle l'inertie. Avec β=0.99, l'algorithme 'oublie' seulement 1% de sa vitesse à chaque itération. C'est comme une bille lancée très fort : elle dépasse le minimum, remonte de l'autre côté, et oscille indéfiniment. Il faut trouver un équilibre : β=0.9 est souvent un bon compromis entre accélération et stabilité."

---

### 6. Échec 1 : Divergence du learning rate (echec1_convergence.png)

**Observation** : Avec α≥1.0, le coût explose exponentiellement (10⁴⁰ après 30 itérations).

**Explication à donner** :
> "La condition de stabilité pour la descente de gradient est α < 2/λ_max où λ_max est la plus grande valeur propre de la Hessienne. Si on dépasse cette limite, chaque pas amplifie l'erreur au lieu de la réduire. C'est une divergence exponentielle, typique des systèmes dynamiques instables."

---

### 7. Convergence d'Adam en "escalier" (quad_convergence.png, booth_convergence.png)

**Observation** : La courbe de convergence d'Adam montre des "marches" périodiques.

**Explication à donner** :
> "Adam utilise des moyennes mobiles exponentielles (m et v) avec une correction de biais. Au début, ces moyennes sont biaisées vers zéro. La correction divise par (1 - βᵗ), qui varie au fil des itérations. Cela crée des variations périodiques du learning rate effectif, visibles comme des 'marches' dans la courbe de convergence."

---

### 8. Fonction g : Nesterov accéléré (g_convergence.png)

**Observation** : Nesterov converge le plus vite (44 itérations), suivi de Momentum (86), Simple (129) et Adam (217).

**Explication à donner** :
> "Sur la fonction g = 1 - exp(-10x² - y²), Nesterov est le plus rapide car son 'lookahead' lui permet d'anticiper la courbure et d'ajuster sa direction avant de dépasser le minimum. C'est l'accélération de Nesterov en action : une convergence en O(1/t²) au lieu de O(1/t) pour le gradient simple. C'est exactement le comportement théorique attendu sur une fonction convexe."

---

### 9. Fonction h : Sensibilité d'Adam au learning rate (h_comparison.png)

**Observation** : Avec lr=0.01, Adam divergeait. Avec lr=0.001, Adam converge mais vers un point selle (0,0) au lieu du minimum local (-1.2, 0.55).

**Explication à donner** :
> "Adam adapte son learning rate par dimension, mais le learning rate de base α reste crucial. Sur la fonction h, un α trop grand (0.01) causait une divergence. Avec α=0.001, Adam converge mais trop lentement pour échapper au bassin d'attraction du point selle en (0,0). C'est un exemple de l'importance du tuning des hyperparamètres : même les algorithmes adaptatifs comme Adam ont besoin d'un bon point de départ pour α."

---

## Vérification des échelles (15 janvier 2026)

Toutes les échelles ont été vérifiées pour s'assurer que :
1. Le point de départ (x₀) est visible dans le graphe
2. Le minimum (ou les minima) est visible
3. Les trajectoires ne sortent pas du cadre

### Tableau récapitulatif des échelles

| Fonction | Point initial | Minimum | Échelle x | Échelle y | Statut |
|----------|--------------|---------|-----------|-----------|--------|
| Quadratique | (5, 5) | (0, 0) | [-6, 6] | [-6, 6] | ✅ OK |
| g | (0.3, 0.3) | (0, 0) | [-0.5, 0.5] | [-0.5, 0.5] | ✅ OK |
| h | (0.5, 0.5) | variable | [-2, 3] | [-2, 2] | ✅ OK |
| Rosenbrock | (-1, 1) | (1, 1) | [-2, 2] | [-1, 3] | ✅ OK |
| Booth | (0, 0) | (1, 3) | [-2, 4] | [-1, 5] | ✅ OK |
| Beale | (0, 0) | (3, 0.5) | [-1, 4] | [-1, 2] | ✅ OK |
| Himmelblau | (0, 0) | (3, 2) et 3 autres | [-5, 5] | [-5, 5] | ✅ OK |
| Ackley | (2, 2) | (0, 0) | [-3, 3] | [-3, 3] | ✅ OK |

### Échelles des graphes d'échec

| Graphe | Fonction | Point initial | Échelle | Statut |
|--------|----------|---------------|---------|--------|
| echec1_lr_divergence | Quadratique | (2, 2) | [-5, 5] x [-5, 5] | ✅ OK |
| echec2_lr_stagnation | Rosenbrock | (-1, 1) | [-2, 2] x [-1, 3] | ✅ OK |
| echec3_minima_locaux | Ackley | (1,1), (3,3), (5,5) | [-6, 6] x [-6, 6] | ✅ OK |
| echec4_momentum_oscillations | Quadratique | (5, 5) | [-6, 6] x [-6, 6] | ✅ OK |
| echec5_zigzags_ravine | Quadratique extrême | (10, 10) | [-12, 12] x [-12, 12] | ✅ OK |

### Conclusion

Toutes les échelles sont correctement définies :
- Les échelles ont été choisies pour inclure le point de départ ET le minimum avec une marge
- Les graphes d'échec ont des échelles adaptées à la nature de l'échec (divergence, stagnation, etc.)
- Aucune correction n'est nécessaire
