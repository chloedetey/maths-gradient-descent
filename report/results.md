# Descente de Gradient - Résultats obtenus

Ce document contient :
1. **Théorie** : explications pour comprendre le sujet
2. **Résultats** : analyse de chaque graphe pour le rapport et l'oral

---

# PARTIE 1 : THÉORIE

## 1. Pourquoi l'optimisation ?

En mathématiques appliquées, beaucoup de problèmes consistent à **minimiser une fonction**. Cette fonction représente souvent une erreur ou un coût qu'on veut réduire.

**Exemples concrets :**
- En IA/Machine Learning : minimiser l'erreur entre les prédictions du modèle et les vraies valeurs
- En traitement du signal : minimiser le bruit dans une image ou un son
- En économie : minimiser les coûts de production

La descente de gradient est l'un des algorithmes les plus utilisés pour résoudre ce type de problèmes.

---

## 2. Le problème d'optimisation

**Objectif :** Trouver le point x* où la fonction f(x) atteint sa valeur minimale.

**Pourquoi c'est difficile ?**
- Quand la fonction dépend de beaucoup de variables (grande dimension), on ne peut pas trouver le minimum de façon analytique (en résolvant f'(x) = 0 à la main)
- On utilise donc des **méthodes numériques** comme la descente de gradient

**Analogie :** Imagine que tu es sur une montagne dans le brouillard et tu veux descendre au point le plus bas. Tu ne vois pas le paysage entier, mais tu peux sentir la pente sous tes pieds. La descente de gradient, c'est simplement : "va toujours dans la direction où ça descend le plus".

---

## 3. Le gradient - c'est quoi ?

Le **gradient** ∇f(x) est un vecteur qui indique :
- La **direction** dans laquelle la fonction augmente le plus rapidement
- La **magnitude** (longueur) indique à quel point ça monte

**Pour minimiser**, on va donc dans la **direction opposée** au gradient (là où ça descend le plus).

**Exemple en 2D :**
Pour f(x,y) = x² + y², le gradient est ∇f = [2x, 2y].
- Au point (3, 4), le gradient est [6, 8] → ça monte vers le haut-droite
- Pour descendre, on va vers le bas-gauche (direction opposée)

---

## 4. L'algorithme de descente de gradient

### La formule fondamentale

```
x_{k+1} = x_k - α × ∇f(x_k)
```

**Traduction :**
- `x_k` : position actuelle (à l'étape k)
- `∇f(x_k)` : gradient au point actuel (direction de la montée)
- `α` : **pas d'apprentissage** (learning rate) - contrôle la taille du pas
- `x_{k+1}` : nouvelle position après la mise à jour

### Les étapes de l'algorithme

1. **Initialisation** : choisir un point de départ x₀ (souvent aléatoire)
2. **Calcul du gradient** : calculer ∇f(x_k) au point actuel
3. **Mise à jour** : appliquer la formule x_{k+1} = x_k - α × ∇f(x_k)
4. **Répéter** les étapes 2-3 jusqu'à convergence

### Critère d'arrêt (convergence)

On s'arrête quand :
- Le gradient devient très petit : ||∇f(x)|| < ε (ex: ε = 10⁻⁶)
- Ou après un nombre maximum d'itérations

**Convergence** = la suite (x_k) tend vers un point où le gradient est nul ou quasi-nul.

---

## 5. Le pas d'apprentissage α (learning rate)

C'est LE paramètre crucial. Il contrôle la taille des pas à chaque itération.

### Trois cas possibles :

| α trop petit | α bien choisi | α trop grand |
|--------------|---------------|--------------|
| Convergence très lente | Convergence rapide | Divergence ! |
| Petits pas prudents | Pas optimaux | Saute par-dessus le minimum |
| Peut prendre des millions d'itérations | Atteint le minimum efficacement | Oscillations qui s'amplifient |

### Règle mathématique de stabilité

Pour une fonction quadratique, la condition de stabilité est :
```
α < 2 / λ_max
```
où λ_max est la plus grande valeur propre de la Hessienne.

**Exemple :** Pour f(x,y) = x² + 2y², la Hessienne a des valeurs propres 2 et 4.
Donc α < 2/4 = 0.5. Au-delà de 0.5, l'algorithme diverge.

---

## 6. Les difficultés de la descente de gradient classique

### 6.1 Minima locaux (fonctions non convexes)

**Problème :** L'algorithme peut rester bloqué dans un minimum local qui n'est pas le minimum global.

**Analogie :** Tu descends dans un petit creux de la montagne, mais ce n'est pas la vallée la plus basse. Comme le gradient est nul dans ce creux, tu ne bouges plus.

**Exemple :** La fonction d'Ackley a des centaines de minima locaux (aspect "boîte à œufs"). L'algorithme tombe dans le trou le plus proche du point de départ.

### 6.2 Plateaux

**Problème :** Sur un plateau, le gradient est très faible → pas très petits → convergence extrêmement lente.

**Analogie :** Tu marches sur un terrain presque plat. Tu sens à peine la pente, donc tu avances très lentement.

### 6.3 Ravines (mauvais conditionnement)

**Problème :** La fonction varie très différemment selon les directions. Le gradient pointe perpendiculairement à la direction du minimum → zigzags.

**Analogie :** Tu es dans une vallée étroite. La pente te pousse vers les parois, pas vers le fond de la vallée. Tu rebondis d'un côté à l'autre sans avancer.

**Exemple :** Pour f(x,y) = x² + 100y² (ratio 1:100), l'algorithme fait des zigzags verticaux extrêmes.

---

## 7. Les variantes améliorées

### 7.1 Momentum (quantité de mouvement)

**Idée :** Ajouter de l'inertie. On garde une partie de la vitesse des itérations précédentes.

**Formule :**
```
v = β × v + α × ∇f(x)    (accumulation de vitesse)
x = x - v                 (mise à jour)
```

**Paramètre β** (souvent 0.9) : contrôle l'inertie
- β = 0 : pas d'inertie (retour à la descente simple)
- β = 0.9 : garde 90% de la vitesse précédente
- β = 0.99 : garde 99% → peut causer des oscillations

**Avantages :**
- Traverse les plateaux (accumule de la vitesse même quand le gradient est faible)
- Réduit les zigzags dans les ravines (la vitesse horizontale s'accumule)

**Analogie :** Une bille qui roule. Elle garde son élan et peut traverser des zones plates ou des petites bosses.

### 7.2 Nesterov (Momentum amélioré)

**Idée :** Calculer le gradient à la position "anticipée" (où on serait si on continuait avec l'élan actuel).

**Formule :**
```
x_lookahead = x - β × v        (position anticipée)
grad = ∇f(x_lookahead)         (gradient à cette position)
v = β × v + α × grad           (mise à jour vitesse)
x = x - v                      (mise à jour position)
```

**Avantage :** Permet de "voir venir" et de corriger plus vite. Convergence théorique en O(1/t²) vs O(1/t) pour la descente simple.

**Analogie :** Au lieu de regarder tes pieds, tu regardes où tu vas atterrir et tu ajustes en conséquence.

### 7.3 Adam (Adaptive Moment Estimation)

**Idée :** Adapter automatiquement le pas d'apprentissage pour chaque direction, en fonction de l'historique des gradients.

**Formule simplifiée :**
```
m = β₁ × m + (1-β₁) × grad       (moyenne des gradients)
v = β₂ × v + (1-β₂) × grad²      (moyenne des gradients²)
x = x - α × m / (√v + ε)         (mise à jour adaptative)
```

**Paramètres classiques :** β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸

**Avantages :**
- S'adapte automatiquement à la géométrie de la fonction
- Très robuste sur les problèmes complexes (deep learning)
- Réduit le pas dans les directions qui oscillent, l'augmente dans les directions stables

**Inconvénient :** Sur des fonctions simples, Adam est "overkill" et peut être plus lent que les méthodes simples (il perd du temps à accumuler des statistiques).

---

## 8. Résumé comparatif des algorithmes

| Algorithme | Avantages | Inconvénients | Quand l'utiliser |
|------------|-----------|---------------|------------------|
| **Simple** | Facile à comprendre, peu de paramètres | Zigzags, lent sur ravines | Fonctions simples, pédagogie |
| **Momentum** | Traverse plateaux et ravines | Peut osciller si β trop grand | Fonctions avec plateaux/ravines |
| **Nesterov** | Plus rapide que Momentum, anticipe | Un peu plus complexe | Amélioration de Momentum |
| **Adam** | Très robuste, s'adapte tout seul | Lent sur fonctions simples | Deep learning, problèmes complexes |

---

## 9. Points clés pour l'oral

### Questions fréquentes et réponses :

**Q : C'est quoi le gradient ?**
R : Un vecteur qui pointe vers la direction de plus forte montée. Pour descendre, on va dans la direction opposée.

**Q : Pourquoi l'algorithme peut échouer ?**
R : 3 raisons principales : (1) minima locaux, (2) plateaux, (3) ravines/mauvais conditionnement.

**Q : C'est quoi le learning rate ?**
R : La taille du pas à chaque itération. Trop petit = lent, trop grand = diverge.

**Q : Pourquoi Adam est utilisé en IA ?**
R : Il s'adapte automatiquement et fonctionne bien sur des problèmes complexes avec beaucoup de paramètres.

**Q : C'est quoi la convergence ?**
R : Quand l'algorithme s'arrête car le gradient est devenu quasi-nul (on est arrivé à un minimum).

---

# PARTIE 2 : ANALYSE DES GRAPHES

Guide pour le rapport et l'oral. Pour chaque graphe :
- **Pertinence** : ⭐ = dans le rapport, très pertinent
- **Rapport** : version courte à écrire
- **Oral** : notes détaillées si le prof pose des questions
- **Explication** : est-ce normal ? erreurs ? comment interpréter ?

---

## 1. Fonction Quadratique f(x,y) = x² + 2y²

### quad_simple.png
**Rapport :**
La descente de gradient simple converge vers le minimum (0,0) mais avec des zigzags. Ces oscillations viennent du fait que la fonction est plus "pentue" en y qu'en x (coefficient 2), donc l'algorithme corrige trop fort en y à chaque pas.

**Oral :**
Les zigzags apparaissent quand la fonction n'a pas la même courbure dans toutes les directions. Ici, la courbure en y est 2 fois plus forte qu'en x. Le gradient pointe vers la direction de plus forte pente, mais cette direction n'est pas celle du minimum. Du coup l'algo oscille perpendiculairement à la vallée au lieu d'aller droit au but.

**Explication :**
Graphe correct et attendu. Les zigzags sont normaux pour une fonction quadratique avec des courbures différentes (γ = 2). C'est exactement ce comportement qu'on veut illustrer. Pas d'erreur.

---

### quad_comparison.png ⭐
**Rapport :**
Les 4 algorithmes convergent vers (0,0). Sur cette fonction quadratique simple, Nesterov converge le plus vite (67 itérations), suivi de Simple (74). Momentum fait des oscillations importantes à cause de l'inertie accumulée (148 itérations). Adam, bien qu'il ait une trajectoire directe, est le plus lent ici (241 itérations).

**Oral :**
- **Simple** : trajectoire assez directe sur cette fonction simple
- **Momentum** : accumule trop d'inertie et fait des spirales autour du minimum avant de converger
- **Nesterov** : oscille aussi mais l'anticipation lui permet de corriger plus vite, d'où le meilleur résultat
- **Adam** : trajectoire directe mais lent car il doit d'abord "apprendre" les statistiques du gradient. Adam est conçu pour des problèmes complexes, sur une fonction simple il est "overkill".

**Explication :**
Graphe correct. Le fait qu'Adam soit le plus lent sur une fonction simple est NORMAL - Adam est conçu pour des problèmes complexes (deep learning). Sur une quadratique, son mécanisme d'adaptation est un handicap car il perd du temps à accumuler des statistiques. Les spirales de Momentum sont aussi normales : avec β = 0.8 ou 0.9, l'inertie peut causer des dépassements. Pas d'erreur.

---

### quad_convergence.png ⭐
**Rapport :**
Le graphe montre le coût (valeur de f) en fonction des itérations. Tous les algorithmes atteignent un coût proche de 0. L'ordre de convergence est : Nesterov (67), Simple (74), Momentum (148), Adam (241).

**Oral :**
L'échelle logarithmique permet de voir les différences quand les valeurs deviennent très petites. Momentum oscille (courbe en dents de scie) avant de converger. Adam démarre lentement car il accumule des statistiques au début. Sur une fonction simple, les méthodes simples suffisent - Adam montre son avantage sur des fonctions plus complexes comme Rosenbrock.

**Explication :**
Graphe correct. Les oscillations de Momentum (dents de scie) sont normales et cohérentes avec les spirales visibles sur quad_comparison. La lenteur d'Adam au démarrage est due à la correction de biais (les moments m et v sont initialisés à 0). Pas d'erreur.

---

## 2. Fonctions g et h

### g_comparison.png
**Rapport :**
La fonction g = 1 - exp(-10x² - y²) a un plateau loin de l'origine où le gradient est quasi nul. Les algorithmes démarrent sur ce plateau et descendent vers le minimum en (0,0). Nesterov converge le plus rapidement.

**Oral :**
Un plateau c'est une zone où la fonction est presque plate (gradient ≈ 0). L'algo avance très lentement car il fait des petits pas. C'est un piège classique. Momentum/Nesterov aident car ils accumulent de la vitesse même quand le gradient est faible.

**Explication :**
Graphe correct. Si le point de départ est trop loin (ex: (3,3)), le gradient est quasi nul (exp(-99) ≈ 0) et l'algo ne bouge pas. C'est pour ça qu'on a choisi un point de départ plus proche (0.3, 0.3) ou (1,1). Comportement attendu.

---

### g_convergence.png
**Rapport :**
Nesterov converge en 44 itérations, Momentum en 86, Simple en 129 et Adam en 217. Sur cette fonction, l'accélération de Nesterov est particulièrement efficace.

**Oral :**
Adam est plus lent ici car il doit "apprendre" les statistiques du gradient avant de bien s'adapter. Sur une fonction simple comme celle-ci, Nesterov suffit. Adam brille plutôt sur des fonctions compliquées avec beaucoup de dimensions (comme en deep learning).

**Explication :**
Graphe correct. L'ordre Nesterov > Momentum > Simple > Adam est cohérent avec la théorie : Nesterov a une convergence accélérée O(1/t²) vs O(1/t) pour Simple. Adam est lent car fonction trop simple pour lui. Pas d'erreur.

---

### h_comparison.png
**Rapport :**
La fonction polynomiale h a un paysage complexe. Les algorithmes convergent vers différents points selon leurs trajectoires. Adam converge vers un point selle près de l'origine tandis que les autres trouvent un minimum local.

**Oral :**
Un point selle c'est un point où le gradient est nul mais qui n'est ni un min ni un max (comme une selle de cheval). Adam s'y fait piéger à cause de son learning rate trop petit (0.001). Ça montre que même les bons algos peuvent échouer si les paramètres sont mal réglés.

**Explication :**
Graphe un peu problématique mais acceptable. Adam se fait piéger à cause du learning rate (0.001) qu'on a dû réduire pour éviter la divergence. C'est un cas particulier qui montre la sensibilité aux hyperparamètres. Pas idéal pour le rapport car le message est confus.

---

### h_convergence.png
**Rapport :**
Les courbes montrent qu'Adam stagne à un coût plus élevé (point selle) tandis que les autres algorithmes atteignent un coût plus bas (minimum local).

**Oral :**
Pas très intéressant pour le rapport car c'est juste un exemple où Adam se fait piéger. Le message c'est : aucun algorithme n'est parfait, tout dépend de la fonction et des paramètres.

**Explication :**
Graphe correct mais pas très utile. Éviter dans le rapport car ça complique le message. La fonction h est un cas particulier qui ne représente pas bien le comportement général des algorithmes.

---

## 3. Rosenbrock (vallée en banane)

### rosenbrock_comparison.png ⭐
**Rapport :**
La fonction de Rosenbrock est un test classique en optimisation. Elle forme une vallée étroite et courbée (visible sur les lignes de niveau). Seul Adam (orange) fait des progrès visibles vers le minimum (1,1), en suivant la vallée. Simple, Momentum et Nesterov sont quasi-immobiles près du point de départ (-1, 1) malgré 2001 itérations.

**Oral :**
Rosenbrock c'est LA fonction difficile en optimisation. Le minimum est au fond d'une vallée en forme de banane. Le problème : dans la vallée, le gradient pointe perpendiculairement aux parois, pas vers le minimum. Du coup Simple/Momentum font des micro-pas sans avancer (leurs trajectoires sont invisibles car trop petites). Adam s'adapte car il réduit le learning rate dans les directions qui oscillent et l'augmente dans la direction de la vallée.

**Explication :**
Graphe parfait et très important. Le fait que seul Adam soit visible est normal - les autres font des pas si petits qu'on ne voit pas leur trajectoire. Adam converge en 1568 itérations, les autres n'ont pas convergé après 2001. C'est LE graphe qui montre pourquoi Adam est utilisé en deep learning. À inclure absolument.

---

### rosenbrock_convergence.png ⭐
**Rapport :**
Adam (orange) atteint un coût très faible (≈10⁻⁸) en ~1500 itérations. Nesterov (vert) converge aussi mais plus lentement, atteignant ~10⁻⁸ à 2000 itérations. Simple (rouge) stagne autour de 10⁻¹. Momentum (bleu) est superposé avec Simple.

**Oral :**
C'est pour ce genre de fonction qu'Adam a été inventé. En deep learning, on a souvent des "vallées" similaires dans l'espace des paramètres. La différence de performance entre Adam et Simple est de 7 ordres de grandeur (10⁻¹ vs 10⁻⁸), c'est énorme. Nesterov rattrape Adam à la fin car son accélération finit par payer sur le long terme.

**Explication :**
Graphe parfait. Adam converge vite (1568 iter), Nesterov rattrape vers 2000 iter, Simple/Momentum stagnent. La différence de 7 ordres de grandeur est impressionnante. Pas d'erreur, résultat très parlant.

---

## 4. Booth

### booth_comparison.png
**Rapport :**
Tous les algorithmes convergent facilement vers le minimum (1,3). Cette fonction est relativement simple et ne pose pas de difficulté particulière.

**Oral :**
Booth c'est un cas "facile" pour montrer que tous les algos marchent quand la fonction est gentille. Pas très intéressant pour le rapport, sert juste de point de comparaison.

**Explication :**
Graphe correct mais sans intérêt. Tous les algos convergent car la fonction est trop facile (bien conditionnée, pas de minima locaux). À éviter dans le rapport car n'apporte rien.

---

### booth_convergence.png
**Rapport :**
Convergence rapide pour tous (< 100 itérations pour la plupart). Les différences entre algorithmes sont moins marquées que sur des fonctions difficiles.

**Oral :**
Même remarque, fonction trop facile pour voir des différences significatives.

**Explication :**
Graphe correct mais inutile. Quand tout marche bien, on ne voit pas les différences entre algorithmes. À éviter.

---

## 5. Beale

### beale_comparison.png
**Rapport :**
La fonction de Beale a des gradients qui varient fortement selon la zone. Tous les algorithmes convergent vers le minimum (3, 0.5) mais avec des vitesses différentes.

**Oral :**
Beale est intéressante car le gradient peut être très grand près de l'origine et très petit ailleurs. Ça demande d'adapter le learning rate. Adam le fait automatiquement, les autres ont besoin d'un learning rate bien choisi.

**Explication :**
Graphe correct. Beale est une fonction classique de benchmark. Les résultats sont cohérents avec la théorie. Peut être inclus si on veut montrer un autre exemple de fonction de test.

---

### beale_convergence.png
**Rapport :**
Momentum et Nesterov convergent plus vite qu'Adam sur cette fonction. Simple est le plus lent.

**Oral :**
Contre-exemple intéressant : Adam n'est pas toujours le meilleur ! Sur certaines fonctions "classiques", Momentum/Nesterov suffisent largement.

**Explication :**
Graphe correct et intéressant. Montre qu'Adam n'est pas toujours le meilleur - sur des fonctions "normales", Momentum/Nesterov peuvent suffire. Bon contre-exemple si le prof demande "Adam est-il toujours le meilleur ?".

---

## 6. Himmelblau

### himmelblau_comparison.png ⭐
**Rapport :**
Himmelblau possède 4 minima globaux équivalents (visibles aux 4 coins). Tous les algorithmes convergent vers le même minimum (3,2) depuis le point de départ (0,0). Nesterov est le plus rapide (56 iter), suivi de Simple (63), puis Momentum (104) qui fait un dépassement visible avant de converger. Adam est le plus lent (348 iter).

**Oral :**
C'est un bon exemple pour expliquer que la descente de gradient trouve UN minimum, pas forcément LE minimum global. Si on partait d'un autre point, on pourrait tomber sur un des 3 autres minima. Le dépassement de Momentum (boucle bleue) est dû à l'inertie accumulée - il "dépasse" le minimum puis revient. Adam est lent car la fonction est simple et il perd du temps à accumuler ses statistiques.

**Explication :**
Graphe correct et pédagogique. L'ordre Nesterov (56) > Simple (63) > Momentum (104) > Adam (348) est cohérent : sur une fonction "facile", les méthodes simples suffisent. La boucle de Momentum est normale (inertie). Adam est lent car overkill pour cette fonction. Pas d'erreur.

---

### himmelblau_convergence.png
**Rapport :**
Convergence rapide pour tous les algorithmes (< 100 itérations). Nesterov est légèrement plus rapide.

**Oral :**
Rien de spécial, la fonction est assez "gentille" malgré ses 4 minima.

**Explication :**
Graphe correct mais peu informatif. La convergence est trop facile pour voir des différences marquées. Le graphe de trajectoires (himmelblau_comparison) est plus intéressant.

---

## 7. Ackley

### ackley_comparison.png ⭐
**Rapport :**
Ackley est une fonction avec des centaines de minima locaux (aspect "boîte à œufs" visible sur les lignes de niveau). Le minimum global est au centre (0,0) en rose. Tous les algorithmes se font piéger immédiatement dans un minimum local près du point de départ (2,2) - les trajectoires sont quasi-invisibles car très courtes.

**Oral :**
C'est LE cas d'échec de la descente de gradient. Peu importe l'algorithme, une fois dans un "trou" local, on ne peut pas en sortir car le gradient est nul. Simple converge en 22 iter, Nesterov en 34, mais vers un minimum LOCAL. Momentum (226) et Adam (241) oscillent plus longtemps mais finissent piégés aussi. Pour résoudre ce problème, il faudrait des méthodes globales comme le recuit simulé ou les algorithmes génétiques.

**Explication :**
Graphe parfait pour illustrer les limites de la descente de gradient. L'échec collectif est NORMAL et ATTENDU. Les trajectoires sont invisibles car les algos tombent directement dans le minimum local le plus proche. Ackley est conçue pour piéger les méthodes locales. Pas d'erreur.

---

### ackley_convergence.png ⭐
**Rapport :**
Toutes les courbes convergent vers ~6.55, qui correspond au minimum local où les algorithmes sont piégés. Le minimum global (valeur 0) n'est jamais atteint. Adam (orange) et Momentum (bleu) oscillent fortement au début avant de se stabiliser.

**Oral :**
La valeur finale (~6.55) est la profondeur du minimum local. Simple (22 iter) et Nesterov (34 iter) s'y stabilisent vite. Adam oscille beaucoup au début (pics jusqu'à 6.8) car ses statistiques s'adaptent, puis il se calme. Momentum oscille aussi à cause de l'inertie. Au final, tous finissent au même endroit.

**Explication :**
Graphe correct. La valeur finale ~6.55 (au lieu de 0) prouve qu'on est dans un minimum local. Les oscillations d'Adam et Momentum au début sont normales. L'échelle fine (6.55 à 6.8) permet de voir ces oscillations. Pas d'erreur.

---

## 8. Comparaison Gradient Numérique vs Dual Numbers

### ackley_gradient_comparison.png
**Rapport :**
Les deux méthodes de calcul du gradient (différences finies et nombres duaux) donnent des résultats quasi-identiques. Les nombres duaux sont plus précis car ils calculent la dérivée exacte, sans approximation.

**Oral :**
- **Numérique** : on approxime f'(x) ≈ (f(x+h) - f(x)) / h avec h très petit. Simple mais approximatif.
- **Dual numbers** : on utilise des nombres de la forme a + bε où ε² = 0. Quand on calcule f(a + ε), on obtient f(a) + f'(a)ε automatiquement. C'est de la "dérivation automatique".

La différence entre les deux est < 0.03%, donc négligeable en pratique. Mais les dual numbers sont plus élégants mathématiquement.

**Explication :**
Graphe correct. Les deux méthodes donnent presque la même chose car h = 10⁻⁵ est assez petit pour une bonne approximation. La petite différence (0.03%) vient de l'erreur de troncature du gradient numérique. Les dual numbers sont exacts. Pas d'erreur.

---

## 9. Cas d'échec

### echec1_lr_divergence.png ⭐
**Rapport :**
Quand le learning rate est trop grand (α > 0.5 pour cette fonction), l'algorithme diverge au lieu de converger. α=0.4 (orange) converge le plus vite (33 iter). α=0.1 (vert) converge avec des zigzags (51 iter). α=0.52 et α=0.6 (rouge) divergent avec des oscillations qui s'amplifient.

**Oral :**
La limite de stabilité est α < 2/λ_max où λ_max est la plus grande valeur propre de la Hessienne. Pour f(x,y) = x² + 2y², la Hessienne a des valeurs propres 2 et 4, donc α < 2/4 = 0.5. Au-delà, chaque pas "dépasse" le minimum et amplifie l'erreur. On voit bien que α=0.52 (juste au-dessus de 0.5) oscille sur l'axe y, et α=0.6 explose complètement.

**Explication :**
Graphe parfait et très pédagogique. La limite α = 0.5 est mathématiquement correcte (2/λ_max = 2/4). Les valeurs choisies montrent bien : α=0.4 optimal, α=0.52 instable, α=0.6 divergent. Pas d'erreur, excellent graphe.

---

### echec1_convergence.png ⭐
**Rapport :**
α=0.4 (orange) converge le plus vite : de 10¹ à 10⁻¹¹ en 30 iter. α=0.1 (vert) converge plus lentement jusqu'à 10⁻⁵. α=0.52 (saumon) et α=0.6 (rouge) DIVERGENT : le coût monte jusqu'à 10⁵ et 10¹⁰ respectivement.

**Oral :**
Un coût qui monte = échec garanti. C'est le test le plus simple pour détecter un learning rate trop grand. On voit clairement la séparation : en dessous de α=0.5 ça descend, au-dessus ça monte. En pratique, on commence avec un petit α et on augmente progressivement jusqu'à trouver l'optimum (ici α=0.4).

**Explication :**
Graphe parfait. Deux courbes descendent (convergence), deux montent (divergence). L'échelle log permet de voir 20 ordres de grandeur de différence (10⁻¹¹ vs 10¹⁰). Pas d'erreur, très pédagogique.

---

### echec2_lr_stagnation.png ⭐
**Rapport :**
À l'inverse, un learning rate trop petit cause une stagnation sur Rosenbrock. α=0.001 (rouge) avance un peu dans la vallée. α=0.0001 (orange) et α=1e-05 (jaune) sont quasi-immobiles près du départ (-1, 1). Le minimum global (1,1) est marqué en bleu, très loin.

**Oral :**
Avec α=0.00001, après 1001 itérations on n'a presque pas bougé ! Il faudrait des millions d'itérations pour atteindre (1,1). C'est le compromis du learning rate : trop grand = divergence (echec1), trop petit = stagnation (echec2).

**Explication :**
Graphe correct. La stagnation est visible : α=0.001 avance un peu dans la vallée, les autres sont immobiles. Le point bleu (1,1) montre bien la distance restante. Bon complément à echec1.

---

### echec2_convergence.png ⭐
**Rapport :**
α=0.001 (rouge) descend de ~4 à ~0.1 en 1000 iter - progrès visible mais lent. α=0.0001 (orange) et α=1e-05 (jaune) sont quasi-plats, stagnant autour de 4.

**Oral :**
Même α=0.001 (le plus "grand") n'atteint que ~0.1 après 1000 iter. Le minimum est à 0. C'est pour ça qu'on utilise des techniques comme le "learning rate scheduling" ou Adam qui adapte automatiquement.

**Explication :**
Graphe correct. La différence entre α=0.001 (qui descend) et les deux autres (quasi-plats) est bien visible. Pas d'erreur.

---

### echec3_minima_locaux.png ⭐
**Rapport :**
Trois exécutions avec des points de départ différents sur Ackley : Proche (1,1) en 23 iter, Moyen (3,3) en 21 iter, Loin (5,5) en 20 iter. Seul le départ proche (1,1) se dirige vers le minimum global (0,0) visible au centre en rose. Les trajectoires sont quasi-invisibles car très courtes (piégeage immédiat).

**Oral :**
Le point de départ est crucial sur les fonctions multimodales. Si on démarre dans le "bassin d'attraction" du minimum global, on le trouve. Sinon, on tombe dans un minimum local. Les trajectoires sont si courtes qu'on ne les voit presque pas - l'algo tombe directement dans le trou le plus proche. C'est pourquoi en pratique on fait plusieurs essais avec des points de départ aléatoires ("random restarts").

**Explication :**
Graphe parfait. Les points de départ (1,1), (3,3), (5,5) sont visibles aux positions correspondantes sur la boîte à œufs. Seul (1,1) est dans le bassin d'attraction du global (0,0). Les autres sont piégés immédiatement. Pas d'erreur.

---

### echec3_convergence.png ⭐
**Rapport :**
Les trois courbes stagnent à des niveaux différents : Proche (1,1) vert à ~3, Moyen (3,3) orange à ~8, Loin (5,5) rouge à ~13. Le seuil de succès (f<0.1) en pointillé bleu n'est atteint par aucune courbe.

**Oral :**
Même le départ "proche" (1,1) n'atteint pas vraiment le minimum global - il stagne à ~3 au lieu de 0. C'est parce que sur Ackley, même (1,1) tombe dans un minimum local proche du global mais pas exactement dedans. Le graphe montre que plus on part loin, plus le minimum local est "haut" (moins bon).

**Explication :**
Graphe correct mais attention : AUCUNE courbe ne passe sous le seuil 0.1. Même (1,1) est piégé dans un minimum local à ~3, pas au global (0). C'est cohérent avec la structure d'Ackley où les minima locaux sont partout. Pas d'erreur, illustre bien la difficulté.

---

### echec4_momentum_oscillations.png ⭐
**Rapport :**
β=0.5 (vert) converge en 45 iter avec une spirale serrée vers (0,0). β=0.9, 0.95 et 0.99 font 201 iter avec des spirales de plus en plus grandes. β=0.99 (rose) fait des spirales géantes qui couvrent tout le graphe.

**Oral :**
Le momentum c'est comme une bille qui roule : β contrôle à quel point elle "garde sa vitesse". Avec β = 0.99, elle garde 99% de sa vitesse à chaque pas, donc elle ne freine presque pas et dépasse le minimum encore et encore. Avec β = 0.5, elle freine assez pour spiraler vers le centre. Les spirales sont de plus en plus grandes quand β augmente.

**Explication :**
Graphe parfait et très visuel. Les 4 valeurs de β montrent clairement la progression : β=0.5 (spirale serrée), β=0.9 (spirale moyenne), β=0.95 (grande spirale), β=0.99 (spirale géante). Pas d'erreur, excellent pour illustrer l'effet du momentum.

---

### echec4_convergence.png ⭐
**Rapport :**
β=0.5 (vert) descend rapidement jusqu'à 10⁻¹¹. β=0.9 (bleu) oscille en dents de scie mais atteint 10⁻⁷. β=0.95 (orange) oscille plus fort, atteint ~10⁻³. β=0.99 (rose) stagne vers 10¹ et ne converge pas vraiment.

**Oral :**
Les dents de scie correspondent aux spirales : à chaque tour de spirale, on repasse par le minimum (creux) puis on s'en éloigne (pic). Plus β est grand, plus les oscillations sont violentes. β=0.9 est un bon défaut en pratique.

**Explication :**
Graphe parfait. β=0.5 converge proprement, β=0.9/0.95 oscillent mais convergent, β=0.99 stagne. Les dents de scie correspondent aux spirales du graphe de trajectoires. Pas d'erreur.

---

### echec5_zigzags_ravine.png ⭐
**Rapport :**
Sur une fonction très mal conditionnée (f = x² + 100y², γ=100), Simple (rouge) fait des zigzags verticaux extrêmes qui occupent tout le graphe mais n'avancent pas vers (0,0). Momentum (bleu) zigzague aussi mais converge vers l'optimum. Les deux font 301 itérations.

**Oral :**
Le "conditionnement" d'une fonction c'est le ratio entre sa plus grande et sa plus petite courbure. Ici c'est 100, ce qui est très mal conditionné. Les lignes de niveau sont des ellipses très aplaties (presque des lignes horizontales). Le gradient pointe presque verticalement, d'où les zigzags de Simple qui "rebondit" entre les parois sans avancer. Momentum accumule de la vitesse horizontale et finit par traverser.

**Explication :**
Graphe parfait et très visuel. Simple (rouge) rebondit verticalement de +10 à -10 sans progresser vers x=0. Momentum (bleu) zigzague aussi mais avec une composante horizontale qui le fait converger. C'est LE graphe qui justifie l'invention du momentum. Pas d'erreur.

---

### echec5_convergence.png ⭐
**Rapport :**
Simple (rouge) STAGNE à ~10⁴ et ne converge pas du tout ! Momentum (bleu) descend en dents de scie de 10⁴ jusqu'à 10⁻¹⁰ en 300 itérations. C'est une différence de 14 ordres de grandeur.

**Oral :**
C'est exactement pour ce cas que le momentum a été inventé : Simple est piégé dans les zigzags et ne peut pas avancer, son coût reste constant. Momentum accumule de la vitesse horizontale et converge malgré les oscillations (visibles en dents de scie). La différence est spectaculaire : 10⁴ vs 10⁻¹⁰.

**Explication :**
Graphe parfait et très parlant. Simple (ligne rouge plate en haut) ne converge PAS - c'est un échec total. Momentum (bleu en dents de scie) converge malgré les oscillations. Différence de 14 ordres de grandeur. Pas d'erreur, excellent pour le rapport.

---

## Ordre pour le rapport

### Partie 1 : Présentation des algorithmes (fonction quadratique)
1. **quad_comparison.png** → présente les 4 algos et leurs trajectoires
2. **quad_convergence.png** → compare leurs vitesses de convergence

### Partie 2 : Cas difficile - la vallée de Rosenbrock
3. **rosenbrock_comparison.png** → montre la difficulté (vallée étroite)
4. **rosenbrock_convergence.png** → Adam clairement supérieur (4 ordres de grandeur)

### Partie 3 : Problème des minima multiples / locaux
5. **himmelblau_comparison.png** → 4 minima équivalents, le départ compte
6. **ackley_comparison.png** → tous piégés en local, échec collectif
7. **echec3_minima_locaux.png** → 3 départs → 3 résultats différents
8. **echec3_convergence.png** → preuve visuelle (1 seul trouve le global)

### Partie 4 : Importance du learning rate
9. **echec1_lr_divergence.png** → α trop grand = divergence
10. **echec1_convergence.png** → le coût explose

### Partie 5 : Utilité du momentum
11. **echec5_zigzags_ravine.png** → Simple zigzague, Momentum traverse

### Optionnels 
- **g_comparison.png** → illustre les plateaux
- **ackley_gradient_comparison.png** → dual numbers vs numérique
- **echec2** → learning rate trop petit (stagnation)
- **echec4** → momentum trop fort (oscillations)

### Pas dedans
- booth → trop facile, rien d'intéressant
- h → cas confus, pas parlant
