# Résultats Attendus - Expériences Gradient Descent

Ce fichier contient tous les résultats attendus pour chaque graphe généré, ainsi que le cahier des charges correspondant.

---

## 1. Fonction Quadratique f(x,y) = x² + 2y²

### quad_simple.png
**Résultat attendu :**
- Courbes de niveau : Ellipses concentriques centrées en (0,0), plus serrées selon y
- Trajectoire : Zigzags en descendant de (5,5) vers (0,0), oscillations perpendiculaires
- Point vert en (5,5), point rouge près de (0,0)

**Cahier des charges :**
- ✓ Fonction simple f(x,y) = x² + γy² avec γ=2
- ✓ Illustre les pièges : ZIGZAGS dans les ravines
- ✓ Implémentation : Descente SIMPLE
- ✓ Étude du rôle du point initial (5,5)

---

### quad_comparison.png
**Résultat attendu :**
- 4 trajectoires de couleurs différentes (rouge/bleu/vert/orange) depuis (5,5) vers (0,0)
- Simple (rouge) : zigzags marqués
- Momentum (bleu) : trajectoire plus lisse, légèrement plus directe
- Nesterov (vert) : encore plus lisse que Momentum
- Adam (orange) : trajectoire la plus directe et rapide
- Les nombres d'itérations dans la légende montrent : Simple > Momentum > Nesterov > Adam

**Cahier des charges :**
- ✓ Comparaison des 4 algorithmes : Simple, Momentum, Nesterov, Adam
- ✓ Illustre l'amélioration successive : Simple → Momentum → Nesterov → Adam
- ✓ Visualisation des différences de trajectoires

---

### quad_convergence.png
**Résultat attendu :**
- Axe Y en échelle logarithmique (10⁰, 10⁻², 10⁻⁴, etc.)
- 4 courbes qui descendent toutes vers 0
- Adam descend le plus vite (atteint 10⁻¹⁰ en ~50 itérations)
- Nesterov légèrement meilleur que Momentum
- Simple est le plus lent
- Toutes les courbes se stabilisent à f≈0 (convergence réussie)

**Cahier des charges :**
- ✓ Comparer les VITESSES (nombre de pas nécessaire)
- ✓ Quantifier les performances : Adam > Nesterov > Momentum > Simple
- ✓ Visualisation de la convergence

---

## 2. Fonction g : 1 - exp(-10x² - y²)

### g_comparison.png
**Résultat attendu :**
- Courbes de niveau : Cercles concentriques (légèrement elliptiques) centrés en (0,0)
- Zone JAUNE/VERTE large (plateau) loin de l'origine où g≈1
- Zone BLEUE au centre où g≈0
- Trajectoires de (0.3,0.3) vers (0,0)
- Dans le plateau : progression très lente (gradient faible)
- Près du centre : accélération visible
- Adam traverse le plateau plus rapidement grâce à son adaptation

**Cahier des charges :**
- ✓ Fonction simple : g(x,y) = 1 - exp(-10x² - y²)
- ✓ Illustre les pièges : PLATEAUX (gradient très faible loin de l'origine)
- ✓ Test des 4 algorithmes

---

### g_convergence.png
**Résultat attendu :**
- Toutes les courbes commencent haut (g(0.3,0.3) ≈ 0.63)
- Descente LENTE au début (plateau) puis accélération près de 0
- Forme en "L" caractéristique : plat puis descente rapide
- Adam maintient une meilleure vitesse dans le plateau
- Convergence finale vers g ≈ 0 pour tous

**Cahier des charges :**
- ✓ Visualise le problème du plateau (progression lente)
- ✓ Compare vitesses sur fonction avec plateau

---

## 3. Fonction h : x²y - 2xy³ + 3xy + 4

### h_comparison.png
**Résultat attendu :**
- Courbes de niveau avec forme complexe (pas d'ellipses simples)
- Paysage non-convexe avec possibles zones plates ou crêtes
- Les 4 algorithmes convergent (normalement vers le même minimum local)
- Trajectoires variées selon l'algorithme (momentum peut explorer différemment)
- Illustration d'un paysage polynomial complexe

**Cahier des charges :**
- ✓ Fonction simple : h(x,y) = x²y - 2xy³ + 3xy + 4
- ✓ Test sur paysage non-convexe avec termes croisés
- ✓ Vérification de robustesse des algorithmes

---

### h_convergence.png
**Résultat attendu :**
- Convergence vers une valeur (pas nécessairement 0, dépend du minimum trouvé)
- Descente régulière pour tous les algorithmes
- Adam probablement plus rapide
- Tous atteignent le même minimum local (si bien réglés)

**Cahier des charges :**
- ✓ Compare vitesses sur polynôme complexe
- ✓ Vérifie convergence sur paysage non-trivial

---

## 4. Fonction de Rosenbrock

### rosenbrock_adam.png
**Résultat attendu :**
- Courbes de niveau en forme de BANANE (vallée incurvée)
- Vallée étroite qui suit y ≈ x²
- Trajectoire Adam (-1,1) → (1,1) suit relativement bien la vallée
- Moins de zigzags que les autres algorithmes
- Converge vers le centre bleu foncé en (1,1)

**Cahier des charges :**
- ✓ Fonction classique de test : ROSENBROCK
- ✓ Illustre les pièges : RAVINES (vallée étroite)
- ✓ Montre la supériorité d'Adam sur fonction difficile

---

### rosenbrock_comparison.png
**Résultat attendu :**
- Simple/Momentum/Nesterov (rouge/bleu/vert) : Trajectoires avec beaucoup d'oscillations en zigzag, tentent de suivre la vallée mais "tapent" les parois
- Ces 3 algorithmes atteignent probablement max_iter=2000 sans convergence complète
- Adam (orange) : Trajectoire beaucoup plus lisse, suit mieux la vallée
- ÉNORME différence de performance visible visuellement

**Cahier des charges :**
- ✓ Comparer les 4 algorithmes sur fonction DIFFICILE
- ✓ Illustre quand Simple/Momentum/Nesterov échouent ou sont très lents
- ✓ Cas où Adam est VRAIMENT supérieur (facteur 10x ou plus)

---

### rosenbrock_convergence.png
**Résultat attendu :**
- Simple/Momentum/Nesterov : Courbes qui descendent très lentement, atteignent ~10⁻² après 2000 itérations mais ne vont pas jusqu'à 0 (plateaux vers la fin)
- Adam : Descente rapide jusqu'à ~10⁻⁶ ou mieux
- Échelle log montre clairement l'écart de 3-4 ordres de grandeur
- C'est LA figure qui montre pourquoi Adam est utilisé en deep learning !

**Cahier des charges :**
- ✓ Comparer vitesses (Adam converge ~10x plus vite)
- ✓ Illustre les PLATEAUX (gradient très petit dans la vallée)
- ✓ Cas d'échec relatif (Simple ne converge pas complètement en 2000 itérations)

---

## 5. Fonction de Booth

### booth_comparison.png
**Résultat attendu :**
- Courbes de niveau elliptiques centrées en (1,3)
- 4 trajectoires de (0,0) vers (1,3), toutes convergent
- Trajectoires relativement directes (pas de ravine)
- Différences plus subtiles qu'avec Rosenbrock
- Adam reste le plus rapide mais tous arrivent au minimum

**Cahier des charges :**
- ✓ Fonction classique : BOOTH
- ✓ Cas favorable (tous les algos marchent bien)
- ✓ Sert de point de comparaison avec les fonctions difficiles

---

### booth_convergence.png
**Résultat attendu :**
- Toutes les courbes descendent rapidement vers 0
- Convergence en < 100 itérations pour tous
- Ordre : Adam légèrement meilleur, puis Nesterov, Momentum, Simple
- Différences moins marquées que Rosenbrock (fonction plus facile)

**Cahier des charges :**
- ✓ Comparer vitesses sur fonction facile
- ✓ Vérifier que tous les algos fonctionnent correctement

---

## 6. Fonction de Beale

### beale_comparison.png
**Résultat attendu :**
- Courbes de niveau avec gradients très forts près de l'origine
- Minimum en (3, 0.5)
- Simple/Momentum/Nesterov : Trajectoires hésitantes, convergence lente (2000 itér.)
- Adam : Convergence plus rapide grâce à l'adaptation du learning rate
- SI learning rate trop grand : divergence visible (trajectoire qui part hors limites)

**Cahier des charges :**
- ✓ Fonction classique : BEALE
- ✓ Illustre le rôle du LEARNING RATE (α=0.001 vs α=0.01)
- ✓ Problème des gradients à échelles différentes

---

### beale_convergence.png
**Résultat attendu :**
- Simple/Momentum/Nesterov : Convergence très lente (n'atteignent que ~10⁻² en 2000 itér.)
- Adam : Atteint ~10⁻⁶ ou mieux, beaucoup plus rapide
- Différence claire : Adam peut utiliser α=0.01 alors que les autres nécessitent α=0.001
- C'est un excellent exemple de l'avantage de l'adaptation automatique

**Cahier des charges :**
- ✓ Comparer vitesses avec learning rates différents
- ✓ Illustre quand l'adaptation automatique d'Adam fait vraiment la différence

---

## 7. Fonction de Himmelblau

### himmelblau_comparison.png
**Résultat attendu :**
- Paysage complexe avec 4 "trous" (minima) visibles
- Les 4 algorithmes partent de (0,0)
- ATTENTION : Ils peuvent converger vers des minima DIFFÉRENTS !
  - (3, 2) le plus probable depuis (0,0)
  - (-2.805, 3.131) possible
  - Les 2 autres moins probables depuis ce point
- Toutes les convergences sont valides (4 minima globaux équivalents)
- Illustre la non-unicité de la solution

**Cahier des charges :**
- ✓ Fonction classique : HIMMELBLAU
- ✓ Illustre les MINIMA LOCAUX (ici globaux multiples)
- ✓ Rôle du POINT INITIAL (détermine quel minimum on atteint)

---

### himmelblau_convergence.png
**Résultat attendu :**
- Toutes les courbes descendent rapidement vers 0
- Convergence en < 200 itérations pour tous
- Les courbes peuvent avoir des formes légèrement différentes selon le minimum atteint
- Si tous convergent vers le même minimum : courbes similaires
- Si vers des minima différents : début différent puis convergence à 0

**Cahier des charges :**
- ✓ Comparer vitesses sur fonction multi-minima
- ✓ Vérifier que tous atteignent f=0 (un des minima globaux)

---

## 8. Fonction d'Ackley

### ackley_adam.png
**Résultat attendu :**
- Paysage en "boîte à œufs" avec oscillations régulières
- Centre bleu foncé très petit en (0,0)
- Plein de petits creux partout (minima locaux)
- Trajectoire Adam de (2,2) vers (0,0) si succès
- Sinon, trajectoire vers un minimum local proche
- Pattern très différent des autres fonctions (multimodal extrême)

**Cahier des charges :**
- ✓ Fonction classique : ACKLEY
- ✓ Illustre les pièges : MINIMA LOCAUX (des centaines !)
- ✓ Test ultime de robustesse des algorithmes

---

### ackley_comparison.png
**Résultat attendu :**
- 4 trajectoires qui peuvent finir à des endroits DIFFÉRENTS
- Simple (rouge) : probablement bloqué dans un minimum local proche de (2,2)
- Momentum (bleu) : peut-être légèrement mieux, mais risque de local aussi
- Nesterov (vert) : chances améliorées de trouver le global
- Adam (orange) : meilleures chances d'atteindre (0,0)
- C'est LA figure qui montre la différence entre algorithmes sur fonction difficile !

**Cahier des charges :**
- ✓ Comparer les 4 algorithmes sur fonction TRÈS difficile
- ✓ Cas où Momentum et même Adam peuvent échouer (minima locaux)
- ✓ Illustre pourquoi l'optimisation globale est difficile

---

### ackley_convergence.png
**Résultat attendu :**
- Ceux qui trouvent le global : descendent jusqu'à f ≈ 0
- Ceux bloqués en local : se stabilisent à f ≈ 1-5 (selon le minimum local)
- Possibles oscillations (algorithme explore différents minima locaux)
- Adam devrait avoir la courbe la plus basse (ou égale si d'autres trouvent aussi)
- C'est clair visuellement qui a réussi vs échoué

**Cahier des charges :**
- ✓ Visualise succès vs échec (convergence vers 0 vs blocage à valeur > 0)
- ✓ Quantifie la difficulté : seuls les meilleurs algorithmes atteignent 0

---

## 9. Comparaison Dual Numbers vs Dérivée Numérique

### ackley_gradient_comparison.png
**Résultat attendu :**
- Deux graphes quasiment identiques (même pattern de couleurs)
- Les zones à fort gradient (jaune) et faible gradient (bleu) sont aux mêmes endroits
- Visuellement impossible de distinguer les deux méthodes
- Preuve visuelle que les deux méthodes donnent le même résultat

**Cahier des charges :**
- ✓ Comparer DUAL NUMBERS vs DÉRIVÉE NUMÉRIQUE
- ✓ Test sur fonction complexe (ACKLEY)
- ✓ Comparer précision et temps de calcul

---

## 10. Cas d'Échec

### Échec 1 : Learning Rate Trop Grand (Divergence)

#### echec1_lr_divergence.png
**Résultat attendu :**
- α=0.1 (vert) : Converge normalement vers (0,0)
- α=0.5 (orange) : Oscille un peu mais converge
- α=1.0 (rouge) : DIVERGE ! Part vers l'infini, trajectoire qui s'éloigne
- α=1.5 (rouge foncé) : EXPLOSE encore plus vite
- Message console montre clairement les divergences

**Cahier des charges :**
- ✓ Cas où ça NE MARCHE PAS (divergence)
- ✓ Décrire POURQUOI ça échoue (learning rate trop grand)
- ✓ Visualisation claire du problème

---

#### echec1_convergence.png
**Résultat attendu :**
- α=0.1 : Courbe qui DESCEND (bon comportement)
- α=0.5 : Quelques oscillations puis descend
- α=1.0 : Courbe qui MONTE ! (échec clair)
- α=1.5 : Monte encore plus vite
- En échelle log, on voit bien l'explosion exponentielle

---

### Échec 2 : Learning Rate Trop Petit (Stagnation)

#### echec2_lr_stagnation.png
**Résultat attendu :**
- 3 trajectoires de couleurs différentes depuis (-1,1)
- Plus le learning rate est petit, plus la trajectoire est "courte" (peu de progrès)
- α=0.00001 (jaune) : À peine bougé de (-1,1)
- α=0.0001 (orange) : A avancé un peu mais loin de (1,1)
- α=0.001 (rouge) : Meilleur progrès mais encore insuffisant
- AUCUN n'atteint l'étoile bleue en (1,1)

**Cahier des charges :**
- ✓ Cas où ça NE MARCHE PAS (trop lent)
- ✓ Illustre les PLATEAUX (Rosenbrock)
- ✓ Montre l'importance du choix du learning rate

---

#### echec2_convergence.png
**Résultat attendu :**
- 3 courbes qui descendent TRÈS lentement
- α=0.00001 : Presque plate, descente imperceptible
- α=0.0001 : Descend un peu mais reste haut
- α=0.001 : Meilleure descente mais loin de 0
- Toutes finissent loin de 10⁻⁶ (objectif normal)

---

### Échec 3 : Minimum Local (Piège d'Ackley)

#### echec3_minima_locaux.png
**Résultat attendu :**
- Point vert : Part de (1,1), ATTEINT l'étoile bleue (succès)
- Point orange : Part de (3,3), se BLOQUE dans un minimum local proche
- Point rouge : Part de (5,5), se BLOQUE encore plus loin
- Les points orange et rouge finissent dans des "trous" locaux, pas au centre
- Illustration parfaite du problème : point initial détermine le succès

**Cahier des charges :**
- ✓ Illustre les MINIMA LOCAUX (centaines dans Ackley)
- ✓ Rôle du POINT INITIAL (crucial !)
- ✓ Cas d'échec même avec bon algorithme

---

#### echec3_convergence.png
**Résultat attendu :**
- Courbe verte : Descend jusqu'à ~10⁻¹⁰ (sous le seuil bleu) ✅
- Courbe orange : Descend puis SE STABILISE à ~1-3 (au-dessus du seuil) ❌
- Courbe rouge : Descend puis SE STABILISE encore plus haut ❌
- C'est CLAIR visuellement qui a réussi vs échoué

---

### Échec 4 : Momentum Trop Élevé (Oscillations)

#### echec4_momentum_oscillations.png
**Résultat attendu :**
- β=0.5 (vert) : Trajectoire relativement lisse vers (0,0)
- β=0.9 (bleu) : Légèrement plus d'oscillations mais OK
- β=0.95 (orange) : BEAUCOUP d'oscillations, dépasse le minimum
- β=0.99 (rouge) : OSCILLATIONS EXTRÊMES, fait des grands allers-retours
- Plus le momentum est élevé, plus la trajectoire "serpente"

**Cahier des charges :**
- ✓ Rôle du paramètre MOMENTUM
- ✓ Cas d'échec : momentum trop grand → instabilité

---

#### echec4_convergence.png
**Résultat attendu :**
- β=0.5 : Descente lisse
- β=0.9 : Descente avec petites vagues
- β=0.95 : Descente avec GROSSES vagues (coût monte et descend)
- β=0.99 : Vagues ÉNORMES, peut même remonter temporairement
- En échelle log, on voit clairement les oscillations

---

### Échec 5 : Ravine Étroite (Zigzags Extrêmes)

#### echec5_zigzags_ravine.png
**Résultat attendu :**
- Ellipses TRÈS étirées (ratio 1:100)
- Simple (rouge) : ZIGZAGS marqués perpendiculaires à la direction du minimum
- Momentum (bleu) : Trajectoire BEAUCOUP plus lisse
- Les deux partent de (10,10) et visent (0,0)
- Différence spectaculaire : zigzags vs ligne relativement directe

**Cahier des charges :**
- ✓ Illustre les RAVINES (fonction mal conditionnée)
- ✓ Montre pourquoi Momentum améliore Simple
- ✓ Cas où Simple est très inefficace

---

#### echec5_convergence.png
**Résultat attendu :**
- Simple : Descente en "escalier" (oscillations)
- Momentum : Descente beaucoup plus lisse et rapide
- Momentum converge ~2-3x plus vite
