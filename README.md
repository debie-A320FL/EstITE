# EstITE: Estimation of Individual Treatment Effects with recent popular models

Ce repo contient est un fork contenant le code implémentant les modules du papier [Caron et. al. (2020)](https://arxiv.org/pdf/2009.06472.pdf), afin d'estimer les effets de traitement.

## Papier

Les dossiers *Illustrative Examples*, *Real-World Example* et *Simulations* proviennent du fork original. Des modifications mineures par rapport au repo original peuvent exister (chemin d'accès par exemple).

## Stage

Le dossier *Simulations_Stage* comprend les différentes simulations de ma part, dans le but d'appliquer les méthodes dans un cadre proche de celui étudié en pratique dans le projet Graph4Health.

Les différents scénario étudiés dans les simulations sont isolés dans des dossiers distincts *Setup X* dont une description est disponible ci-dessous.

On retrouve les codes dans le dossier *Code*, partagé entre *R* et *Python*. Se trouve également le dossier *Data* dans lequel figure les fichiers de données (micro données ainsi que paramètres ayant générés le fichier). Le dossier *Result* contient les métriques des codes R et Python, et *Figure* le résumé graphique de ces derniers.

De manière générale, les données simulées proviennent du fichier *Sample_creation.R* ou issu d'un scénario précédant. Des statistiques sur les données sont disponibles dans le ficher *StatisticsOnData.Rmd*.


Les méthodes sont exécutées pour la plupart dans *Simulations.R* puisque c'est dans ce langage que les méthodes sont implémentées. Néanmoins les M-Learners (GP) sont implémentées en Python (*causal_model.py*). Leur simulations sont alors exécutées en Python dans le fichier *Simulation.py*. C'est également dans ce fichier que l'on retrouve la méthode logistique.

Les résultats sont ensuite analysées de manière chiffrée dans *Results_analysis.R* ou bien exporté en figure à l'aide du fichier *PEHE_plots.R*.

### Elements communs à travers les scénarios

#### Caractéristiques (X,Z)

- **X : caractéristiques d'une personne**
  - âge $\sim \mathcal{N}(48, 6)$
  - genre $\sim \mathcal{B}(0.5)$
  - poids $\sim \mathcal{N}(80, 8)$ si genre = 0, poids $\sim \mathcal{N}(65, 6)$ sinon
  - comorbidités $\sim \mathcal{B}(0.3)$

- **Traitement (avoir un médecin traitant)**
  - $\mathbb{P}(Z = 1 \mid X) = \text{logit}^{-1}(\beta_0 + \beta_1 \times \text{âge} + \beta_2 \times \text{poids} + \beta_3 \times \text{comorbidités} + \beta_4 \times \text{genre})$

#### Variable cible Y

- **Fonction de base** :
  $\mu_0(X) = \gamma_0 + \gamma_1 \times \text{âge} + \gamma_2 \times \text{poids} + \gamma_3 \times \text{comorbidités} + \gamma_4 \times \text{genre}$

- **Effet causal moyen conditionnel "pseudo" (CATE)** :
  $\tau(X) = \delta_0 + \delta_1 \times \text{âge} + \delta_2 \times \text{poids} + \delta_3 \times \text{comorbidités} + \delta_4 \times \text{genre}$

- **Issue Y** :
  $Y \sim \mathcal{B}(p_i) \quad \text{où} \quad p_i = \text{logit}^{-1}(\mu_0(X) + Z \times \tau(X) + \epsilon) \quad \text{et} \quad \epsilon \sim \mathcal{N}(0, \sigma)$

### Setup 1a

Les variables $\beta$, $\delta$ et $\gamma$ sont considérés fixés à travers les scénarios, sauf mention contraire.

Dans notre configuration, 65 % des lignes ont $Z=1$

On travaille ici avec le $Y$ binaire $Y \sim \mathcal{B}(p_i)$.


### Setup 1b

Ce scénario est identique est précédant, mais dans ce dernier l'effet du traitement est fixé à 0. Ainsi $\delta$ est fixé à 0.

### Setup 2a

Ce scénario reprend le scénario 1a. Néanmoins on considère ici un $Y$ continue valant la probabilité de valoir 1 et non pas le résultat d'une Bernoulli.

### Setup 2b

Ce scénario reprend le scénario 1b. Néanmoins on considère ici un $Y$ continue valant la probabilité de valoir 1 et non pas le résultat d'une Bernoulli.

### Setup 3

### Setup 4

