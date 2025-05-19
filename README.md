# EstITE: Estimation of Individual Treatment Effects with recent popular models

Ce repo contient est un fork contenant le code implémentant les modules du papier [Caron et. al. (2020)](https://arxiv.org/pdf/2009.06472.pdf), afin d'estimer les effets de traitement.

## Papier

Les dossiers *Illustrative Examples*, *Real-World Example* et *Simulations* proviennent du fork original. Des modifications mineures par rapport au repo original peuvent exister (chemin d'accès par exemple).

## Stage

Le dossier *Simulations_Stage* comprend les différentes simulations de ma part, dans le but d'appliquer les méthodes dans un cadre proche de celui étudié en pratique dans le projet Graph4Health.

Les différents scénario étudiés dans les simulations sont isolés dans des dossiers distincts *Setup X* dont une description est disponible ci-dessous.

On retrouve les codes dans le dossier *Code*, partagé entre *R* et *Python*. Se trouve également le dossier *Data* dans lequel figure les fichiers de données (micro données ainsi que paramètres ayant générés le fichier). Le dossier *Result* contient les metriques des codes R et Python, et *Figure* le résumé graphique de ces derniers.

De manière générale, les données simulées proviennent du fichier *Sample_creation.R* ou issu d'un scénario précédant. Des statistiques sur les données sont disponibles dans le ficher *StatisticsOnData.Rmd*.


Les méthodes sont éxécutées pour la plupart dans *Simulations.R* puisque c'est dans ce langage que les méthodes sont implémentées. Néanmoins les M-Learners (GP) sont implémentées en Python (*causal_model.py*). Leur simulations sont alors éxécutées en Python dans le fichier *Simulation.py*. C'est également dans ce fichier que l'on retrouve la méthode logistique.

Les résultats sont ensuite analysées de manière chiffrée dans *Results_analysis.R* ou bien exporté en figure à l'aide du fichier *PEHE_plots.R*.

### Setup 1a

### Setup 1b

### Setup 2a

### Setup 2b

### Setup 3

### Setup 4

