# EstITE: Estimation of Individual Treatment Effects with recent popular models

Ce repo est un fork contenant le code implémentant les modules du papier [Caron et. al. (2020)](https://arxiv.org/abs/2009.06472.pdf), afin d'estimer les effets de traitement.

## Papier

Les dossiers *Illustrative Examples*, *Real-World Example* et *Simulations* proviennent du fork original. Des modifications mineures par rapport au repo original peuvent exister (chemin d'accès par exemple).

## Stage

Le dossier *Simulations_Stage* comprend les différentes simulations de ma part, dans le but d'appliquer les méthodes dans un cadre proche de celui étudié en pratique dans le projet Graph4Health.

Les différents scénarios étudiés dans les simulations sont isolés dans des dossiers distincts *Setup X*. Les résultats ont été exportés sur un S3, ils ne sont donc pas disponibles sur le Github.

Les scénarios 1a/1b, 2a/2b sont dans le prolongement des travaux du papier original. Ce sont les premières simulations où les méthodes du papier sont utilisées sur un autre jeu de données. Il s'agit d'étudier l'estimation du CATE lorsque celui-ci est nul ou non, et lorsque la variable cible est réelle ou binaire. Il s'agit de simulations préliminaires pour me familiariser avec le sujet, qui ne sont pas présentées dans mon rapport.

Le scénario 4 vise à étudier les performances des Meta-Learners lorsque la taille de l'échantillon varie. C'est à partir de ce scénario que les hyperparamètres sont optimisés (contrairement au papier lequel garde les paramètres par défaut). Encore une fois, il s'agit de simulations préliminaires pour me familiariser avec le sujet, qui ne sont pas présentées dans mon rapport.

Le scénario 5 vise à étudier les performances de Meta-Learner à grande taille d'échantillon (N=1E5), et lorsque la proportion de non traités varie.Les scénarios 5b/5c reprennent le scénario 5, en l'adaptant plus spécifiquement à un base learner de type LASSO, ou bien de type Random Forest. Cela vise à comprendre l'impact du base learner sur les Meta-Learner. A noter que l'optimisation des méthodes RF prend du temps, ainsi des techniques tels que le DR-RF ont eu une implémentation mais n'ont pas été étudiées en détail.

Les scénarios suivants concernent la combinaison d'un réseau de neurones avec un Meta-Learner. Le scénario 6b est le prolongement de [Curth et van der Schaar (2021)](https://arxiv.org/abs/2101.10943). Il s'agit d'utiliser un réseau de neurones (très simple), de le combiner à des Meta-Learners, et de l'utiliser dans des scénarios inspirés par le papier, mais en grande dimension et avec un échantillon de grande taille.
Le fichier *draft_neural_network.py* est un fichier jouet pour se familiariser avec l'entraînement des réseaux de neurones. Le fichier *nn_learner_draft.py* implémente différents learners. Les simulations sont dans *Simulation.py*.

Le scénario 7 utilise les mêmes implémentations que setup 6b mais l'applique à un nouveau cas. Il s'agit cette fois de considérer un mélange gaussien dans lequel le groupe détermine à la fois le CATE et la distribution de X, et où l'on essaie d'estimer le CATE à partir de X (qui est une conséquence du groupe).

## Suite des travaux

Les scénarios précédents ont permis d'observer de premiers résultats empiriques grâce à des algorithmes de base très simples. Les codes précédents peuvent donc être utilisés pour confirmer les conclusions du rapport, néanmoins ces algorithmes simplistes n'ont pas vocation à être utilisés dans le projet final.

Comme indiqué dans le rapport, une suite pertinente des simulations pourrait être d'étudier des architectures de réseaux de neurones plus complexes, telles que le SNet (voir  [Curth et van der Schaar (2021a)](https://arxiv.org/abs/2101.10943)) ou bien le FlexTENet (voir [Curth et van der Schaar  (2021b)](https://arxiv.org/abs/2106.03765)).

Les implémentations de ces papiers sont disponibles sur Github, notamment dans [ce repo](https://github.com/AliciaCurth/CATENets).

Une première étape pourrait être de se familiariser avec l'idée globale derrière chacun de ces algorithmes (voir les papiers correspondant). Ensuite, de premières simulations et l'optimisation de leurs nombreux hyperparamètres devraient constituer un sujet majeur. Enfin, il faudrait étudier si dans la forme finale envisagée pour le projet (bonne "taille" du modèle etc via ses paramètres), la performance d'estimation "plug-in" peut être améliorée via l'utilisation d'un X-Learner (Meta-Learner choisi suite à la conclusion issue des travaux précédents). À Noter que comme indiqué dans le rapport, le X-Learner demanderait également un choix d'algorithme une fois les contrefactuels créés via une des architectures précédentes.

