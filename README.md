# EstITE: Estimation of Individual Treatment Effects with recent popular models

Ce repo contient est un fork contenant le code implémentant les modules du papier [Caron et. al. (2020)](https://arxiv.org/pdf/2009.06472.pdf), afin d'estimer les effets de traitement.

## Papier

Les dossiers *Illustrative Examples*, *Real-World Example* et *Simulations* proviennent du fork original. Des modifications mineures par rapport au repo original peuvent exister (chemin d'accès par exemple).

## Stage

Le dossier *Simulations_Stage* comprend les différentes simulations de ma part, dans le but d'appliquer les méthodes dans un cadre proche de celui étudié en pratique dans le projet Graph4Health.

Les différents scénario étudiés dans les simulations sont isolés dans des dossiers distincts *Setup X*. Les résultats ont été exporté sur un S3, ils ne sont donc pas disponible sur le Github.

Les scénarios 1a/1b, 2a/2b sont dans le prolongement des travaux du papier original. Ce sont les premières simulations où les méthodes du papier sont utilisé sur un autre jeu de données. Il s'agit d'étudier l'estimation du CATE lorsque celui-ci est nul ou non, et lorsque la variable cible est réelle ou binaire. Il s'agit de simulations préliminaires pour me familiariser avec le sujet, qui ne sont pas présentées dans mon rapport.

Le scénario 4 vise à étudier les performances des Meta-Learners lorsque la taille de l'échantillon varie. C'est à partir de ce scénario que les hyperparamètres sont optimisés (contrairement au papier lequel garde les paramètres par défaut). Encore une fois, il s'agit de simulations préliminaires pour me familiariser avec le sujet, qui ne sont pas présentées dans mon rapport.

Le scénario 5 vise à étudier les performances de Meta-Learner à grande taille d'échantillon (N=1E5), et lorsque la proportion de non traités varie.Les scénario 5b/5c reprennent la scénario 5, en l'adaptant plus spécifiquement à un base learner de type LASSO, ou bien de type Random Forest. Cela vise à comprendre l'impact du base learner sur les Meta-Learner. A noter que l'optimisation des méthodes RF prend du temps, ainsi des techniques tels que le DR-RF ont eu une implémentation mais n'ont pas été étudiées en détail.

Les scénarios suivant concernent la combinaison d'un réseau de neurones avec un Meta-Learner. Le scénario 6b est le prolongement de [Curth et van der Schaar (2021)](https://arxiv.org/pdf/2101.10943). Il s'agit d'utiliser un réseau de neurones (très simple) de le combiner à des Meta-Learner, et de l'utiliser dans des scénarios inspiré par le papier, mais à grande dimension et échantillon de grande taille.
Le fichier *draft_neural_network.py* est un fichier jouet pour se familiariser avec l'entraînement des réseaux de neurones. La fichier *nn_learner_draft.py* implémente différent learner. Les simulations sont dans *Simulation.py*.

Le scénario 7 utilise les mêmes implémentation que setup 6b mais l'applique à un nouveau cas. Il s'agit cette fois de considérer un mélange gaussien dans lequel le groupe déterminerait à la fois le CATE et la distribution de X, et où on essayerait d'estimer le CATE à partir de X (qui est une conséquence du groupe).

## Suite des travaux