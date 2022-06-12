# linear-regression

Une introduction au machine learning avec un algo de régression linéaire

#### Linear regression

Implémentez un algorithme de regression linéaire sur un seul element, en l’occurence
le kilométrage d’une voiture Pour ce faire vous devez faire 2 programmes :

        • Le premier programme sera utilisé pour prédire le prix d’une voiture en fonction
        de son kilométrage. Quand vous lancerez le programme, celui ci vous demandera le
        kilométrage et devrait vous donner un prix approximatif de la voiture en utilisant
        l’hypothèse suivante :
        prixEstime(kilométrage) = θ0 + (θ1 ∗ kilométrage)
        Avant de lancer le programme d’entrainement, theta0 et theta1 auront pour valeur
        0.
        • Le second programme sera utilisé pour entrainer votre modèle. Il lira le jeu de
        données et fera une regression linéaire sur ces données.
        Une fois la regression linéaire terminée, vous sauvegarderez la valeur de theta0 et
        theta1 pour pouvoir l’utiliser dans le premier programme.
        Vous utiliserez la formule suivante :
        tmpθ0 = ratioDApprentissage ∗ 1/m mX−1 i=0 (prixEstime(kilométrage[i]) − prix[i])
        tmpθ1 = ratioDApprentissage ∗ 1/m mX−1 i=0 (prixEstime(kilométrage[i])−prix[i])∗kilométrage[i]

        Veuillez noter que le prixEstime est la même chose que dans notre premier programme, mais ici il utilise vos valeures temporaires afin de calculer theta0 et theta1.
        Attention a bien mettre a jour theta0 et theta1 en même temps.

#### Bonus

- expliquer l'overfitting
- Visualiser les données sur un graph avec leur repartition
- Afficher la ligne résultant de votre regression linéaire sur ce même graphe et voir si ca marche !
- Un programme qui vérifie la precision de votre algorithme
- Un programme qui permet de visualiser la précision de votre algorithme sur le graphe

#### a faire

- documenter le code
