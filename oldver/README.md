## char_probabilities.txt:
  À partir de 2 grands textes Français, note la probabilité d'apparition d'un caractère.
  Ex.: e: 0.12049907181950065 implique que 12.05% des caractères du texte sont des "e"

## pair_probabilities.txt:
  À partir de 2 grands textes Français, note la probabilité d'apparition de chaque paire de caractère.
  Ex.: es: 0.01884327163261385 implique que 1.88% des paires de caractères du texte sont "es"

## encoded_text.txt:
  Texte chiffré "C" obtenu avec url = "https://www.gutenberg.org/cache/epub/42131/pg42131.txt"

## liste.de.mots.francais.frgut.txt:
  Dictionnaire Français. Ne contient aucun nom propre (majuscules)

## probabilities.txt:
  Une clé secrète générée pour un certain long texte. 
  Note les caractères de substitution et la fréquence d'apparition du symbole de 8 bits correspondant dans le texte chiffré.
  Ex: me: 0.008366081009837389 implique que 0.8366% des bytes du message chiffré représentent "me".

## probabilities1.txt:
  Même chose que probabilities.txt mais généré à partir d'un texte différent.

## texte.txt:
  Un exemple de livre Français.

## q2.py: 
Première tentative d'algorithme utilisant char_probabilities et pair_probabilities (probabilités séparées)

## q2-v2.py:
  Deuxième tentative d'algorithme utilisant probabilities.txt (probabilités combinées)




