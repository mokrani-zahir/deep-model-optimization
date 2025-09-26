# Optimisation d'un model avec OpenVINO

Ce projet démontre comment optimiser un modèle existant déjà pré-entraîné sans modification de l’architecture ni quantization.

Nous utilisons ici CRAFT, un modèle de détection de texte.
Le code de base provient du dépôt craft-pytorch
, que nous avons adapté afin de supporter les versions récentes de PyTorch.

L’objectif est d’optimiser le modèle pour un déploiement efficace sur matériel Intel (CPU, GPU, VPU) grâce à OpenVINO.

**Note importante :** 
le but n’est pas de traiter un lot ou plusieurs fichiers en parallèle.
Je me limite volontairement à un seul fichier pour évaluer si la conversion PyTorch → ONNX → OpenVINO apporte un gain réel en termes de vitesse et de précision.

Cependant, il est important de noter que l’utilisation d’OpenVINO avec un traitement par lot permet d’obtenir un gain de temps beaucoup plus significatif.

Des tests avec plusieurs fichiers seront réalisés plus tard.

## Pipeline

PyTorch (modèle .pth) → ONNX → ONNX Slim → OpenVINO → Tests & Résultats

## Environnement de test

Les tests ont été réalisés dans deux environnements distincts afin de comparer les résultats sur différents matériels :

- **Google Colab** (CPU générique fourni par Google)
- **PC personnel** : Intel® Core™ i7-8565U (8ᵉ génération, 4 cœurs / 8 threads)

## Résultats prévisionnels

⚠️ Pour l’instant, les tests sont réalisés sur **une seule image**.  
Les tests avec traitement par lot (batch) seront effectués plus tard.

| Modèle        | Temps d’inférence (1 image) | Temps d’inférence (lot / batch)  | Remarques |
|---------------|-----------------------------|----------------------------------|-----------|
| PyTorch       | 120 ms                      | Effectués plus tard              | Modèle original |
| ONNX (brut)   | 95 ms                       | Effectués plus tard              | Conversion directe |
| ONNX Slim     | 80 ms                       | Effectués plus tard              | Structure simplifiée |
| OpenVINO FP32 | 60 ms                       | Effectués plus tard              | Optimisation CPU Intel |
| OpenVINO FP16 | 45 ms                       | Effectués plus tard              | Gain notable |

Vous pouvez reproduire les résultats directement sur Colab grâce au notebook :

[➡️ Open in Deepnote](https://deepnote.com/workspace/deep-model-optimization-154223c9-4a76-4983-91cd-281475179f1c/project/Zahir-MOKRANIs-Untitled-project-a68662c9-ece5-4640-8a37-7140181a55cd/notebook/Notebook-1-c0deb2558b2845918a8f2f75cb6f873d?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=a68662c9-ece5-4640-8a37-7140181a55cd)