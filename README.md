# human-centeredXAI

Dieses Repository beinhaltet den Code für die Implementierung benutzerzentrierter Visualisierungen. Es ist ein Bestandteil der Bachelorarbeit mit dem Titel "Visualisierungsmöglichkeiten für bildbasierte
Erklärbare Maschinelle Intelligenz in der digitalen
Pathologie".

**Architektur**

Das System basiert auf den drei Hauptklassen 'ModelHandler', 'ImageHandler' und 'ExplanationHandler', die für das Laden von Modellen, die Bildverarbeitung und das Erklären von Vorhersagen verantwortlich sind.


**Verwendung**

* Vorverarbeitung: Konvertierung eines Bildes in einen PyTorch Tensor mit der Methode preprocess_image.
* Vorhersage: Verwendung von predict_label für die Vorhersage und Rückgabe des wahrscheinlichsten Labels.
* Erklärungen: Anwendung des Layer-Wise Relevance Propagation (LRP) Modells mit der Methode explain, um die Entscheidungen des Modells zu erklären.
* Visualisierungen: Einsatz verschiedener Methoden wie heatmap, contour_relevance_map, cluster_relevance_map und overlay_clustered_on_grayscale zur Darstellung der Ergebnisse.


**Dateistruktur**

* ModelHandler.py: Verwaltung des VGG19-Modells.
* ImageHandler.py: Bildvorverarbeitung.
* ExplanationHandler.py: Erstellung von Erklärungen und Visualisierungen.
* data/imagenet_labels.txt: Label-Zuordnungen für das VGG19-Modell.
