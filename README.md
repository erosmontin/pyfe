# pyFe
Features extraction and machine learning 

## Installation
if you only want to use only PYRAD

```
pip install git+https://www.github.com/erosmontin/pyfe
```

if you want to use also all the others classes, continue the installation by

1. compile our [Features Extractor](https://github.com/erosmontin/FeaturesExtractor)
1. copy bld directory in the library path



## Cite Us
1. Montin, E., Kijowski, R., Youm, T., & Lattanzi, R. (2024). Radiomics features outperform standard radiological measurements in detecting femoroacetabular impingement on three‐dimensional magnetic resonance imaging. In Journal of Orthopaedic Research. Wiley. https://doi.org/10.1002/jor.25952

1. Montin, E., Kijowski, R., Youm, T., & Lattanzi, R. (2023). A radiomics approach to the diagnosis of femoroacetabular impingement. In Frontiers in Radiology (Vol. 3). Frontiers Media SA. https://doi.org/10.3389/fradi.2023.1151258

1. Cavatorta, C., Meroni, S., Montin, E., Oprandi, M. C., Pecori, E., Lecchi, M., Diletto, B., Alessandro, O., Peruzzo, D., Biassoni, V., Schiavello, E., Bologna, M., Massimino, M., Poggi, G., Mainardi, L., Arrigoni, F., Spreafico, F., Verderio, P., Pignoli, E., & Gandola, L. (2021). Retrospective study of late radiation-induced damages after focal radiotherapy for childhood brain tumors. In S. D. Ginsberg (Ed.), PLOS ONE (Vol. 16, Issue 2, p. e0247748). Public Library of Science (PLoS). https://doi.org/10.1371/journal.pone.0247748

1. Montin, E., Belfatto, A., Bologna, M., Meroni, S., Cavatorta, C., Pecori, E., Diletto, B., Massimino, M., Oprandi, M. C., Poggi, G., Arrigoni, F., Peruzzo, D., Pignoli, E., Gandola, L., Cerveri, P., & Mainardi, L. (2020). A multi-metric registration strategy for the alignment of longitudinal brain images in pediatric oncology. In Medical &amp; Biological Engineering &amp; Computing (Vol. 58, Issue 4, pp. 843–855). Springer Science and Business Media LLC. https://doi.org/10.1007/s11517-019-02109-4


## Change log:
1. version 0.1.17 **2024-07-23**
    - normalize options for pyrad and correct binning
1. version 0.1.03 **2023-10-13**
    - pyradiomic PYRAD
1. version 0.1.02 **2023-10-13**
    - Benford Law FE {"type":"BENFORD","options":{"min":0,"max":5000,"bin":128},"name":"BENFORD"}
1. version 0.1.00 **2023-09-20**
    - add the group name in the group = {"type":"FOS","options":{"min":0,"max":5000,"bin":128},**"name":"FOS32"**}
    - change the "name" field in the groups to type and that can be FOS,GLCM,GLRLM or SS {**"type":"FOS"**,"options":{"min":0,"max":5000,"bin":128},"name":"FOS32"}
    - multiprocess of pyml and geenral debug of the classification accuracy function
    - added utils.py, where there's a new class to customize the extraction:
1. vesion 0.0.0 **2022-08-01**
    - main class for exrtaction and machine learning for classification



[*Dr. Eros Montin, PhD*](http://me.biodimensional.com)
**46&2 just ahead of me!**

