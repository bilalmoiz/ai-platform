# Image Classification
## Pneumonia Chest X-Ray Image Classifier

##### Premise:
This simple classifier is trained to distinguish between chest xrays of patients with pneumonia
versus those who do not have pneumonia.


##### Data:
Data is available from Kaggle: https://kaggle.com/paultimothymooney/chest-xray-pneumonia


To train model with default values, simply run with 
```
python pneumonia_model.py
```

To train model with adjusted values, run with custom numeric values for batch_size, img_size, epoch, and nodes in the format: python pneumonia_model.py <batch_size> <img_size> <epoch> <nodes>

##### Example: 
```
python pneumonia_model.py 32 64 20 120
```

## MLFlow

After running models, use the following commands to analyze the results:
```
mlflow ui
```

Navigate to:
```
http://localhost:5000/#/
```
