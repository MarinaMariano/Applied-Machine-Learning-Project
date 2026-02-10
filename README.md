# A COMPARATIVE STUDY OF COSTUM AND PRETRAINED CNNs FOR CHEST X-RAY ClASSIFICATION


## INTRODUCTION 

This project was initially designed to evaluate the same CNN architecture on two different medical imaging datasets, namely Optical Coherence Tomography (OCT) and Chest X-Ray dataset, following the lightweight model proposed by Yen & Tsao (2024), (Yen, C.-T., & Tsao, C.-Y. (2024). Lightweight convolutional neural network for chest X-ray images classification. Scientific Reports, 14, 29759. https://doi.org/10.1038/s41598-024-80826-z). The first one is a large multi-class dataset with 109,309 images, while the latter is a smaller dataset dividend into two classes and made of 5856 images.

The original idea was to keep the model architecture fixed and evaluate it on two different datasets: the Yen & Tsao CNN would have been applied to the Chest X-Ray dataset and, in parallel, to the much larger OCT dataset.
However, the OCT dataset scale and I/O overhead made an end-to-end training computationally impractical: despite multiple optimizations, model depth reduction, lower image resolution, reduced batch size and epochs, removal of expensive feature modules, and local dataset staging on Colab to avoid remote disk access training, the training time remained prohibitive (exceeding one hour per epoch).

To overcome these constraints while preserving a meaningful comparison, the project design shifted to a **comparison on the Chest X-ray dataset** only, **evaluating**:
- the lightweight **custom architecture** derived from Yen & Tsao (2024) (implemented by Luca), **and**
- the **pretrained** DenseNet-121 from TorchXRayVision with weights="densenet121-res224-all" (implemented by Marina).

A pretrained model is a neural network whose weights have already been optimized on large-scale datasets, providing a strong initialization for downstream tasks. Specifically, we used a pretrained DenseNet backbone and re-purpose it for **binary classification**, producing a discrete label **0/1** for the two target classes (**PNEUMONIA vs NORMAL**).
The transfer-learning pipeline consists of loading the pretrained feature extractor and replacing the original classifier with a task-specific head (binary output).
The backbone is kept fixed (non-trainable) and only the new classification head is optimized.

----

## Project Pipeline


## **Dataset organization**

The Chest X-ray dataset is a folder that contains two subdirectories, train and test and inside each split, images are organized into class-specific folders named `NORMAL` and `PNEUMONIA`. In total, the dataset contains 5,856 images.

### **Train / Validation / Test split**
The dataset is divided into training, validation and test sets. The validation set is obtained by the initial training set that was randomly shuffled and split using an 80/20 ratio to avoid data leakage. The test set is used exclusively for final model evaluation.

### X-Ray Dataset Split (Luca – Custom CNN, Yen & Tsao)                                         

| Dataset      | Number of Images | Classes |
|--------------|------------------|---------|
| Training     | 4,186            | 2       |
| Validation   | 1,046            | 2       |
| Test         | 624              | 2       |
| **Total**    | **5,856**        | 2       |

### X-Ray Dataset Split (Marina – Pretrained DenseNet-121)

| Dataset      | Number of Images | Classes |
|--------------|------------------|---------|
| Training     | 4,204            | 2       |
| Validation   | 1,050            | 2       |
| Test         | 624              | 2       |
| **Total**    | **5,878**        | 2       |

----

## **Image Preprocessing – Pretrained DenseNet-121**

For the pretrained DenseNet-121 model, image preprocessing follows the standardized pipeline provided by TorchXRayVision, ensuring compatibility with the pretrained weights. Images are converted to grayscale and transformed into tensors with shape [1, H, W]. Pixel values are rescaled to [0, 255] and normalized using xrv.datasets.normalize with a maximum value of 255, producing intensity values in the range **[-1024, 1024]. A **center crop** (XRayCenterCrop) is applied, followed by resizing to **224 × 224 pixels** (XRayResizer). The final output is a float tensor with shape [1, 224, 224]`.
The same preprocessing pipeline is applied to both training and test images.

## **Image Preprocessing – Custom CNN (Yen & Tsao)**
This section reports the preprocessing steps used for this model.


### **Data augmentation applied just for the Custom CNN, Yen & Tsao**


----

## **Model architecture**
### **First model: Custom CNN, Yen & Tsao**
We adopted the same model architecture based on a lightweight convolutional neural 
network inspired by the one proposed by Yen and Tsao (2024), specifcillay designed for chest X-ray classification (which consisted of a redesigned feature extraction (FE) module and multiscale feature (MF) module and validated using publicly available COVID-19 datasets).
The proposed CNN architecture is composed, indeed, of two main feature extraction stages, followed by a classification head. Scrivi qualcosa in aggiunta e che sia specifico.

#### 1. Feature Extraction (FE Module)

The Feature Extraction (FE) module is designed to efficiently extract local spatial features while reducing redundancy. Structure:

- **1×1 Convolution**
- **Channel Split** (A portion of channels is kept unchanged,The remaining channels undergo further processing)
- **Depthwise 3×3 Convolution** (Captures local spatial patterns, Lightweight compared to standard convolutions)
- **1×1 Convolution** (Recombines channel information)
- **Concatenation** (Merges processed and unprocessed channel branches)
- **Residual Connection**

#### 2. Multi-scale Feature Module (MF Module)

The Multi-scale Feature (MF) module captures contextual information at different spatial scales. Structure:

- Parallel Max Pooling branches with different receptive fields, pool sizes: 2×2, 4×4, 8×8
- Depthwise Dilated Convolutions, Different dilation rates per branch Capture both local and global context
- 1×1 Convolutions
- Align channel dimensions
-Concatenation Combines multi-scale representations
-Final 1×1 Convolution

#### **3. Classification Head**

After feature extraction, the network uses a lightweight classification head:

- Global Average Pooling Reduces spatial dimensions Prevents overfitting
- Fully Connected Layer (128 units, ReLU)
- Dropout (0.5)
- Regularization
- Output Layer: 1 neuron with Sigmoid activation
- Binary classification

Reference: Yen, C.-T., & Tsao, C.-Y. (2024). Lightweight convolutional neural network for chest X-ray images classification. Scientific Reports, 14, 29759. https://doi.org/10.1038/s41598-024-80826-z


### **Second model: Pretrained DenseNet-121**
We instantiated a DenseNet-121 architecture from the TorchXRayVision (xrv) library. The weights "densenet121-res224-all" indicate pretrained on large-scale chest X-ray datasets, trained with 224×224 input resolution. This backbone acts as a feature extractor, not a classifier. We move the backbone’s parameters and buffers to the selected compute device so that input tensors and model weights are on the same device. We switch the backbone to evaluation mode disableing batch normalization updates, dropout randomness and gradient computation for all backbone parameters. Since the backbone is frozen, we want its behavior to remain deterministic and identical to pretraining, letting it acts as a fixed feature extractor. Then we added a task specific binary classification head.

----

## **Training strategy**

### **Training of the Pretrained DenseNet-121**
Class imbalance was handled via BCEWithLogitsLoss(pos_weight), computed from the training subset label counts; in our split 
Nneg​/Npos​<1, indicating a relative over-representation of positive samples. In this setting, the loss weighting counterbalances the natural bias of the optimization process toward the majority class by increasing the penalty associated with misclassified negative examples, thus preventing the classifier from trivially favoring the positive class. Conversely, in the more common case Nneg​/Npos​ >1, the same strategy would up-weight positive samples to mitigate majority-negative dominance. Optimization uses Adam (lr = 1e-3) on the classifier head (model.fc) only, keeping the pretrained DenseNet backbone frozen. The learning-rate scheduler (ReduceLROnPlateau) and early stopping monitor the validation loss, and the final checkpoint corresponds to the epoch with the lowest val_loss. Performance is also reported with MCC computed from epoch-level TP/TN/FP/FN counts at a fixed threshold of 0.5.

### **Training of the Custom CNN, Yen & Tsao**
Supervised training is performed using the training set and monitored on the validation set. Early Stopping is applied to prevent overfitting.
Training Configuration
- Optimizer: Adam
- Loss: Binary Crossentropy
- Batch size: 16
- Epochs : da definire
- Early stopping (patience: 5)

### **Final evaluation**

### **Of the Pretrained DenseNet-121**
On the test set, the model achieved strong performance, with an F1-score of 0.949 and an MCC of 0.861. The confusion matrix shows a limited number of misclassifications, with 36 false positives and only 5 false negatives. The decision threshold was selected on the validation set to maximize the F1-score, favoring sensitivity to pneumonia cases. In this setting, accepting a higher number of false positives while keeping false negatives low can considered a more precautionary and clinically safer approach. 
Threshold-free metrics (AUROC and AUPRC) are first computed from predicted probabilities to assess ranking performance. Error analysis is further supported by explicitly identifying false positives and false negatives at the image level. Results show strong discriminative performance (AUROC 0.985, AUPRC 0.990) and a recall-oriented behavior for the PNEUMONIA class (recall 0.987), minimizing false negatives.


| Metric | Value |
|--------|-------|
| F1-score | 0.955 |
| MCC | 0.877 |


### **Of the Custom CNN, Yen & Tsao**

### **Reproducibility**

Modular and version-controlled codebase suitable for GitHub. The dataset is excluded from the repository using `.gitignore`.

