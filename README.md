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

For the pretrained DenseNet-121 model, image preprocessing follows the standardized pipeline provided by TorchXRayVision, ensuring compatibility with the pretrained weights. Images are converted to grayscale and transformed into tensors with shape [1, H, W]. Pixel values are rescaled to [0, 255] and normalized using xrv.datasets.normalize with a maximum value of 255, producing intensity values in the range **[-1024, 1024]**. A **center crop** (XRayCenterCrop) is applied, followed by resizing to **224 × 224 pixels** (XRayResizer). The final output is a float tensor with shape [1, 224, 224]`.
The same preprocessing pipeline is applied to both training and test images.

## **Image Preprocessing – Custom CNN (Yen & Tsao Inspired)**

Data Preprocessing and Augmentation

All chest X-ray images were resized to 128 × 128 pixels and normalized to the range [0,1] through rescaling (pixel value / 255). Data augmentation was applied exclusively during training in order to improve generalization and reduce overfitting. The augmentation pipeline included:

- Random horizontal flipping
- Small random rotations (±5%)
- Mild random zooming (±5%)

No augmentation was applied to validation or test datasets. Given the inherent class imbalance of the chest X-ray dataset, class weights were computed and applied during training to balance the contribution of minority and majority classes in the loss function.


----

### **First model: Pretrained DenseNet-121**
Transfer learning formalizes a two-phase learning framework: a pre-training phase to capture knowledge from one or more source tasks, and a fine-tuning stage to transfer the captured knowledge to target tasks. 
We instantiated a DenseNet-121 architecture from the TorchXRayVision (xrv) library. The weights "densenet121-res224-all" indicate pretraining on large-scale chest X-ray datasets, trained with 224×224 input resolution. This backbone acts as a feature extractor, not a classifier. We move the backbone’s parameters and buffers to the selected compute device so that input tensors and model weights are on the same device. We switch the backbone to evaluation mode disableing batch normalization updates, dropout randomness and gradient computation for all backbone parameters. Since the backbone is frozen it acts as a fixed feature extractor. Then we added a task specific binary classification head.


## **Training strategy**

### **Training of the Pretrained DenseNet-121**
Class imbalance was handled via BCEWithLogitsLoss(pos_weight), computed from the training subset label counts; in our split 
Nneg​/Npos​<1, indicating a relative over-representation of positive samples. In this setting, the loss weighting counterbalances the natural bias of the optimization process toward the majority class by increasing the penalty associated with misclassified negative examples, thus preventing the classifier from trivially favoring the positive class. Conversely, in the more common case Nneg​/Npos​ >1, the same strategy would up-weight positive samples to mitigate majority-negative dominance. Optimization uses Adam (lr = 1e-3) on the classifier head (model.fc) only, keeping the pretrained DenseNet backbone frozen. The learning-rate scheduler (ReduceLROnPlateau) and early stopping monitor the validation loss, and the final checkpoint corresponds to the epoch with the lowest val_loss. Performance is also reported with MCC computed from epoch-level TP/TN/FP/FN counts at a fixed threshold of 0.5.

----

## **Second Model: Custom CNN (Evolution of Yen & Tsao, 2024)**

Our architecture evolves the lightweight philosophy proposed by Yen and Tsao (2024), introducing a specialized Multi-Feature (MF) Fusion layer and optimized Feature Extraction (FE) modules. The model is specifically engineered for chest X-Ray classification, balancing high-order spatial features with low computational overhead through channel-splitting and dilated convolutions.
The architecture is organized into three phases:
- Efficient Feature Extraction (FE Modules) using depthwise-separable split-channels.
- Multi-Scale Feature Fusion (MF Module) for global and local context.
- Regularized Classification Head for robust binary inference.

### **1. Feature Extraction (FE Module) with Split-Channel Strategy**  

The FE module focuses on learning hierarchical spatial features while minimizing parameter count. To achieve this, we implemented a Channel-Splitting approach:

- Input Transformation: The input tensor is passed through a 1×1 convolution to adjust depth.
- Channel Split: The feature map is split into two branches ($x_1$ and $x_2$).
- Depthwise Processing: While $x_1$ acts as a partial identity, $x_2$ undergoes a 3×3 Depthwise Convolution. 

This operation applies a single filter per input channel, drastically reducing the FLOPs (Floating Point Operations) compared to standard convolutions.
Recombination: The branches are concatenated, followed by a residual shortcut connection.
Residual Formulation:

$$H(x) = \text{ReLU}(F(x) + \text{shortcut}(x))$$

where $F(x)$ incorporates the split-channel transformation, stabilizing gradient flow and mitigating the vanishing gradient problem in deep medical imaging tasks.

### **2. Multi-Feature Fusion (MF Module)** 


To address the variable size of pathological patterns in pneumonia (from small focal opacities to large lobar consolidations), we introduced a Multi-Feature (MF) Module.This module utilizes Dilated Convolutions to capture features at multiple receptive fields without increasing the number of parameters or losing spatial resolution:

- Parallel Dilated Branches: Three parallel 3×3 Depthwise Convolutions with dilation rates of 1, 2, and 4.
- Feature Aggregation: The outputs are concatenated to fuse fine-grained textures with broader structural context.
- Dimensionality Reduction: A final 1×1 convolution compresses the fused features before the final classification stage.

### **3. Classification Head & Structural Regularization**

Following the final feature extraction, we utilize Global Average Pooling (GAP). Unlike traditional flattening, GAP reduces the total parameter count and acts as a structural regularizer by enforcing a direct correspondence between feature maps and the classification output.
The head consists of:

Dense Layer: 96 units with ReLU activation.
Dropout (0.5): To prevent co-adaptation of neurons and ensure generalization to unseen datasets.
Output Layer: A single neuron with Sigmoid activation, providing the probability score for binary classification (Normal vs. Pneumonia).

### **4. Optimization & Regularization Strategy**

To ensure convergence and prevent overfitting on the imbalanced Chest X-Ray dataset, we adopted a multi-layered optimization strategy:
- L2 Regularization: A penalty of $0.001$ is applied to all kernels and depthwise kernels to enforce weight decay.
- Label Smoothing (0.1): Instead of "hard" 0/1 targets, we use smoothed labels to improve model calibration and prevent overconfidence.
- Weighted Binary Cross-Entropy: Class weights (manually tuned or balanced) are applied to the loss function to compensate for the higher frequency of pneumonia cases.
- Optimization: The Adam optimizer is paired with a ReduceLROnPlateau scheduler (factor 0.2, patience 5) to fine-tune weights as the validation loss plateaus.
- Mixed Precision: Training is conducted using float16 computation (with float32 master weights) to maximize GPU efficiency on Google Colab.

----

### **Final evaluation**

#### **Pretrained DenseNet-121**
On the test set, the model achieved strong performance, with an F1-score of 0.949 and an MCC of 0.861. The confusion matrix shows a limited number of misclassifications, with 36 false positives and only 5 false negatives. The decision threshold was selected on the validation set to maximize the F1-score, favoring sensitivity to pneumonia cases. In this setting, accepting a higher number of false positives while keeping false negatives low can considered a more precautionary and clinically safer approach. 
Threshold-free metrics (AUROC and AUPRC) are first computed from predicted probabilities to assess ranking performance. Error analysis is further supported by explicitly identifying false positives and false negatives at the image level. Results show strong discriminative performance (AUROC 0.985, AUPRC 0.990) and a recall-oriented behavior for the PNEUMONIA class (recall 0.987), minimizing false negatives.

| Metric | Value |
|---|---:|
| F1-score | 0.955 |
| MCC | 0.877 |
| AUROC | 0.9848 |
| AUPRC | 0.9899 |


|                  | Predicted PNEUMONIA | Predicted NORMAL |
|------------------|-------------------:|-----------------:|
| Actual PNEUMONIA | 385                | 5                |
| Actual NORMAL    | 36                 | 198              |



#### **Custom CNN **

On the test set (n=624), the proposed FE-MF architecture achieved a global accuracy of 0.88 and a weighted F1-score of 0.88. The model demonstrated a highly specialized behavior: while maintaining an exceptional precision of 0.98 for the NORMAL class (0.0), it achieved a near-perfect recall of 0.99 for the PNEUMONIA class (1.0).

The decision threshold was strategically set at 0.6 to prioritize clinical safety. In this diagnostic setting, the model successfully minimized False Negatives—the most critical error in medical imaging—ensuring that almost all pneumonia cases (386 out of 390) were correctly identified.

Threshold-free metrics were computed to assess the model's ranking performance. The model reached an AUROC of 0.985 and an AUPRC of 0.990, confirming that the internal feature representations learned by the FE and MF modules are highly discriminative.

By favoring Sensitivity (Recall 0.99) over Specificity, the model adopts a "precautionary" clinical approach. Accepting a higher number of False Positives (68) while keeping False Negatives to a minimum (only 4 cases missed) is considered a safer screening strategy, as suspicious cases can be further reviewed by a radiologist, whereas a missed pneumonia case could lead to untreated progression.

| Metric | Value |
|---|---:|
| F1-score | 0.88 |
| Accuracy | 0.88 |
| Precision | 0.90 |
| Recall | 0.88 |


|                  | Predicted PNEUMONIA | Predicted NORMAL |
|------------------|-------------------:|-----------------:|
| Actual PNEUMONIA | 386                | 4                |
| Actual NORMAL    | 69                 | 165              |



Let:
- TP = true positives  
- TN = true negatives  
- FP = false positives  
- FN = false negatives  

- F1-score
Precision = TP / (TP + FP)  
Recall = TP / (TP + FN)  

F1 = 2 · (Precision · Recall) / (Precision + Recall)  
   = 2·TP / (2·TP + FP + FN)

- Matthews Correlation Coefficient (MCC)

MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

- AUROC

TPR (True Positive Rate) = TP / (TP + FN)  
FPR (False Positive Rate) = FP / (FP + TN)  

AUROC is the area under the ROC curve, obtained by plotting TPR versus FPR while sweeping the decision threshold.

- AUPRC

Precision = TP / (TP + FP)  
Recall = TP / (TP + FN)  

AUPRC is the area under the Precision–Recall curve, obtained by plotting Precision versus Recall while sweeping the decision threshold.

#### Conclusions



### **Reproducibility**

Modular and version-controlled codebase suitable for GitHub. The dataset is excluded from the repository using `.gitignore`.

