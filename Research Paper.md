# A Comparative Study of Deep Learning Architectures for Automated Mobile Phone Surface Defect Classification

**Authors:** [Author Name(s)]
**Affiliation:** [Department, University]
**Corresponding Email:** [email@university.edu]

---

## Abstract

Mobile phone manufacturing demands strict quality control to detect surface defects such as oil marks, scratches, and stains. Manual visual inspection is slow, inconsistent, and prone to human error. This paper presents a systematic comparison of ten deep learning models for automatic four-class surface defect classification on mobile phone screens. Eight individual backbone architectures were evaluated: VGG16, ResNet50, MobileNetV2, EfficientNet-B0, DeiT-Small, Swin Transformer Tiny, DINOv2-Small, and ConvNeXt-Tiny. Two hybrid approaches were also tested: a CNN ensemble fusing ResNet50, VGG16, and MobileNetV2 features, and a DINOv2 foundation model paired with XGBoost. All features were extracted from frozen ImageNet-pretrained backbones and cached using FAISS for efficiency. Lightweight MLP classifier heads were trained on the cached features with SMOTE oversampling to handle class imbalance. The DeiT-Small vision transformer achieved the best overall performance with 99.45% accuracy, 0.9958 macro F1-score, and 0.9998 macro AUC, misclassifying only one out of 183 test images. All ten models exceeded 96% accuracy, and no model required elimination. These findings demonstrate that transfer learning with vision transformers is highly effective for industrial surface defect detection, even on CPU hardware with a small dataset of 1,220 images.

**Keywords:** Mobile Phone Defect Detection, Transfer Learning, Vision Transformers, Convolutional Neural Networks, FAISS, Image Classification, Quality Control, DeiT, SMOTE

---

## 1. Introduction

The global smartphone market has grown rapidly over the past decade. As of recent estimates, billions of mobile devices are manufactured every year across factories worldwide. With such large production volumes, ensuring that each device meets visual quality standards is a significant engineering challenge. Even a small percentage of defective units reaching consumers can damage brand reputation and increase return rates. Therefore, robust and reliable quality inspection systems are needed at every stage of the manufacturing pipeline.

Surface defects on mobile phone screens are among the most common quality issues encountered during production. These defects include oil residue from handling, scratches from contact with hard surfaces during assembly, and stains from chemical processes or environmental contamination. Each of these defect types has distinct visual characteristics. Oil marks tend to appear as diffuse, slightly translucent patches. Scratches show up as thin, elongated lines with sharp edges. Stains present as irregular discolored regions on the screen surface. A "good" unit, by contrast, has a clean and uniform appearance.

Traditionally, trained human inspectors perform visual quality control on production lines. An inspector picks up each device, examines the screen under controlled lighting, and decides whether the unit passes or fails. This approach has several well-known limitations. First, it is slow. A human inspector can only examine a fixed number of devices per hour, which becomes a bottleneck as production speeds increase. Second, it is subjective. Two inspectors may disagree on whether a faint oil mark counts as a defect. Third, it is tiring. After several hours of repetitive visual work, an inspector's accuracy naturally drops due to fatigue. These factors together mean that manual inspection is neither scalable nor fully reliable for modern high-volume manufacturing.

Computer vision and machine learning have emerged as promising alternatives for automating visual inspection tasks. In particular, deep learning methods based on convolutional neural networks have shown strong results in image classification problems across many domains, including medical imaging, satellite imagery, and industrial defect detection. The core idea is to train a neural network on labeled images of defective and non-defective surfaces so that it learns to distinguish between different defect categories automatically. Once trained, such a system can classify new images in a fraction of a second, operating consistently without fatigue.

Transfer learning has further reduced the practical barriers to applying deep learning in specialized domains like manufacturing. Instead of training a large neural network from scratch on a small factory dataset, a model that was already trained on a large general-purpose dataset like ImageNet can be reused. The lower layers of such a pretrained model have already learned to detect edges, textures, and shapes that are useful across many visual tasks. Only the final classification layers need to be adapted for the specific defect detection problem. This approach requires far less data and far less computation than training from scratch, making it accessible even to organizations without massive GPU clusters.

Over the past five years, vision transformer models have begun to rival and even surpass convolutional networks on many image classification benchmarks. Unlike CNNs, which process images through local sliding filters, transformers use a self-attention mechanism that allows every part of the image to attend to every other part. This global receptive field can be particularly useful for detecting subtle or spatially distributed defects that a local filter might miss. Models like DeiT, Swin Transformer, and DINOv2 represent different approaches to applying the transformer idea to images, each with different trade-offs in terms of accuracy, speed, and training requirements.

Despite the availability of many pretrained architectures, it is not always obvious which model will work best for a specific industrial task. The characteristics of manufacturing defect images differ from natural photographs in important ways. Defect images are often taken under controlled lighting with a fixed camera position, the differences between classes can be very subtle, and the dataset sizes are typically much smaller than large-scale academic benchmarks. A model that ranks first on ImageNet may not necessarily be the best choice for detecting oil marks on a phone screen. Therefore, systematic comparative studies that evaluate multiple architectures on the same dataset under identical conditions are valuable for guiding practical deployment decisions.

Class imbalance is another challenge that commonly arises in defect detection datasets. In a typical production environment, the vast majority of units are non-defective. Even among defective units, some defect types may be much rarer than others. In the dataset used for this study, the "good" class contains only 20 images compared to 400 images for each defect type. If this imbalance is not addressed, a classifier can achieve deceptively high accuracy simply by predicting the majority class most of the time while performing poorly on the minority class. Techniques such as SMOTE oversampling and weighted loss functions are therefore essential components of a practical defect classification pipeline.

This paper presents a thorough comparative study of ten deep learning models for four-class mobile phone surface defect classification. The study includes four traditional CNN architectures (VGG16, ResNet50, MobileNetV2, and EfficientNet-B0), three vision transformer models (DeiT-Small, Swin Transformer Tiny, and DINOv2-Small), one modern CNN (ConvNeXt-Tiny), and two hybrid approaches (a CNN feature ensemble and a DINOv2 plus XGBoost combination). All models were evaluated under identical preprocessing, splitting, and evaluation conditions. Features were extracted once from frozen pretrained backbones and cached to disk using FAISS, allowing rapid experimentation with different classifier heads without repeated feature extraction.

The main contributions of this work are as follows. First, a comprehensive benchmark of ten diverse architectures on a publicly available mobile phone defect dataset is provided. Second, the effectiveness of vision transformers for industrial defect classification is demonstrated, with DeiT-Small achieving 99.45% accuracy. Third, the use of FAISS-based feature caching is shown to make the entire training and evaluation pipeline efficient enough to run on CPU hardware. Fourth, detailed per-class analyses, correlation studies, and statistical comparisons are presented to provide actionable insights for practitioners choosing a model for deployment.

The rest of this paper is organized as follows. Section 2 reviews related work in defect detection. Section 3 describes the proposed method, including the dataset, preprocessing, data splitting, model workflow, and correlation analysis. Section 4 explains the supporting techniques used in the pipeline, namely FAISS, SMOTE, and PCA. Section 5 explains the technology behind each model and the hyperparameters used. Section 6 defines the evaluation metrics. Section 7 presents the experimental results with tables and discussion. Section 8 gives the conclusion, and Section 9 outlines directions for future work.

---

## 2. Related Work

Deep learning based defect detection has been studied widely in manufacturing research. Early work relied on hand-crafted feature descriptors such as Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP) combined with classifiers like Support Vector Machines. While these methods achieved reasonable accuracy on simple defect types, they struggled with more complex or subtle visual patterns and required significant manual feature engineering.

With the rise of convolutional neural networks, several studies applied models like VGG, ResNet, and Inception to surface defect classification on metal, fabric, and glass surfaces. These studies generally showed that CNNs could outperform traditional methods by a significant margin, particularly when transfer learning from ImageNet was used. For example, studies on the NEU steel surface defect dataset demonstrated that ResNet50 with transfer learning could achieve over 98% accuracy on six defect classes.

More recently, vision transformers have been explored for defect detection. DeiT and Swin Transformer have been applied to industrial inspection tasks with promising results, often matching or exceeding CNN performance while offering better interpretability through attention maps. Self-supervised foundation models like DINOv2 have also attracted attention because they produce general-purpose visual features without requiring labeled data during pretraining, making them particularly useful when labeled defect samples are scarce.

Hybrid approaches that combine features from multiple models or pair deep features with traditional machine learning classifiers like XGBoost have also been investigated. The idea is that different architectures capture different aspects of the visual information, and combining them can improve overall robustness. Feature-level fusion, where feature vectors from multiple backbones are concatenated before classification, is one of the simplest and most commonly used fusion strategies.

Despite this growing body of work, comprehensive studies that compare a large number of architectures including both CNNs and transformers on the same mobile phone defect dataset under identical conditions remain limited. This study aims to fill that gap by providing a fair and thorough comparison across ten models on a publicly available benchmark.

---

## 3. Proposed Method

### 3.1 Dataset Used

The dataset used in this study is the Mobile Phone Defect Segmentation Dataset, publicly available on Kaggle. It contains 1,220 images of mobile phone screen surfaces organized into four classes: good (20 images), oil (400 images), scratch (400 images), and stain (400 images). All images have a resolution of 1920 by 1080 pixels in JPEG format. The good class represents defect-free phone screens, while the three defect classes represent the most common surface quality issues found during manufacturing. There is a severe class imbalance, as the good class has only 20 samples compared to 400 for each defect class. This imbalance ratio of 1:20 is a realistic reflection of production environments where non-defective units captured for training are often limited.

### 3.2 Data Preprocessing

All images were resized to 224 by 224 pixels to match the input requirements of the pretrained backbone networks. Pixel values were normalized using the ImageNet channel-wise mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225) values. For the training set, data augmentation was applied including random horizontal flips with a probability of 0.5, random vertical flips with a probability of 0.3, random rotation up to 15 degrees, color jitter with brightness and contrast variations of 0.2, and random affine translations of up to 10 percent. These augmentations help the model generalize better by exposing it to realistic variations of the training images. Validation and test images received only resizing and normalization without any augmentation, ensuring clean and consistent evaluation. To address the class imbalance in the training set, SMOTE (Synthetic Minority Over-sampling Technique) was applied in the feature space, increasing the good class from 14 training samples to 280 samples to match the defect classes. A WeightedRandomSampler was also used during training to ensure balanced class representation in each batch.

### 3.3 Data Splitting

The dataset was split into three subsets using stratified sampling to preserve the class proportions in each subset. Seventy percent of the data (854 images) was assigned to the training set, fifteen percent (183 images) to the validation set, and fifteen percent (183 images) to the test set. The stratified split was performed using scikit-learn's train_test_split function with a fixed random seed of 42 for reproducibility. The training set contained 14 good, 280 oil, 280 scratch, and 280 stain images. The validation and test sets each contained 3 good, 60 oil, 60 scratch, and 60 stain images. This split ensures that each subset has a representative distribution of all four classes, which is particularly important given the extreme rarity of the good class.

### 3.4 Model Workflow

The overall experimental workflow follows a feature extraction and classification pipeline. First, each of the eight pretrained backbone networks (VGG16, ResNet50, MobileNetV2, EfficientNet-B0, DeiT-Small, Swin Transformer Tiny, DINOv2-Small, ConvNeXt-Tiny) was loaded with its ImageNet pretrained weights. All backbone parameters were frozen, meaning no gradient updates were applied to the backbone during training. Images from the training, validation, and test sets were passed through each backbone to extract feature vectors. These features were stored on disk using FAISS (Facebook AI Similarity Search) indices and NumPy arrays, creating a persistent cache. This caching approach avoids the need to re-run feature extraction when experimenting with different classifier configurations, saving significant computation time. For each backbone, a two-layer MLP (Multi-Layer Perceptron) classifier head was trained on the cached features. The MLP head consists of a linear layer followed by batch normalization, ReLU activation, and dropout, repeated twice, with a final linear output layer producing four class logits. Training used the Adam optimizer with cosine annealing learning rate scheduling and early stopping with a patience of seven epochs. For hybrid models, features from multiple backbones were concatenated and reduced using PCA before feeding into the MLP head or an XGBoost classifier.

### 3.5 Correlation Matrix

Two types of correlation analysis were performed to understand the relationships between defect classes. First, a pixel-level inter-class correlation matrix was computed by averaging all images within each class at 224 by 224 resolution and computing the Pearson correlation coefficient between the flattened pixel vectors. This analysis revealed that the oil and scratch classes have the highest pixel-level similarity (r = 0.896), while the good class is most distinct from scratch (r = 0.489). Second, a feature-level correlation matrix was computed using the mean DeiT-Small feature vectors for each class. In feature space, the correlations between classes were more uniformly distributed (ranging from 0.778 to 0.880), suggesting that the transformer features capture more abstract and discriminative information beyond raw pixel similarity. A per-channel RGB correlation analysis was also performed, showing consistent patterns across all three color channels. Additionally, the Pearson correlation between the six evaluation metrics (Accuracy, F1 Macro, F1 Weighted, Precision Macro, Recall Macro, and AUC Macro) was computed, revealing that F1 Macro has the strongest correlation with Precision (r = 0.955) and that Recall has less correlation with overall Accuracy (r = 0.475).

---

## 4. Supporting Techniques

### 4.1 FAISS (Facebook AI Similarity Search)

FAISS is an open-source library developed by Meta AI that is built for fast similarity search and clustering of dense vectors. In simple terms, it lets you store a large collection of number arrays and then quickly find which stored arrays are most similar to a new one. In this study, we used FAISS to cache the feature vectors that each backbone network extracts from the images. Once a backbone processes all the images and produces their feature vectors, those vectors are saved into a FAISS index file along with a NumPy array file on disk. The next time we want to train a different classifier head or test a new setting, we simply load the cached features from disk instead of running all the images through the backbone again. This saves a huge amount of time because feature extraction through large networks like VGG16 or DeiT-Small is the most expensive step in the pipeline.

The FAISS index used in this study is the IndexFlatIP type, which stores vectors and retrieves them using inner product (cosine) similarity. Each backbone has its own cache folder containing the FAISS index and the matching label arrays for the training, validation, and test splits. When the cache files already exist on disk, the system loads them directly, and when they do not exist, it runs the backbone on all images and creates the cache for the first time. This caching design made it possible to run all ten model experiments on CPU hardware in a reasonable amount of time, since the expensive feature extraction step only needed to happen once per backbone rather than once per experiment.

### 4.2 SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE is a technique used to fix class imbalance in a dataset by creating new synthetic samples for the minority class. The problem in this study is that the good class has only 20 images in total and only 14 in the training set, while each defect class has around 280 training images. If we train a model on this imbalanced data directly, it will learn to predict defect classes most of the time and will perform poorly on the good class. SMOTE solves this by looking at the feature vectors of the minority class samples, finding their nearest neighbors in feature space, and generating new synthetic feature vectors that lie along the line between a real sample and its neighbor. This way, the good class is expanded from 14 to approximately 280 training samples, bringing it to the same size as the defect classes.

In this study, SMOTE was applied in the feature space rather than in the image space. This means we first extracted the feature vectors from a backbone, and then applied SMOTE to generate synthetic feature vectors for the good class. This is more effective than generating synthetic images because feature vectors are compact numerical representations that capture the important visual patterns, and interpolating between two feature vectors produces a meaningful new representation. The number of nearest neighbors used by SMOTE was set to the minimum of 5 or one less than the number of minority class samples, which ensures it works correctly even when the minority class is very small. Along with SMOTE, a WeightedRandomSampler was used during training so that each mini-batch contains a roughly equal number of samples from all four classes.

### 4.3 PCA (Principal Component Analysis)

PCA is a mathematical technique that reduces the number of dimensions in a dataset while keeping as much of the important information as possible. It works by finding the directions in the data where the variance is highest and projecting the data onto those directions. In this study, PCA was used in two main places. First, in the Hybrid CNN Ensemble model (Model 9), the concatenated feature vectors from ResNet50, VGG16, and MobileNetV2 have a combined length of 7,424 numbers per image. This is too large to feed directly into a small classifier, so PCA was applied to reduce it down to 512 dimensions while still keeping 97.6 percent of the total variance. Second, in the DINOv2 plus XGBoost model (Model 10), PCA reduced the 384-dimensional DINOv2 features to 256 dimensions before passing them to the XGBoost classifier.

PCA also played a role in the data exploration and visualization stage of the study. Before training any models, PCA was used to reduce the high-dimensional feature vectors to 2 dimensions so they could be plotted on a flat chart. These 2D PCA plots, along with t-SNE visualizations, helped confirm that the feature spaces produced by the pretrained backbones already separate the four defect classes reasonably well, even before any classifier training. The amount of variance explained by the first two principal components varied across backbones, with some models producing more compact and separable clusters than others. Seeing these visualizations early in the pipeline gave confidence that the frozen pretrained features would work well for this classification task.

---

## 5. Technology Used

### 5.1 VGG16

VGG16 is a deep convolutional neural network consisting of 16 weight layers, proposed by the Visual Geometry Group at Oxford University. It uses a simple and uniform architecture with 3 by 3 convolution filters stacked in increasing depth (64, 128, 256, 512 filters) with max-pooling layers between blocks. The model was loaded with ImageNet pretrained weights and the classifier was truncated after the first fully connected layer, producing a 4096-dimensional feature vector per image. A two-layer MLP classifier head with a hidden dimension of 256 and dropout of 0.3 was trained on these features using the Adam optimizer with a learning rate of 0.001, weight decay of 0.0001, cosine annealing scheduling, and early stopping with patience 7 over 50 epochs. VGG16 was chosen as a baseline because its simple architecture and large feature dimension provide a good reference point for comparing against more modern architectures. In our test, VGG16 scored 96.72% accuracy, 0.8583 macro F1-score, and 0.9940 macro AUC.

### 5.2 ResNet50

ResNet50 is a 50-layer deep residual network that introduced skip connections to address the vanishing gradient problem in very deep networks. These skip connections allow the gradient to flow directly through shortcut paths, enabling the training of much deeper networks than was previously possible. The model was loaded with ImageNet V2 pretrained weights and the final fully connected layer was replaced with an identity function, producing 2048-dimensional feature vectors. The same MLP classifier configuration as VGG16 was used: hidden dimension 256, dropout 0.3, learning rate 0.001, weight decay 0.0001, 50 epochs with patience 7. ResNet50 was included because it remains one of the most widely used backbones in computer vision and its residual learning framework has proven effective across many tasks including defect detection. In our test, ResNet50 scored 98.91% accuracy, 0.9438 macro F1-score, and 0.9967 macro AUC.

### 5.3 MobileNetV2

MobileNetV2 is a lightweight convolutional neural network designed for mobile and embedded devices. It uses depthwise separable convolutions and inverted residual blocks to achieve a good balance between accuracy and computational efficiency. The inverted residual block first expands the channel dimension with a 1 by 1 convolution, then applies a depthwise 3 by 3 convolution, and finally projects back to a narrow output with another 1 by 1 convolution. The model produces 1280-dimensional feature vectors. The MLP head used hidden dimension 256, dropout 0.3, learning rate 0.001, and weight decay 0.0001. MobileNetV2 was included because its efficiency makes it a strong candidate for real-time deployment on edge devices in a factory setting. In our test, MobileNetV2 scored 97.27% accuracy, 0.8783 macro F1-score, and 0.9931 macro AUC.

### 5.4 EfficientNet-B0

EfficientNet-B0 is the base model of the EfficientNet family, which uses a compound scaling method to jointly scale network width, depth, and resolution. It was designed using neural architecture search and achieves strong accuracy with fewer parameters than comparable models. The model produces 1280-dimensional feature vectors. The MLP head used hidden dimension 256, dropout 0.3, learning rate 0.001, and weight decay 0.0001. EfficientNet-B0 was included because its optimized architecture often provides better accuracy-per-parameter than hand-designed networks, making it useful for resource-constrained deployment scenarios. In our test, EfficientNet-B0 scored 97.81% accuracy, 0.9247 macro F1-score, and 0.9994 macro AUC.

### 5.5 DeiT-Small (Data-efficient Image Transformer)

DeiT-Small is a vision transformer that was specifically designed to train effectively on ImageNet-scale data without requiring extremely large datasets like the original ViT model. It splits each 224 by 224 image into 196 non-overlapping patches of 16 by 16 pixels, projects each patch into a 384-dimensional embedding, and processes the sequence through 12 transformer encoder layers with 6 attention heads. DeiT introduces a distillation token alongside the class token to incorporate knowledge from a CNN teacher model during training. The model produces 384-dimensional feature vectors. A smaller MLP head with hidden dimension 192 was used together with a lower learning rate of 0.0005 and higher weight decay of 0.005 to prevent overfitting on the compact 384-dimensional features. DeiT-Small was included because it represents the state-of-the-art in data-efficient transformer architectures and its self-attention mechanism can capture global patterns that are useful for detecting spatially distributed defects. In our test, DeiT-Small scored 99.45% accuracy, 0.9958 macro F1-score, and 0.9998 macro AUC.

### 5.6 Swin Transformer Tiny

Swin Transformer Tiny uses a hierarchical design with shifted window based self-attention. Unlike standard vision transformers that compute attention over all patches, Swin computes attention within local windows and shifts these windows between layers to enable cross-window connections. This approach reduces the computational cost from quadratic to linear in the number of patches while still allowing information to flow across the entire image. The model produces 768-dimensional feature vectors. The MLP head used hidden dimension 256, learning rate 0.0005, and weight decay 0.005. Swin Transformer was included because its hierarchical structure produces multi-scale representations similar to CNNs while retaining the benefits of self-attention, making it well-suited for detecting defects of varying sizes. In our test, Swin-Tiny scored 98.91% accuracy, 0.9438 macro F1-score, and 0.9991 macro AUC.

### 5.7 DINOv2-Small

DINOv2 is a self-supervised vision foundation model developed by Meta AI. It was trained on a curated dataset of 142 million images using self-distillation with no labels, meaning it learned visual representations purely from the data without human annotations. The model uses a Vision Transformer Small architecture with a patch size of 14 pixels and produces 384-dimensional feature vectors. A smaller MLP head with hidden dimension 192 and lower dropout of 0.2 was used, with a learning rate of 0.001 and weight decay of 0.0001. DINOv2 was included because foundation models represent a paradigm shift toward general-purpose visual features, and evaluating whether such features transfer effectively to specialized industrial tasks is of significant practical interest. In our test, DINOv2-Small scored 96.72% accuracy, 0.8373 macro F1-score, and 0.9852 macro AUC.

### 5.8 ConvNeXt-Tiny

ConvNeXt-Tiny is a modernized convolutional neural network that incorporates several design ideas from vision transformers into a pure convolutional architecture. It uses larger 7 by 7 depthwise convolution kernels, GELU activation functions, LayerNorm instead of BatchNorm, and an inverted bottleneck design. The result is a CNN that matches or exceeds the performance of Swin Transformer while maintaining the simplicity and efficiency of convolutional operations. The model produces 768-dimensional feature vectors. The MLP head used hidden dimension 256, learning rate 0.0005, and weight decay 0.005. ConvNeXt was included to test whether modernizing the CNN design to incorporate transformer-inspired elements can close the performance gap observed between traditional CNNs and transformers. In our test, ConvNeXt-Tiny scored 97.81% accuracy, 0.8823 macro F1-score, and 0.9981 macro AUC.

### 5.9 Hybrid CNN Ensemble (Model 9)

This hybrid model combines the feature representations from three CNN backbones: ResNet50 (2048 dimensions), VGG16 (4096 dimensions), and MobileNetV2 (1280 dimensions). The features from all three backbones are concatenated to form a 7424-dimensional feature vector per image. PCA (Principal Component Analysis) was applied to reduce this to 512 dimensions while retaining 97.6 percent of the total variance. The reduced features were then passed through the same MLP classifier head with hidden dimension 256 and dropout 0.3, trained with learning rate 0.001 and weight decay 0.0001. This ensemble approach was included to test whether combining complementary feature representations from multiple architectures could outperform any single backbone. In our test, the HybridCNN-Ensemble scored 97.27% accuracy, 0.8506 macro F1-score, and 0.9989 macro AUC.

### 5.10 DINOv2 + XGBoost (Model 10)

This hybrid model pairs the DINOv2-Small foundation model features with an XGBoost gradient boosting classifier instead of a neural network head. The 384-dimensional DINOv2 features were standardized and reduced to 256 dimensions using PCA. XGBoost was then trained with extensive hyperparameter tuning through GridSearchCV using 3-fold stratified cross-validation. The search space included n_estimators in {100, 300, 500}, max_depth in {4, 6, 8}, learning_rate in {0.01, 0.05, 0.1}, and subsample in {0.8, 1.0}. The best configuration was selected based on macro F1-score. This hybrid was included to evaluate whether a strong gradient boosting classifier could extract better decision boundaries from foundation model features than a simple MLP head. In our test, DINOv2 + XGBoost scored 97.27% accuracy, 0.7358 macro F1-score, and 0.9820 macro AUC.

---

## 6. Evaluation Metrics

### 6.1 Accuracy

Accuracy measures the proportion of correctly classified samples out of all test samples. It is computed as the number of correct predictions divided by the total number of predictions. While accuracy gives a quick overall sense of model performance, it can be misleading when class distributions are imbalanced, as a model that always predicts the majority class can still achieve high accuracy. In this study, accuracy ranged from 96.72% to 99.45% across all models.

### 6.2 Precision (Macro)

Precision measures how many of the samples predicted as a given class actually belong to that class. Macro precision computes precision independently for each class and then takes the unweighted average. A high precision means that when the model predicts a defect type, it is usually correct. Low precision indicates many false positives. Macro averaging ensures that performance on the small good class contributes equally to the overall metric, which is important for balanced evaluation with imbalanced data.

### 6.3 Recall (Macro)

Recall measures how many of the actual samples belonging to a class were correctly identified by the model. Macro recall averages the recall across all classes equally. A high recall means the model rarely misses true defects. Low recall means the model fails to detect some defects, which could be costly in a manufacturing setting where missed defects reach consumers. In this study, recall was particularly important for the good class, which had very few test samples.

### 6.4 F1-Score (Macro)

The F1-score is the harmonic mean of precision and recall. It provides a single number that balances both metrics. The macro F1-score averages the F1-score across all classes without weighting by class size. This makes it sensitive to performance on minority classes. The macro F1-score is often considered the most informative single metric for imbalanced classification problems because it penalizes models that achieve high accuracy by ignoring rare classes.

### 6.5 F1-Score (Weighted)

The weighted F1-score computes the F1-score for each class and then averages them weighted by the number of samples per class. This gives more importance to classes with more test samples. It is included alongside the macro F1-score to provide a complementary view: while the macro score treats all classes equally, the weighted score reflects overall performance proportional to the actual class distribution.

### 6.6 AUC-ROC (Macro)

The Area Under the Receiver Operating Characteristic Curve measures the model's ability to discriminate between classes across all possible classification thresholds. It is computed using a one-versus-rest strategy for each class, and the macro average is taken. An AUC of 1.0 indicates perfect discrimination, while 0.5 indicates random guessing. AUC is threshold-independent and provides a comprehensive picture of how well the model's probability estimates separate the classes, not just how well it performs at the default threshold of 0.5.

### 6.7 Confusion Matrix

The confusion matrix is a table that shows the counts of correct and incorrect predictions for each pair of true and predicted classes. Each row represents the actual class and each column represents the predicted class. The diagonal entries show correct classifications and the off-diagonal entries show misclassifications. In this study, both absolute count and normalized percentage confusion matrices were generated for each model to identify specific class-pair confusions.

---

## 7. Results and Discussion

### 7.1 Overall Model Comparison

Table 1 presents the complete results of all ten models evaluated on the held-out test set of 183 images. The models are ranked by accuracy in descending order. DeiT-Small achieved the highest accuracy of 99.45%, followed by ResNet50 and Swin Transformer Tiny tied at 98.91%, and EfficientNet-B0 and ConvNeXt-Tiny tied at 97.81%. All ten models exceeded 96% accuracy, which is notable given that this was achieved using only frozen pretrained features and lightweight MLP classifier heads without any backbone fine-tuning. The smallest performance gap across all models was only 2.73 percentage points (from 99.45% to 96.72%), indicating that all tested architectures extract features that are highly suitable for this defect classification task.

**Table 1: Comprehensive Model Comparison (sorted by Accuracy)**

| Rank | Model | Accuracy | F1 Macro | F1 Weighted | Precision Macro | Recall Macro | AUC Macro | Train Time (s) |
|------|-------|----------|----------|-------------|-----------------|--------------|-----------|----------------|
| 1 | DeiT-Small | 0.9945 | 0.9958 | 0.9945 | 0.9959 | 0.9958 | 0.9998 | 1.7 |
| 2 | ResNet50 | 0.9891 | 0.9438 | 0.9886 | 0.9919 | 0.9125 | 0.9967 | 3.5 |
| 3 | Swin-Tiny | 0.9891 | 0.9438 | 0.9886 | 0.9919 | 0.9125 | 0.9991 | 1.7 |
| 4 | EfficientNet-B0 | 0.9781 | 0.9247 | 0.9792 | 0.8918 | 0.9833 | 0.9994 | 1.7 |
| 5 | ConvNeXt-Tiny | 0.9781 | 0.8823 | 0.9792 | 0.8669 | 0.9042 | 0.9981 | 1.2 |
| 6 | MobileNetV2 | 0.9727 | 0.8783 | 0.9738 | 0.8626 | 0.9000 | 0.9931 | 2.3 |
| 7 | DINOv2+XGBoost | 0.9727 | 0.7358 | 0.9649 | 0.7308 | 0.7417 | 0.9820 | 2962.7 |
| 8 | HybridCNN-Ensemble | 0.9727 | 0.8506 | 0.9771 | 0.8292 | 0.9000 | 0.9989 | 1.6 |
| 9 | VGG16 | 0.9672 | 0.8583 | 0.9698 | 0.8381 | 0.8958 | 0.9940 | 3.5 |
| 10 | DINOv2-Small | 0.9672 | 0.8373 | 0.9735 | 0.8172 | 0.8958 | 0.9852 | 0.8 |

### 7.2 Best Model Analysis: DeiT-Small

DeiT-Small was the clear best performer across every evaluation metric. With only one misclassification out of 183 test images (a scratch sample misclassified as oil), it achieved near-perfect performance. Its confusion matrix shows 100% accuracy on the good, oil, and stain classes, with 98.3% on the scratch class. The per-class precision ranged from 0.983 to 1.000, and per-class recall ranged from 0.983 to 1.000. The ROC curves for all four classes approached perfect discrimination, with per-class AUC values of 1.000 for good, oil, and stain, and 0.9991 for scratch. This outstanding performance can be attributed to the self-attention mechanism of the DeiT transformer, which can model long-range dependencies across the entire 224 by 224 image, capturing global texture and pattern differences between defect types that local convolutional filters may miss.

### 7.3 Architecture Category Comparison

When grouping models by architecture category, vision transformers (DeiT-Small and Swin-Tiny) achieved the highest average accuracy of 0.9918, followed by modern CNNs (ConvNeXt-Tiny) at 0.9781, traditional CNNs (VGG16, ResNet50, MobileNetV2, EfficientNet-B0) at 0.9768, and hybrid models at 0.9727. Notably, the hybrid CNN ensemble that concatenated features from three CNN backbones did not outperform the best individual CNN (ResNet50) or any of the transformers. This suggests that for this particular task, the quality of individual features matters more than the quantity of combined features. The DINOv2+XGBoost hybrid also did not improve upon the DINOv2-Small with an MLP head, and it required significantly longer training time (2962.7 seconds for XGBoost grid search versus 0.8 seconds for the MLP) due to the extensive hyperparameter search grid.

### 7.4 Inter-Model Agreement and Error Analysis

The inter-model prediction agreement heatmap showed that most model pairs agreed on 96% to 99% of test predictions. The highest agreement was between ResNet50 and Swin-Tiny (100.0%), and between ResNet50 and DeiT-Small (98.9%). The lowest agreement was between VGG16/DINOv2-Small and the HybridCNN-Ensemble (94.5%). This analysis reveals that the models that disagree most tend to be those based on fundamentally different architecture paradigms (pure CNNs versus ensemble features), suggesting that their errors are somewhat complementary. The pixel-level correlation analysis showed that the good class has the lowest pixel correlation with other classes (r = 0.489 to 0.831), while oil and scratch are most similar at the pixel level (r = 0.896). Despite this high pixel similarity, all models achieved strong discrimination between oil and scratch, confirming that the deep features capture discriminative patterns beyond raw pixel statistics.

### 7.5 FAISS k-NN Baseline and Practical Efficiency

A FAISS-based k-nearest-neighbor classifier was evaluated as a non-parametric baseline using DeiT-Small features. With k=1, the k-NN achieved 98.36% accuracy, which is remarkably close to the trained MLP head's 99.45%. However, as k increased, performance dropped due to the class imbalance in the training set—with only 14 good class samples, majority voting for larger k values was dominated by the defect classes. This result confirms that the DeiT-Small features are inherently well-separated and that even a simple nearest-neighbor approach can achieve strong results when the feature space is sufficiently discriminative. From a practical standpoint, the entire pipeline runs on CPU hardware. Feature extraction for all eight backbones takes approximately 15 to 30 minutes total (one-time cost), and training each MLP classifier head takes between 0.8 and 3.5 seconds on cached features. This makes the approach highly feasible for deployment in production environments where GPU access may be limited.

---

## 8. Conclusion

This study presented a comprehensive evaluation of ten deep learning models for four-class mobile phone surface defect classification. Eight individual backbone architectures spanning traditional CNNs, vision transformers, foundation models, and modern CNNs were compared alongside two hybrid approaches. All models used frozen pretrained features cached with FAISS and trained lightweight MLP or XGBoost classifier heads with SMOTE oversampling to address class imbalance. The DeiT-Small vision transformer achieved the best performance across all metrics with 99.45% accuracy, 0.9958 macro F1-score, and 0.9998 macro AUC, misclassifying only one out of 183 test samples. All ten models exceeded 96% accuracy, demonstrating that transfer learning with modern pretrained architectures is highly effective for industrial surface defect detection even on small datasets and CPU hardware. The hybrid ensemble and XGBoost-based approaches did not outperform the best individual models, suggesting that architecture quality matters more than feature combination for this task. These findings provide a practical guide for manufacturing engineers selecting a vision model for automated quality control, with DeiT-Small recommended as the primary choice and ResNet50 or Swin-Tiny as strong alternatives.

---

## 9. Future Work

Several directions could extend this research. First, fine-tuning the backbone layers with a low learning rate could potentially push accuracy even closer to 100%. Second, the approach could be tested on larger and more diverse defect datasets from actual production lines to assess generalization. Third, model interpretability through Grad-CAM or attention map visualization could help identify which image regions drive the classification decisions, building trust among quality inspectors. Fourth, the pipeline could be adapted for defect segmentation rather than just classification, enabling precise localization of the defect area. Fifth, knowledge distillation could be used to create a smaller and faster student model that retains the accuracy of DeiT-Small while meeting the latency requirements of real-time production line deployment. Sixth, exploring few-shot and zero-shot learning techniques could address scenarios where new defect types emerge with very limited labeled examples. Finally, integrating the classifier with a robotic inspection system and testing end-to-end performance under real factory conditions would be a valuable step toward practical deployment.

---

## 10. Figures

- **Figure 1:** Class Distribution — bar chart and pie chart showing the dataset composition (20 good, 400 oil, 400 scratch, 400 stain images).
- **Figure 2:** Sample Images — grid of representative images from each defect class.
- **Figure 3:** Pixel Intensity Distribution — per-channel RGB histograms for each class.
- **Figure 4:** PCA and t-SNE Visualization — 2D projections of feature spaces from selected backbones showing class separability.
- **Figure 5:** Comprehensive Model Comparison — grouped bar chart of all metrics across all ten models.
- **Figure 6:** Metric Correlation Matrix — Pearson correlation heatmap between evaluation metrics.
- **Figure 7:** Performance Heatmap — all models versus all metrics in a color-coded grid.
- **Figure 8:** Radar Chart — top five models plotted on five metric axes.
- **Figure 9:** Best Model Confusion Matrix — DeiT-Small confusion matrix in both absolute and normalized forms.
- **Figure 10:** Best Model ROC Curves — per-class and micro-average ROC curves for DeiT-Small.
- **Figure 11:** Per-Class Metrics — precision, recall, and F1-score bar chart for DeiT-Small.
- **Figure 12:** All Models Confusion Matrix Grid — 2 by 5 grid of normalized confusion matrices for all ten models.
- **Figure 13:** Inter-Model Prediction Agreement Heatmap — pairwise agreement percentages.
- **Figure 14:** Backbone Feature Similarity — cosine similarity heatmap between backbone feature representations.
- **Figure 15:** Inter-Class Pixel Correlation Matrix — pixel-level Pearson correlation between class averages.
- **Figure 16:** Pixel versus Feature Correlation — side-by-side comparison of pixel-space and DeiT-Small feature-space class correlations.
- **Figure 17:** Per-Channel RGB Correlation — separate Red, Green, and Blue channel correlation heatmaps.
- **Figure 18:** FAISS k-NN Results — accuracy versus k plot and k=5 confusion matrix.
- **Figure 19:** Accuracy versus Training Time — bubble chart showing efficiency trade-offs.
- **Figure 20:** Category Performance — average accuracy, F1, and AUC by model category.

---

## 11. References

[1] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," in Proceedings of ICLR, 2015.

[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proceedings of CVPR, pp. 770–778, 2016.

[3] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in Proceedings of CVPR, pp. 4510–4520, 2018.

[4] M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in Proceedings of ICML, pp. 6105–6114, 2019.

[5] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, "Training Data-efficient Image Transformers & Distillation through Attention," in Proceedings of ICML, pp. 10347–10357, 2021.

[6] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," in Proceedings of ICCV, pp. 10012–10022, 2021.

[7] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P. Huang, S. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jégou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, "DINOv2: Learning Robust Visual Features without Supervision," Transactions on Machine Learning Research, 2024.

[8] Z. Liu, H. Mao, C. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in Proceedings of CVPR, pp. 11976–11986, 2022.

[9] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in Proceedings of KDD, pp. 785–794, 2016.

[10] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.

[11] J. Johnson, M. Douze, and H. Jégou, "Billion-scale Similarity Search with GPUs," IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2021.

[12] G. Kumar, "Mobile Phone Defect Segmentation Dataset," Kaggle, 2023. Available: https://www.kaggle.com/datasets/girish17019/mobile-phone-defect-segmentation-dataset

[13] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in Proceedings of ICLR, 2021.

---
