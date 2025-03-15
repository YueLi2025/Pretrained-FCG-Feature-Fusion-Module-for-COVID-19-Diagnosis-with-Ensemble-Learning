# 🧬 COVID-19 Diagnosis with Deep Learning 🚀

Welcome to the repository for **Improving COVID-19 Diagnosis via Pre-trained Fine- and Coarse-Grained Feature Fusion Module and Ensemble Learning**. This project leverages deep learning to enhance COVID-19 diagnosis using chest X-ray images 📸.

## 📚 Overview

COVID-19 remains a major global health crisis, and automated diagnosis can greatly assist in rapid detection. Our model integrates:

- **Fine- and Coarse-Grained Feature Fusion (FCG) Module** 🏢
- **Pre-trained DenseNet121 and VGG-11 models** 🧠
- **Ensemble learning techniques** 🗮️
- **MixUp augmentation for better generalization** 🎭

This approach has achieved **state-of-the-art performance** in classification accuracy and robustness.

---

## ⚙️ Installation & Setup

To set up the environment, install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Model Training

### 📌 Pre-train the Model

```bash
cd scripts
bash train.sh    # Train the model
bash test.sh     # Evaluate the model
```

### 🎯 Train from Scratch

#### Baselines:

```bash
cd scripts
bash train_vgg11.sh        # Train VGG-11
bash train_densenet121.sh  # Train DenseNet-121
```

#### FCG-based Model:

```bash
cd scripts
bash train.sh         # Train our FCG model
bash train_mixup.sh   # Train with MixUp augmentation
```

---

## 🔍 Fine-Tuning

To fine-tune the models on a COVID-19 dataset:

#### Baselines:

```bash
cd scripts
bash finetune_densenet121.sh  # Fine-tune DenseNet-121
```

#### FCG-based Model:

```bash
cd scripts
bash finetune.sh         # Fine-tune FCG model
bash finetune_mixup.sh   # Fine-tune with MixUp augmentation
```

---

## 📊 Results & Performance

Our approach surpasses existing methods in classification accuracy:

| Model              | Accuracy (%) | AUC_OVR | AUC_OVO |
|-------------------|-------------|---------|---------|
| Kaggle VGG16      | 90.33       | -       | -       |
| DenseNet121       | 90.75       | 98.74   | 98.74   |
| **Ours (FCG + Ensemble)** | **92.50**  | **98.80** | **98.80** |

---

## 🖼️ Model Interpretability

Using **Grad-CAM visualization**, we can see that our model focuses on **disease-relevant lung regions**, improving diagnostic reliability. 📱

---

## 🐝 Citation

If you find this work useful, please consider citing:

```
@article{li2025covid,
  author    = {Yue Li},
  title     = {Improving COVID-19 Diagnosis via Pre-trained Fine- and Coarse-Grained Feature Fusion Module and Ensemble Learning},
  journal   = {Medical AI Journal},
  year      = {2025}
}
```

---

## 💌 Contact

For any questions or discussions, feel free to reach out:

🔗 GitHub: [YueLi2025](https://github.com/YueLi2025)

Happy coding! 🧜‍♂️✨
