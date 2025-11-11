# 🎧 Audio Event Classification using Deep Learning

## 📘 Overview
This project focuses on **environmental sound classification** using deep learning models.  
The goal is to accurately detect and classify different types of sounds — such as *footsteps, gunshots, broken branches,* and *background noise* — to support intelligent acoustic monitoring systems.

We experimented with multiple architectures including **MobileNetV2**, **YAMNet**, and **LightGBM**, while also implementing **incremental learning** methods to improve model adaptability to new sound classes.

---

## 🧠 Key Features
- **4-Class Sound Event Classification:**
  - `backgroundnoise`
  - `brokenbranches`
  - `footsteps`
  - `gunshot`
- **Feature Extraction:** MFCCs, Zero-Crossing Rate, Mel Spectrograms  
- **Data Augmentation:** Time-shifting, pitch scaling, and noise injection  
- **Model Comparison:** CNN, LightGBM, YAMNet fine-tuning, and incremental learning (iCaRL, LwF, GEM, TEEN)  
- **Performance Tracking:** Integrated **CodeCarbon** for energy consumption tracking  
- **Evaluation Metrics:** Accuracy, F1-score, AUC, and energy efficiency  

---

## 🧩 Technologies Used
- **Languages:** Python  
- **Libraries:** TensorFlow, Keras, Librosa, scikit-learn, LightGBM  
- **Tools:** CodeCarbon, Matplotlib, Seaborn  

---

## ⚙️ Workflow
1. **Data Preprocessing:** Load, clean, and extract MFCC/ZCR features.  
2. **Feature Engineering:** Compute spectrograms and augment training data.  
3. **Model Training:** Compare CNN, MobileNetV2, and YAMNet fine-tuning approaches.  
4. **Incremental Learning:** Apply TEEN and LwF methods to enable model updates without forgetting.  
5. **Evaluation:** Assess models using precision, recall, F1, AUC, and energy metrics.  

---

## 📊 Results
- YAMNet fine-tuning achieved the best performance with an **F1-score > 0.90**.  
- TEEN-based incremental learning preserved accuracy across new class additions.  
- CodeCarbon integration revealed a **25% reduction in energy consumption** after optimization.  

---

## 🏗️ Future Work
- Extend dataset with real-world field recordings.  
- Deploy lightweight version for edge devices.  
- Integrate real-time inference system with audio streaming.  

---

## 👥 Contributors
Developed collaboratively during the **Summer Deep Learning Project (2025)**.  
Special thanks to the research and data science teams for their contributions and feedback.
