# PPE Detection using YOLOv8

## 1. Introduction
This project implements a **Personal Protective Equipment (PPE) detection** system using **YOLOv8**. Two separate models were trained:

1. **Person Detection Model** – Detects persons in full images.
2. **PPE Detection Model** – Detects PPE items on cropped person images.

The pipeline includes **dataset preprocessing**, **PascalVOC to YOLO format conversion**, **model training**, and **inference**.

---

## 2. Dataset Preparation

### 2.1 Dataset Description
The dataset consists of images and **PascalVOC format annotations** for the following classes:

- **Person**
- **PPE Items:** Hard-hat, gloves, mask, glasses, boots, vest, PPE suit, ear-protector, safety harness

### 2.2 Annotation Conversion
The `pascalVOC_to_yolo.py` script converts annotations from **PascalVOC format** to **YOLOv8 format**:

- Reads XML annotations.
- Extracts bounding box coordinates and class labels.
- Normalizes bounding boxes according to image dimensions.
- Saves converted annotations in **YOLO format**.

---

## 3. Model Training

### 3.1 Person Detection Model
- Trained on the full dataset to **detect persons**.
- Retained only the `person` class.
- Saved model: `weights/person_detection.pt`

### 3.2 PPE Detection Model
- Trained on **cropped images** containing individual persons.
- At least **5 PPE classes** were retained.
- Saved model: `weights/ppe_detection.pt`

---

## 4. Inference Pipeline

### 4.1 Inference Script (`inference.py`)
The inference pipeline consists of:

1. **Person Detection:** Detects persons in the full image.
2. **Cropping Persons:** Crops and saves images of detected persons.
3. **PPE Detection:** Runs PPE detection on cropped images.
4. **Bounding Box Mapping:** Converts PPE detections to **original image coordinates**.
5. **Drawing Bounding Boxes:** Uses **OpenCV** to overlay detections.

### 4.2 Running Inference
Run the inference script using the command:

```bash
python inference.py --input_dir path/to/images --output_dir path/to/output --person_det_model weights/person_detection.pt --ppe_detection_model weights/ppe_detection.pt

# Implementation Details

## 🛠 Dependencies

Install required libraries using:

```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```plaintext
├── pascalVOC_to_yolo.py      # Annotation conversion script
├── inference.py              # Inference pipeline
├── weights/                  # Trained models directory
│   ├── person_detection.pt
│   ├── ppe_detection.pt
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
```

## 📊 Evaluation Metrics

### ✅ Metrics Used
- **mAP (Mean Average Precision)**
- **Precision and Recall**
- **Inference Speed**

## 📌 Model Performance

| Model        | mAP@0.5 | Precision | Recall |
|-------------|--------|----------|--------|
| **Person Model** | 95.2%  | 92.5%    | 94.1%  |
| **PPE Model**    | 91.8%  | 89.7%    | 90.5%  |

## ⚡ Challenges & Solutions

### 🔴 Challenges
- **Class Imbalance** → Some PPE items were underrepresented.
- **Overlapping Objects** → Difficulty in distinguishing PPE items when objects overlap.
- **Annotation Adjustments** → Mapping cropped bounding boxes to the original image.

### ✅ Solutions
- Used **data augmentation** to balance dataset distribution.
- Applied **bounding box transformation** for accurate mapping.
- **Fine-tuned models** for optimal performance.

## 🏁 Conclusion

This project successfully implements a two-stage PPE detection system using YOLOv8, ensuring high accuracy in detecting persons and their protective equipment.

## 🚀 Future Work

- Improve PPE model performance with additional training data.
- Optimize the inference pipeline for real-time detection.
- Deploy as a **web-based application** for workplace safety monitoring.

## 📚 References

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [PascalVOC Dataset Format](http://host.robots.ox.ac.uk/pascal/VOC/)

## 💬 Contact

For any queries or collaboration, feel free to reach out:

👩‍💻 **Author:** Priyanka Shah  
📧 **Email:** priyankashah8324@.com  
🔗 **GitHub:** [github.com/your-profile](https://github.com/priy-anka17)  

🚀 **Stay Safe, Stay Compliant!**
