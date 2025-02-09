### Arabic Font Recognition - README

---

### **Project Overview**
This project implements an Arabic font recognition system by processing images containing Arabic text. The system identifies fonts through the following steps:
1. **Noise Removal**: Cleans the image to enhance text regions.
2. **Segmentation**: Breaks the image into text regions, lines, words, and characters.
3. **Orientation Detection**: Aligns the text for proper processing.
4. **Feature Extraction**: Extracts features using methods like bag-of-words or SIFT.
5. **Font Classification**: Uses a trained classifier to predict the font based on extracted features.

---

### **Features**
- Arabic text segmentation into lines, words, and characters.
- Noise removal for better segmentation accuracy.
- Orientation detection and correction.
- Font recognition through feature extraction and classification.

---

### **Folder Structure**
- `CharCutter.py`: Extracts individual characters from words.
- `Classifier.py`: Implements the font classification model.
- `FeatureExtractor.py`: Handles feature extraction methods.
- `ImageLoader.py`: Loads and visualizes input images.
- `LineCutter.py`: Segments images into individual lines.
- `NoiseRemoval.py`: Reduces noise in the image.
- `OrientationDetector.py`: Corrects the orientation of text.
- `Preprocessing.py`: Prepares images for further analysis.
- `Segmentation.py`: Splits images into meaningful segments (e.g., lines).
- `SeparationRegion.py`: Helps identify regions between characters or words.
- `WordCutter.py`: Segments lines into individual words.
- `main.ipynb`: Example notebook showcasing the pipeline.
- `renaming.py`: Utility script for renaming files.

---

### **How to Run**
1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your dataset in the `fonts-dataset` directory, structured by font types.
3. Run the main script or the notebook to process images and classify fonts:
   ```bash
   python main.py
   ```

---

### **Usage Example**
The following example showcases the pipeline's functionality:

```python
from ImageLoader import *
from NoiseRemoval import *
from Segmentation import *
from OrientationDetector import *
from FeatureExtractor import *
from Classifier import *

# Load an image
path = "./fonts-dataset/2/"
image = ImageLoader.loadImage(path, "1.jpeg")

# Process the image
image = NoiseRemoval.applyGaussianBlur(image)
image = Segmentation.segment(image)
image = OrientationDetector.rotate(image)

# Extract features and classify
feature_extractor = FeatureExtractor()
feature_extractor.loadDataset("./fonts-dataset/")
features, labels = feature_extractor.bagOfWords()

classifier = Classifier()
classifier.train(features, labels)
result = classifier.classify(image[0], feature_extractor, method='SIFT')
print(f"Predicted Font: {result}")
```

---

### **Future Enhancements**
- Improve segmentation accuracy for complex layouts.
- Add support for more robust feature extraction techniques.
- Explore deep learning-based methods for classification.

---

Feel free to contribute or suggest improvements! 😊
