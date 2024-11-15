# VQA Model for Flood Detection Using BERT 🌊💡

This repository contains a **Visual Question Answering (VQA) model** designed for **flood detection** from UAV-captured images. The model answers questions related to the image, such as identifying flooded areas, based on the content of the image and a given textual question. It combines **YOLOv8** for object detection and **BERT** for question understanding.

---

## 🌟 Overview

This project implements a **Visual Question Answering (VQA) model** that combines object detection and natural language processing (NLP) to answer flood-related questions about UAV images. The model takes an image and a corresponding question as input and generates an answer based on the features detected in the image.

### 🛠️ Key Features:
- **Image Feature Extraction**: Uses YOLOv8 to detect objects and flooded areas within the image.
- **Question Processing**: Utilizes BERT for understanding and embedding the textual question.
- **Answer Generation**: Combines image and question features to predict the most relevant answer.
- **Answer Types**: Can answer questions like "Is the area flooded?", and other related questions.

---

## 📂 **Dataset**
[**FloodNet Challenge 2021 - Track 2**](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021?fbclid=IwAR2XIwe5nJg5VSgxgCldM7K0HPtVsDxB0fjd8cJJZfz6WMe3g0Pxg2W3PlE)

### **Structure**:
- **Annotations**:
  - JSON file linking questions, image paths, and answers.  
- **Images**:  
  - UAV-captured flood scenes with diverse scenarios.

### **Example**:
```json
{
  "image_id": "flood_road_01.jpg",
  "question": "Is the road flooded?",
  "answer": "Yes"
}
```
---

## 🎨 Model Architecture

The model architecture is composed of three primary components:

### **YOLOv8: Image Feature Extraction**

The first step involves using **YOLOv8** (You Only Look Once) for **object detection**. YOLOv8 processes the image to identify key objects such as flooded areas and non-flooded regions.

- **Outputs**:
  - **Bounding Boxes**: Coordinates of detected objects (e.g., flooded regions).
  - **Class Labels**: Labels for detected objects (e.g., "flooded", "non-flooded").
  
- **Processing**:
  - These detections are passed through a **fully connected layer** to generate image feature vectors that are compatible for combining with the question features.

---

### **BERT: Question Feature Extraction**

The question input (e.g., "Is the area flooded?") is tokenized and processed by **BERT**, a pre-trained transformer model designed for natural language understanding.

- **Outputs**:
  - **Tokenized Question**: The question is split into tokens.
  - **Feature Vector**: BERT generates a contextual representation of the question, encapsulating its meaning.

- **Processing**:
  - The question is processed through BERT's pre-trained weights, and the **pooler_output** is used to represent the entire question in vector form.

---

## 🔧 Answer Generation Process

Here’s a detailed explanation of how the **answer generation** works in your VQA system, step by step:

### **1. Image Feature Extraction (YOLOv8)**

- **Purpose**: Extract relevant features from the image, such as objects or areas of interest that may help answer the question.
  
- **Process**:
  - The **YOLOv8 model** (a popular object detection model) is used to detect objects within the image.
  - For each detected object, YOLO returns:
    - **Bounding boxes** (coordinates marking the position of the object).
    - **Class labels** (e.g., "flooded", "non-flooded", etc.).
  
- **Feature Processing**:
  - These detections are processed by extracting the bounding box coordinates and class labels.
  - The features are passed through a fully connected layer (`fc_yolo`) to project them into a lower-dimensional hidden space (`hidden_dim`).

- **Example**:
  - If the image contains an area with water, YOLO might detect a "flooded" area, and the bounding box for that area will be used as part of the image features.

---

### **2. Question Tokenization and Feature Extraction (BERT)**

- **Purpose**: Convert the question into a numerical representation that the model can understand and process.
  
- **Process**:
  - The **BERT tokenizer** first converts the question (e.g., "Is the road flooded?") into a sequence of tokens.
  - Each token is then mapped to an embedding using BERT’s **pre-trained model**.
  - BERT processes the tokenized question and generates a **feature vector** for each token.
  
- **Textual Representation**:
  - The **pooler_output** from BERT is used to represent the entire question, capturing the meaning of the question based on the context of its tokens.

- **Example**:
  - The question "Is the road flooded?" is tokenized and passed through BERT to produce a representation of the question, which understands that the focus is on determining whether an area is flooded.

---

### **3. Encoding the Question Type**

- **Purpose**: Incorporate the **type of question** (e.g., "Yes/No", "Counting", etc.), which can provide valuable context for the model.

- **Process**:
  - The **question type** (like "Yes/No" or "Counting") is mapped to a specific index using the `question_type_mapping` dictionary.
  - This index is passed through an **embedding layer** (`fc_question_type`), which learns to represent the question type in a vector space.
  
- **Example**:
  - For the question "Is the road flooded?" (a Yes/No question), the question type index is mapped to the embedding for **Yes/No**.

---

### **4. Combining Image, Text, and Question Type Features**

- **Purpose**: Combine the features from the image (YOLO), the question (BERT), and the question type to form a comprehensive representation for answering the question.

- **Process**:
  - The **image features** from YOLO (bounding boxes and class labels) are processed through the `fc_yolo` layer to obtain image feature vectors.
  - The **text features** from BERT represent the question.
  - The **question type** features are embedded and used to provide context for the model.
  - All these features are **concatenated** into one combined feature vector, which holds information about the image, the question, and its type.
  
- **Example**:
  - For the question "Is the road flooded?" and an image with a flooded area, the combined vector might include:
    - Image feature vector (showing the presence of flooding).
    - Question vector (understanding that the question asks for a Yes/No answer).
    - Question type vector (indicating that this is a Yes/No question).

---

### **5. Projecting Combined Features**

- **Purpose**: Project the combined feature vector into a space that is compatible with BERT’s expected input dimensions.
  
- **Process**:
  - The concatenated features are passed through a **fully connected projection layer** (`fc_proj`), which maps the combined features to a space that BERT can work with.
  - The output from this layer is reshaped to match the input dimensions expected by BERT for further processing.

- **Example**:
  - The combined feature vector (image + question + question type) is passed through the projection layer, resulting in a vector that is suitable for BERT to process.

---

### **6. Passing Through BERT**

- **Purpose**: Process the combined features through BERT to understand the interaction between the image and the question, and generate a final representation that can be used for answering.

- **Process**:
  - The reshaped combined features are passed to BERT as **inputs_embeds**.
  - The **attention mask** is generated to indicate which tokens should be attended to by BERT (here, a simple mask of ones since we're dealing with a single sequence).
  - The **BERT model** processes the input embeddings and returns the **pooled output**, which encapsulates the interaction between the image features and the question.
  
- **Example**:
  - After BERT processes the combined features, the output represents the interaction between the visual content (flooded area) and the textual content (the question "Is the road flooded?").

---

### **7. Answer Prediction**

- **Purpose**: Use the model’s output to generate a **final answer** based on the highest probability among the possible answers.

- **Process**:
  - The pooled output from BERT is used to generate the final answer. 
  - The model’s final layer computes a **softmax** distribution over all possible answers, where each class corresponds to a potential answer (e.g., "Yes", "No", "flooded", "non-flooded", etc.).
  - The **class with the highest probability** is selected as the final answer.

- **Example**:
  - For the question "Is the road flooded?", if the model finds that the image shows a flooded area and understands the question, it might output the answer "Yes" with a high probability.
  
- **Answer Choices**:
  - Possible answers (classes) are pre-defined in the `label_mapping` list, which includes categories like "flooded", "non-flooded", "Yes", "No", and others.

---

### **8. Training and Loss Calculation**

- **Purpose**: During training, the model is optimized to minimize the error between the predicted answer and the true answer.

- **Process**:
  - The predicted answer (from the softmax distribution) is compared with the true answer (from the dataset) using **Cross-Entropy Loss**.
  - The model adjusts its parameters (weights) to minimize the loss, which helps improve the model's ability to correctly answer questions.

- **Example**:
  - If the true answer for the question "Is the area flooded?" is "Yes", but the model predicts "No", the loss will penalize this error and update the model to reduce similar mistakes in the future.

---

### **Summary of Answer Generation Flow:**

1. **Image Feature Extraction** (via YOLO)
2. **Question Tokenization and Processing** (via BERT)
3. **Question Type Encoding**
4. **Feature Combination**
5. **Projection of Features**
6. **BERT Processing**
7. **Answer Prediction**
8. **Training**

By following this sequence of steps, the model is able to understand both the image and the question and generate a relevant, context-aware answer. The key component is combining the information from object detection (YOLO) and text understanding (BERT) to answer questions related to the image.

---
## 🧪 **Testing & Inference**

### **Evaluation Metrics**:
- **Question-wise Accuracy**: Measures answer correctness per question type.  
- **Overall Accuracy**: Evaluates the model's performance across the dataset.

### **Testing Example**:
```python
from PIL import Image

image_path = "flood_image.jpg"
question = "how many flooded buildings are there in the image?"

# Load the image
image = Image.open(image_path)

# Perform inference
predicted_answer = infer(image, question)

print(f"Predicted Answer: {predicted_answer}")
```

---

## Contributing

We welcome contributions to improve the model! Feel free to fork the repository, open issues, and submit pull requests.

### How to Contribute:
- Fork the repository and clone it to your local machine.
- Create a new branch for your changes.
- Write tests for your changes if applicable.
- Submit a pull request detailing the changes.

---

## 💬 **Contact**

Feel free to open an issue or reach out for collaboration!  

**Author**: *Abdelkadir Sellahi*

**Email**: *abdelkadirsellahi@gmail.com* 

**GitHub**: [Abdelkadir Sellahi](https://github.com/AbdelkadirSellahi)
