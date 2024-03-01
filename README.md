# LightSpeed NLP: Distillation and Quantization Enhancements for Efficient Model Deployment

This repository showcases the implementation of model optimization techniques, specifically model distillation and dynamic quantization, to improve the efficiency of NLP models. The project focuses on reducing the model size and accelerating inference speed without a significant drop in accuracy, using BERT and DistilBERT models as benchmarks.

## Objective

The primary goal of this project is to demonstrate how model distillation and quantization can be effectively utilized to enhance the performance and efficiency of NLP models. By applying these techniques, we aim to achieve:

- Reduced model size for easier deployment in resource-constrained environments.
- Increased inference speed to facilitate real-time applications.
- Maintained or minimally impacted accuracy to ensure the utility of the optimized models.

## Technologies

- **Python**: The primary programming language used for the project.
- **PyTorch**: Used for model development and quantization.
- **Transformers**: Leveraged for accessing pre-trained models like BERT and DistilBERT.
- **Optuna**: Employed for hyperparameter optimization during the distillation process.

## Models

- **BERT (Base Model)**: Utilized as the teacher model in the distillation process.
- **DistilBERT (Student Model)**: The model obtained post-distillation, serving as the optimized version of BERT.

## Implementation Highlights

1. **Model Distillation**: Implemented knowledge distillation to transfer learning from BERT (teacher) to DistilBERT (student), achieving a balance between performance and efficiency.

2. **Dynamic Quantization**: Applied dynamic quantization on the distilled model to further reduce the model size and improve the inference speed, focusing on linear layers where the majority of computational savings are realized.

3. **Performance Benchmarking**: Conducted comprehensive benchmarks to evaluate the effects of distillation and quantization on model size, inference speed, and accuracy.

## Results

- The distilled and quantized model showed a **50% reduction** in size compared to the original BERT model.
- Inference speed was **increased by over 300%**, making the model suitable for real-time applications.
- The optimized model maintained an accuracy of **87.6%** on the benchmark dataset, demonstrating the effectiveness of the optimization techniques.

## How to Use

1. **Clone the Repository**:
git clone https://github.com/yourusername/optimized-nlp-model.git

2. **Install Dependencies**:
pip install -r requirements.txt


3. **Run the Optimization Script**:
python model_optimization.py


4. **Evaluate the Model**:
python evaluate_model.py
