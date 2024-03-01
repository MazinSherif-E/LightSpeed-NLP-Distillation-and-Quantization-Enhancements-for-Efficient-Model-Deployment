from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
from datasets import load_metric, load_dataset

def evaluate_model(model_path, test_dataset):
    """
    Evaluate the model on the test dataset and print out the accuracy.
    
    Args:
        model_path (str): Path to the directory containing the saved model and tokenizer.
        test_dataset (datasets.Dataset): A Hugging Face dataset object for evaluation.
    """
    # Load the tokenizer and model from the specified path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Initialize the pipeline for text classification
    nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Initialize the metric
    accuracy_metric = load_metric("accuracy")
    
    # Predict and evaluate
    print("Evaluating model...")
    for example in test_dataset:
        predictions = nlp_pipeline(example["text"])
        # Assuming the dataset has "labels" and predictions are returned with label ids
        # You might need to adjust this depending on your setup
        predicted_label = predictions[0]['label'].split('_')[-1]  # Extracting label id
        true_label = str(example["label"])  # Ensure it's a string to match prediction format
        accuracy_metric.add_batch(predictions=[predicted_label], references=[true_label])

    final_score = accuracy_metric.compute()
    print(f"Accuracy: {final_score['accuracy']:.4f}")

if __name__ == "__main__":
    # Path to your quantized/distilled model
    model_path = "./quantized_model"
    
    # Load your test dataset
    # For demonstration, using a dummy dataset. Replace with your actual dataset.
    test_dataset = load_dataset("imdb", split='test[:1%]')  # Just an example, adjust accordingly

    # Evaluate the model
    evaluate_model(model_path, test_dataset)
