from transformers import pipeline

# Load trained model (make sure folder exists locally or in repo)
MODEL_PATH = "fake_news_bert_model"

classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

label_map = {
    "LABEL_0": "Fake",
    "LABEL_1": "Real"
}

def predict(text):
    result = classifier(text)[0]
    
    return {
        "text": text,
        "label": label_map.get(result["label"], result["label"]),
        "confidence": result["score"]
    }

if __name__ == "__main__":
    print("Fake News Detection System")
    print("Type 'exit' to quit\n")

    while True:
        text = input("Enter news text: ")
        
        if text.lower() == "exit":
            break
        
        output = predict(text)
        print("\nResult:", output, "\n")