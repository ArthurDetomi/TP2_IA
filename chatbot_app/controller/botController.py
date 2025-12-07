
from service.modelo import predict_sentiment

def answer_question(question, model, tokenizer):
    # Process question
    response = predict_sentiment(model=model, tokenizer=tokenizer, text=question)
    
    return response
    