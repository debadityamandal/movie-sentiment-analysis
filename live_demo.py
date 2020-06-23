import pickle
from preprocessing import text_preprocessing
with open('/model_data/model.pkl', 'rb') as f:
    model=pickle.load(f)
with open('/model_data/vect.pkl', 'rb') as f:
    vect=pickle.load(f)
with open('/model_data/le.pkl', 'rb') as f:
    le=pickle.load(f)

def classify_sentiment(review):
    review=list(review.split('\n'))
    review=text_preprocessing(review)
    review=vect.transform(review)
    result=le.inverse_transform(model.predict(review))
    return result

review=input("Please enter review\n")
print(classify_sentiment(review))
