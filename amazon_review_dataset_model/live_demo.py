import pickle
from tensorflow.python.keras.preprocessing import sequence
import tensorflow as tf

model = tf.keras.models.load_model('model_data/amazon_review_sentiment_model.h5')
with open('model_data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
def classify(query,max_len=255):
    query=list(query.split('\n'))
    query=tokenizer.texts_to_sequences(query)
    query=sequence.pad_sequences(query,maxlen=max_len)
    result=model.predict_classes(query)
    return result

review=input("Please enter review\n")
print(classify(review))