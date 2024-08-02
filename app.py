import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from PIL import Image

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
feature_model=tf.keras.models.load_model('feature_model.keras')
model = tf.keras.models.load_model('best_model_inceptionv3.keras')
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(image):
    

    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(35):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], 35, padding='post')
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    # Split the generated sequence to exclude the first and last words
    final_caption = in_text.split()[1:-1]
    # Join the words to form the final caption
    final_caption = ' '.join(final_caption)
    
    return final_caption
    
def generate_caption(image):
    print(image)
    
    image = image.resize((299, 299))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array.reshape((1, 299, 299, 3))
    image = tf.keras.applications.inception_v3.preprocess_input(image_array)
    feature = feature_model.predict(image, verbose=0)
    caption = predict_caption(feature)
    return caption

gr.Interface(fn=generate_caption,
             inputs=gr.Image(label='Upload a photo',type="pil"),
             outputs=gr.Label(label='Caption'),
             examples=['1028205764_7e8df9a2ea.jpg','123.jpg','1001773457.jpg','1024138940_f1fefbdce1.jpg'],
             title='Image Caption Generator',
             ).launch(share=True)

