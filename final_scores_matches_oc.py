#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import torch
import open_clip
import argparse

model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')

embedding_dir = 'C:/Users/Neiv Gupta/fiftyone/coco-2017/validation'

def get_text_embedding(text):
    text_tokens = tokenizer([text])
    with torch.no_grad():
        embedding = model.encode_text(text_tokens)
    print(embedding)
    return embedding.numpy().squeeze()

def load_image_embeddings(embedding_dir):
    embeddings = {}
    for file_name in os.listdir(embedding_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(embedding_dir, file_name)
            embedding = np.load(file_path)
            embeddings[file_name] = embedding
    return embeddings

def find_matches(text_embedding, image_embeddings):
    scores = {}
    for image_name, image_embedding in image_embeddings.items():     
        score = np.dot(text_embedding, image_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(image_embedding))
        scores[image_name] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def main(prompt):
    text_embedding = get_text_embedding(prompt)
    image_embeddings = load_image_embeddings(embedding_dir)
    matches = find_matches(text_embedding, image_embeddings)
    for image_name, score in matches:
        print(f'{image_name}: {score:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    args = parser.parse_args()
    prompt = args.prompt
    main(prompt)


# In[ ]:





# In[ ]:




