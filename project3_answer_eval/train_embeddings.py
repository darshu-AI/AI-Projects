import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import os

DATA_PATH = 'data/answers_dataset.csv'
MODEL_PATH = 'models/semantic_model.pkl'

# Load dataset
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Train embeddings for ideal answers
def train_embeddings():
    df = load_data()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute embeddings for each ideal answer
    df['ideal_embedding'] = df['ideal_answer'].apply(lambda x: model.encode(x))
    
    # Save model and embeddings
    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'data': df}, f)
    
    print(f"Semantic model and embeddings saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_embeddings()


