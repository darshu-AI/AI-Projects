import os
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer, util

MODEL_PATH = os.path.join("models", "semantic_model.pkl")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        saved = pickle.load(open(MODEL_PATH, "rb"))
        model = saved["model"]
        data = saved["data"]
    else:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        data = None
    return model, data

def compute_similarity(model, ideal_embedding, student_answer):
    student_emb = model.encode(student_answer)
    similarity = util.cos_sim(student_emb, ideal_embedding).item()
    return float(similarity)

def generate_feedback(similarity_score):
    if similarity_score >= 0.8:
        label = "Correct"
        explanation = "The student answer is very similar to the ideal answer. Key concepts are well covered."
    elif similarity_score >= 0.5:
        label = "Partially Correct"
        explanation = "The student answer covers some important ideas but misses details or uses less precise wording."
    else:
        label = "Needs Improvement"
        explanation = "The student answer is quite different from the ideal answer. Encourage the student to revisit the core concepts."
    return label, explanation

def main():
    st.title("Answer Evaluation & Feedback System (Semantic)")

    st.write("Compare a student's answer with the ideal answer using semantic similarity.")

    model, data = load_model()
    if model is None:
        st.warning("No semantic model found. Please run train_embeddings.py first.")
        return

    # Show question options from CSV
    question_id = st.number_input("Enter question ID", min_value=1, step=1)
    student_answer = st.text_area("Student Answer", height=150)

    if st.button("Evaluate"):
        # Fetch ideal answer and embedding
        if data is None or question_id not in data['question_id'].values:
            st.error("Question ID not found in dataset.")
            return

        row = data[data['question_id'] == question_id].iloc[0]
        ideal_answer = row['ideal_answer']
        ideal_embedding = row['ideal_embedding']

        similarity = compute_similarity(model, ideal_embedding, student_answer)
        label, explanation = generate_feedback(similarity)

        st.subheader("Ideal Answer")
        st.write(ideal_answer)

        st.subheader("Similarity Score")
        st.write(f"{similarity:.3f} (0 = completely different, 1 = identical)")

        st.subheader("Feedback")
        st.write(f"Result: **{label}**")
        st.write(explanation)

if __name__ == "__main__":
    main()
