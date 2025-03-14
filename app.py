# app.py
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model_path = "./math_riddle_generator"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

# Function to generate riddles
def generate_riddles(generator, num_riddles):
    riddles = []
    for _ in range(num_riddles):
        output = generator("Riddle:", max_length=50, num_return_sequences=1)
        riddles.append(output[0]["generated_text"])
    return riddles

# Streamlit app
def main():
    st.title("Math Riddle Generator 🧩")
    st.write("Welcome to the Math Riddle Generator! Select the number of riddles you want to generate.")

    # Dropdown to select the number of riddles
    num_riddles = st.selectbox("Select the number of riddles:", options=list(range(1, 11)))

    # Load the model
    generator = load_model()

    # Generate riddles when the button is clicked
    if st.button("Generate Riddles"):
        st.write("Generating riddles...")
        riddles = generate_riddles(generator, num_riddles)
        st.write("### Generated Riddles:")
        for i, riddle in enumerate(riddles, 1):
            st.write(f"{i}. {riddle}")

# Run the app
if __name__ == "__main__":
    main()
