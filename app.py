# app.py
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    try:
        model_path = "./math_riddle_generator"
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        logger.info("Model loaded successfully!")
        return generator
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Failed to load the model. Please check the logs.")
        return None

# Function to generate riddles
def generate_riddles(generator, num_riddles):
    riddles = []
    for _ in range(num_riddles):
        try:
            output = generator("Riddle:", max_length=50, num_return_sequences=1)
            riddles.append(output[0]["generated_text"])
        except Exception as e:
            logger.error(f"Error generating riddle: {e}")
            st.error("Failed to generate a riddle. Please check the logs.")
    return riddles

# Function to generate solution for a custom riddle
def generate_solution(generator, riddle):
    try:
        prompt = f"Riddle: {riddle}\nSolution:"
        output = generator(prompt, max_length=100, num_return_sequences=1)
        solution = output[0]["generated_text"].replace(prompt, "").strip()
        return solution
    except Exception as e:
        logger.error(f"Error generating solution: {e}")
        st.error("Failed to generate a solution. Please check the logs.")
        return None

# Streamlit app
def main():
    st.title("Math Riddle Generator & Solver 🧩")
    st.write("Welcome to the Math Riddle Generator & Solver! You can either generate new riddles or input your own riddle to get a solution.")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Generate Riddles", "Solve Your Riddle"])
    
    with tab1:
        st.write("### Generate New Math Riddles")
        # Dropdown to select the number of riddles
        num_riddles = st.selectbox("Select the number of riddles:", options=list(range(1, 11)), key="num_riddles")
        
        # Load the model
        generator = load_model()
        
        # Generate riddles when the button is clicked
        if st.button("Generate Riddles", key="generate_riddles"):
            if generator is None:
                st.error("Model is not loaded. Please check the logs.")
            else:
                st.write("Generating riddles...")
                riddles = generate_riddles(generator, num_riddles)
                st.write("### Generated Riddles:")
                for i, riddle in enumerate(riddles, 1):
                    st.write(f"{i}. {riddle}")
    
    with tab2:
        st.write("### Get a Solution for Your Riddle")
        custom_riddle = st.text_area("Enter your math riddle:", height=100, key="custom_riddle")
        
        if st.button("Get Solution", key="get_solution"):
            if not custom_riddle.strip():
                st.warning("Please enter a riddle first.")
            else:
                generator = load_model()
                if generator is None:
                    st.error("Model is not loaded. Please check the logs.")
                else:
                    st.write("Generating solution...")
                    solution = generate_solution(generator, custom_riddle)
                    if solution:
                        st.write("### Your Riddle:")
                        st.write(custom_riddle)
                        st.write("### Possible Solution:")
                        st.write(solution)

# Run the app
if __name__ == "__main__":
    main()
