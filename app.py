# app.py
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample before/after meme examples
MEME_EXAMPLES = [
    {"before": "2 + 2 × 2 = 8", "after": "2 + 2 × 2 = 6 (PEMDAS: multiplication before addition)"},
    {"before": "10 ÷ 0.5 = 5", "after": "10 ÷ 0.5 = 20 (dividing by 0.5 = multiplying by 2)"},
    {"before": "1/2 + 1/3 = 2/5", "after": "1/2 + 1/3 = 5/6 (need common denominator)"},
    {"before": "5! = 100", "after": "5! = 120 (5 factorial = 5×4×3×2×1)"},
    {"before": "8 ÷ 2(2+2) = 1", "after": "8 ÷ 2(2+2) = 16 (PEMDAS left-to-right for same precedence)"},
]

# Error rating content
ERROR_RATING = """
### Model Performance Rating 🔧

**Sass Factor**: 92%  
Your math errors will feel personally attacked  

**Accuracy**: 88%  
Corrects mistakes with 12% overconfidence  

**Meme Potential**: Maximum  
Turns math fails into teachable memes  

**Patience**: 5%  
Basic errors get maximum sass, minimum sympathy  

**Helpfulness**: 95%  
When it's not roasting you, it's actually helpful  

*Warning: May occasionally invent new math rules just to win arguments*
"""

# Load the riddle model
@st.cache_resource
def load_riddle_model():
    try:
        model_path = "./math_riddle_generator"
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        logger.info("Riddle model loaded successfully!")
        return generator
    except Exception as e:
        logger.error(f"Error loading riddle model: {e}")
        st.error("Failed to load the riddle model. Please check the logs.")
        return None

# Load the meme repair model
@st.cache_resource
def load_meme_model():
    try:
        model_path = "./math_meme_repair"
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        logger.info("Meme repair model loaded successfully!")
        return generator
    except Exception as e:
        logger.error(f"Error loading meme repair model: {e}")
        st.error("Failed to load the meme repair model. Please check the logs.")
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

# Function to repair math meme
def repair_meme(generator, meme_text):
    try:
        prompt = f"Incorrect: {meme_text}\nCorrect:"
        output = generator(prompt, max_length=100, num_return_sequences=1, temperature=0.7)
        correction = output[0]["generated_text"].replace(prompt, "").strip()
        return correction
    except Exception as e:
        logger.error(f"Error repairing meme: {e}")
        st.error("Failed to repair the meme. Please check the logs.")
        return None

# Streamlit app
def main():
    st.set_page_config(page_title="Math Fun", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a mode:", 
                              ["Math Riddles", "Math Meme Repair"], 
                              index=0)
    
    if app_mode == "Math Riddles":
        st.title("Math Riddle Generator & Solver 🧩")
        st.write("Welcome to the Math Riddle Generator & Solver! You can either generate new riddles or input your own riddle to get a solution.")
        
        # Create tabs for different functionalities
        tab1, tab2 = st.tabs(["Generate Riddles", "Solve Your Riddle"])
        
        with tab1:
            st.write("### Generate New Math Riddles")
            num_riddles = st.selectbox("Select the number of riddles:", options=list(range(1, 11)), key="num_riddles")
            
            generator = load_riddle_model()
            
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
                    generator = load_riddle_model()
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
    
    elif app_mode == "Math Meme Repair":
        st.title("Math Meme Repair Tool 🔧")
        st.write("Fix those viral math memes that get shared with incorrect solutions!")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["Before/After Examples", "Solve Your Math Meme", "Error Rating"])
        
        with tab1:
            st.write("### Common Math Meme Mistakes and Corrections")
            
            # Counter for examples
            if 'example_counter' not in st.session_state:
                st.session_state.example_counter = 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Incorrect Version**")
                st.write(f"`{MEME_EXAMPLES[st.session_state.example_counter]['before']}`")
            
            with col2:
                st.write("**Corrected Version**")
                st.write(f"`{MEME_EXAMPLES[st.session_state.example_counter]['after']}`")
            
            if st.button("Next Example"):
                st.session_state.example_counter = (st.session_state.example_counter + 1) % len(MEME_EXAMPLES)
                st.experimental_rerun()
        
        with tab2:
            st.write("### Fix Your Own Math Meme")
            meme_text = st.text_input("Enter the incorrect math statement:", key="meme_text")
            
            if st.button("Repair This Meme", key="repair_meme"):
                if not meme_text.strip():
                    st.warning("Please enter a math statement first.")
                else:
                    generator = load_meme_model()
                    if generator is None:
                        st.error("Model is not loaded. Please check the logs.")
                    else:
                        st.write("Analyzing and repairing...")
                        correction = repair_meme(generator, meme_text)
                        if correction:
                            st.write("### Original:")
                            st.write(f"`{meme_text}`")
                            st.write("### Correction:")
                            st.write(f"`{correction}`")
        
        with tab3:
            st.markdown(ERROR_RATING)

if __name__ == "__main__":
    main()
