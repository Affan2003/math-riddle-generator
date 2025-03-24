# app.py
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Load models with error handling
@st.cache_resource
def load_model(model_path):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device="cpu"
        )
        logger.info(f"Model loaded successfully from {model_path}!")
        return generator
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Failed to load model from {model_path}. Please check the logs.")
        return None

# Function to generate math meme examples dynamically
def generate_meme_examples(generator, num_examples):
    examples = []
    operations = ['+', '-', '*', '/', '^']
    numbers = [str(i) for i in range(1, 10)]
    
    for _ in range(num_examples):
        try:
            # Generate random incorrect math expression
            expr = f"{random.choice(numbers)} {random.choice(operations)} {random.choice(numbers)}"
            if random.random() > 0.5:
                expr += f" {random.choice(operations)} {random.choice(numbers)}"
            
            # Make it intentionally wrong
            wrong_answer = random.randint(1, 20)
            prompt = f"Incorrect: {expr} = {wrong_answer}\nCorrect:"
            
            # Generate correction
            output = generator(
                prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.3,
                truncation=True
            )
            
            generated_text = output[0]["generated_text"]
            
            if "\nCorrect:" in generated_text:
                before = prompt.replace("Incorrect: ", "").split("\n")[0].strip()
                after = generated_text.split("\nCorrect:")[1].strip()
                examples.append({"before": before, "after": after})
                
        except Exception as e:
            logger.error(f"Error generating meme example: {e}")
    
    return examples if examples else None

# Improved meme repair function
def repair_meme(generator, meme_text):
    try:
        if "=" not in meme_text:
            meme_text += " = ?"
            
        prompt = f"""Fix this incorrect math statement and explain the error:

Incorrect: {meme_text}
Correct:"""
        
        output = generator(
            prompt,
            max_length=150,
            num_return_sequences=1,
            temperature=0.2,  # Lower temperature for more precise outputs
            do_sample=True,
            top_k=40,
            top_p=0.85,
            truncation=True
        )
        
        full_response = output[0]["generated_text"]
        correction = full_response.split("Correct:")[1].strip() if "Correct:" in full_response else full_response
        
        # Clean up the output
        correction = correction.split("\n")[0].strip()
        if "=" not in correction:
            correction = f"{meme_text.split('=')[0].strip()} = {correction}"
            
        return correction
    except Exception as e:
        logger.error(f"Error repairing meme: {e}")
        return None

# Streamlit app
def main():
    st.set_page_config(page_title="Math Fun", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a mode:", 
                              ["Math Riddles", "Math Meme Repair"], 
                              index=0)
    
    if app_mode == "Math Meme Repair":
        st.title("Math Meme Repair Tool 🔧")
        st.write("Fix those viral math memes that get shared with incorrect solutions!")
        
        # Load model once
        meme_generator = load_model("./math_meme_repair")
        
        tab1, tab2, tab3 = st.tabs(["Before/After Examples", "Solve Your Math Meme", "Error Rating"])
        
        with tab1:
            st.write("### Common Math Meme Mistakes and Corrections")
            
            num_examples = st.selectbox(
                "Select number of examples to show:", 
                options=list(range(1, 6)), 
                key="num_examples"
            )
            
            if st.button("Generate Examples", key="generate_examples"):
                if meme_generator is None:
                    st.error("Model failed to load. Check logs.")
                else:
                    with st.spinner("Generating fresh examples..."):
                        examples = generate_meme_examples(meme_generator, num_examples)
                        
                        if examples:
                            for example in examples:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Incorrect Version**")
                                    st.code(example["before"], language="text")
                                with col2:
                                    st.write("**Corrected Version**")
                                    st.code(example["after"], language="text")
                                st.markdown("---")
                        else:
                            st.error("Failed to generate examples. Try again or check logs.")
        
        with tab2:
            st.write("### Fix Your Own Math Meme")
            meme_text = st.text_input(
                "Enter the incorrect math statement (e.g., '2 + 2 × 2 = 8'):", 
                key="meme_text"
            )
            
            if st.button("Repair This Meme", key="repair_meme"):
                if not meme_text.strip():
                    st.warning("Please enter a math statement first.")
                elif meme_generator is None:
                    st.error("Model failed to load. Check logs.")
                else:
                    with st.spinner("Analyzing and repairing..."):
                        correction = repair_meme(meme_generator, meme_text)
                        if correction:
                            st.write("### Original:")
                            st.code(meme_text, language="text")
                            st.write("### Correction:")
                            st.code(correction, language="text")
                        else:
                            st.error("Failed to generate correction. Try again or check logs.")
        
        with tab3:
            st.markdown(ERROR_RATING)

if __name__ == "__main__":
    main()
