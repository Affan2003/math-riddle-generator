import streamlit as st
import logging
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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

@st.cache_resource
def load_model(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
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

def generate_meme_examples(generator, num_examples):
    examples = []
    operations = ['+', '-', '*', '/', '^']
    numbers = [str(i) for i in range(1, 10)]
    
    for _ in range(num_examples):
        try:
            expr = f"{random.choice(numbers)} {random.choice(operations)} {random.choice(numbers)}"
            if random.random() > 0.5:
                expr += f" {random.choice(operations)} {random.choice(numbers)}"
            
            wrong_answer = random.randint(1, 20)
            prompt = f"Incorrect: {expr} = {wrong_answer}\nCorrect:"
            
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
            temperature=0.2,
            do_sample=True,
            top_k=40,
            top_p=0.85,
            truncation=True
        )
        
        full_response = output[0]["generated_text"]
        correction = full_response.split("Correct:")[1].strip() if "Correct:" in full_response else full_response
        
        correction = correction.split("\n")[0].strip()
        if "=" not in correction:
            correction = f"{meme_text.split('=')[0].strip()} = {correction}"
            
        return correction
    except Exception as e:
        logger.error(f"Error repairing meme: {e}")
        return None

def generate_riddle_solution(generator, riddle):
    try:
        prompt = f"Riddle: {riddle}\nSolution:"
        
        output = generator(
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            truncation=True
        )
        
        generated_text = output[0]["generated_text"]
        solution = generated_text.split("Solution:")[1].strip() if "Solution:" in generated_text else generated_text
        
        return solution.split("\n")[0].strip()
    except Exception as e:
        logger.error(f"Error generating riddle solution: {e}")
        return None

def main():
    st.set_page_config(page_title="Math Fun", layout="wide")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a mode:", 
                              ["Math Riddles", "Math Meme Repair"], 
                              index=0)
    
    if app_mode == "Math Riddles":
        st.title("Math Riddle Solver 🧠")
        st.write("Solve challenging math riddles with AI assistance!")
        
        riddle_generator = load_model("./math_riddle_generator")
        
        tab1, tab2 = st.tabs(["Solve Riddles", "Create Riddles"])
        
        with tab1:
            st.write("### Enter a Math Riddle")
            user_riddle = st.text_input(
                "Type your math riddle here (e.g., 'What number do you get when you subtract 10 from 20?'):",
                key="user_riddle"
            )
            
            if st.button("Solve Riddle", key="solve_riddle"):
                if not user_riddle.strip():
                    st.warning("Please enter a riddle first.")
                elif riddle_generator is None:
                    st.error("Model failed to load. Check logs.")
                else:
                    with st.spinner("Thinking hard about this riddle..."):
                        solution = generate_riddle_solution(riddle_generator, user_riddle)
                        if solution:
                            st.write("### Your Riddle:")
                            st.code(user_riddle, language="text")
                            st.write("### Solution:")
                            st.code(solution, language="text")
                        else:
                            st.error("Failed to generate solution. Try again or check logs.")
        
        with tab2:
            st.write("### Generate Random Math Riddles")
            if st.button("Generate New Riddle", key="generate_riddle"):
                if riddle_generator is None:
                    st.error("Model failed to load. Check logs.")
                else:
                    with st.spinner("Creating an interesting riddle..."):
                        output = riddle_generator(
                            "Riddle:",
                            max_length=100,
                            num_return_sequences=1,
                            temperature=0.8,
                            truncation=True
                        )
                        generated_text = output[0]["generated_text"]
                        if "Solution:" in generated_text:
                            riddle_part = generated_text.split("Solution:")[0].strip()
                            st.write("### New Riddle:")
                            st.code(riddle_part, language="text")
                        else:
                            st.code(generated_text, language="text")
    
    elif app_mode == "Math Meme Repair":
        st.title("Math Meme Repair Tool 🔧")
        st.write("Fix those viral math memes that get shared with incorrect solutions!")
        
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
