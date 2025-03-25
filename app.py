import streamlit as st
import logging
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

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

@st.cache_resource
def load_emoji_model(base_model_path, adapter_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device="cpu"
        )
        logger.info("Emoji math solver model loaded successfully!")
        return generator
    except Exception as e:
        logger.error(f"Error loading emoji model: {e}")
        st.error("Failed to load emoji math solver model. Please check the logs.")
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

def generate_riddles(generator, num_riddles):
    try:
        riddles = []
        for _ in range(num_riddles):
            output = generator(
                "Riddle:",
                max_length=150,
                num_return_sequences=1,
                temperature=0.8,
                truncation=True
            )
            generated_text = output[0]["generated_text"]
            
            if "Solution:" in generated_text:
                riddle_part = generated_text.split("Solution:")[0].strip()
                solution_part = generated_text.split("Solution:")[1].strip().split("\n")[0]
                riddles.append({"riddle": riddle_part, "solution": solution_part})
            else:
                riddles.append({"riddle": generated_text, "solution": "Could not generate solution"})
        
        return riddles
    except Exception as e:
        logger.error(f"Error generating riddles: {e}")
        return None

def solve_emoji_math(generator, problem):
    try:
        if "→" not in problem:
            problem += " →"
            
        output = generator(
            problem,
            max_length=50,
            num_return_sequences=1,
            temperature=0.3,
            top_k=40,
            top_p=0.9,
            truncation=True
        )
        
        solution = output[0]["generated_text"]
        return solution.split("→")[1].strip() if "→" in solution else solution
    except Exception as e:
        logger.error(f"Error solving emoji math: {e}")
        return None

def generate_emoji_riddles(generator, num_riddles):
    try:
        riddles = []
        emojis = ["🍎", "🚗", "🐶", "🍕", "🎈", "🍦", "🐱", "🚲", "🍩", "🐧", "🍫", "🚀", "🐘", "🍇", "🐦"]
        
        for _ in range(num_riddles):
            emoji = random.choice(emojis)
            operation = random.choice(["+", "-", "×", "÷"])
            count = random.randint(2, 5)
            
            if operation in ["+", "-"]:
                problem = f"{' '.join([emoji]*count)} {operation} {emoji} = {random.randint(5, 30)}"
            else:
                problem = f"{emoji} {operation} {random.randint(2, 5)} = {random.randint(2, 10)}"
                
            output = generator(
                f"{problem} →",
                max_length=50,
                num_return_sequences=1,
                temperature=0.5,
                truncation=True
            )
            
            solution = output[0]["generated_text"]
            if "→" in solution:
                answer = solution.split("→")[1].strip()
                riddles.append({"problem": problem, "solution": answer})
        
        return riddles if riddles else None
    except Exception as e:
        logger.error(f"Error generating emoji riddles: {e}")
        return None

def main():
    st.set_page_config(page_title="Math Fun", layout="wide")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a mode:", 
                              ["Math Riddles", "Math Meme Repair", "Emoji Math Solver"], 
                              index=0)
    
    if app_mode == "Math Riddles":
        st.title("Math Riddle Solver 🧠")
        st.write("Solve challenging math riddles with AI assistance!")
        
        riddle_generator = load_model("./math_riddle_generator")
        
        tab1, tab2 = st.tabs(["Solve Riddles", "Generate Riddles"])
        
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
            st.write("### Generate Math Riddles")
            num_riddles = st.selectbox(
                "Select number of riddles to generate:",
                options=list(range(1, 11)),
                key="num_riddles"
            )
            
            if st.button("Generate Riddles", key="generate_riddles"):
                if riddle_generator is None:
                    st.error("Model failed to load. Check logs.")
                else:
                    with st.spinner(f"Generating {num_riddles} math riddles..."):
                        generated_riddles = generate_riddles(riddle_generator, num_riddles)
                        if generated_riddles:
                            for i, riddle in enumerate(generated_riddles, 1):
                                st.write(f"#### Riddle {i}:")
                                st.code(riddle["riddle"], language="text")
                                st.write("**Solution:**")
                                st.code(riddle["solution"], language="text")
                                st.markdown("---")
                        else:
                            st.error("Failed to generate riddles. Try again or check logs.")
    
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
    
    elif app_mode == "Emoji Math Solver":
        st.title("Emoji Math Solver 🤔➗")
        st.write("Solve and generate emoji math problems!")
        
        emoji_generator = load_emoji_model("gpt2", "./emoji_math_solver")
        
        tab1, tab2 = st.tabs(["Solve Emoji Math", "Generate Emoji Riddles"])
        
        with tab1:
            st.write("### Enter an Emoji Math Problem")
            user_problem = st.text_input(
                "Type your emoji math problem here (e.g., '🍎 + 🍎 + 🍎 = 12 →'):",
                key="user_emoji_problem"
            )
            
            if st.button("Solve Emoji Math", key="solve_emoji_math"):
                if not user_problem.strip():
                    st.warning("Please enter an emoji math problem first.")
                elif emoji_generator is None:
                    st.error("Model failed to load. Check logs.")
                else:
                    with st.spinner("Solving your emoji math problem..."):
                        solution = solve_emoji_math(emoji_generator, user_problem)
                        if solution:
                            st.write("### Your Problem:")
                            st.code(user_problem, language="text")
                            st.write("### Solution:")
                            st.code(solution, language="text")
                        else:
                            st.error("Failed to solve problem. Try again or check logs.")
        
        with tab2:
            st.write("### Generate Emoji Math Problems")
            num_riddles = st.selectbox(
                "Select number of emoji problems to generate:",
                options=list(range(1, 11)),
                key="num_emoji_riddles"
            )
            
            if st.button("Generate Emoji Problems", key="generate_emoji_riddles"):
                if emoji_generator is None:
                    st.error("Model failed to load. Check logs.")
                else:
                    with st.spinner(f"Generating {num_riddles} emoji math problems..."):
                        emoji_riddles = generate_emoji_riddles(emoji_generator, num_riddles)
                        if emoji_riddles:
                            for i, riddle in enumerate(emoji_riddles, 1):
                                st.write(f"#### Problem {i}:")
                                st.code(riddle["problem"], language="text")
                                st.write("**Solution:**")
                                st.code(riddle["solution"], language="text")
                                st.markdown("---")
                        else:
                            st.error("Failed to generate emoji problems. Try again or check logs.")

if __name__ == "__main__":
    main()
