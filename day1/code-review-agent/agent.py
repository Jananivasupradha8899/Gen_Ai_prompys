import ast
import os
from openai import OpenAI
from dotenv import load_dotenv
import prompts

# Load variables from .env
load_dotenv()

def analyze_code_structure(code: str):
    """
    Perform basic static analysis using AST.
    """
    results = {"errors": [], "smells": [], "functions": 0, "classes": 0}
    try:
        tree = ast.parse(code)
        results["functions"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        results["classes"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        
        # Check for simple smells
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node):
                    results["smells"].append(f"Function '{node.name}' is missing a docstring.")
    except SyntaxError as e:
        results["errors"].append(f"Syntax Error: {e.msg} at line {e.lineno}")
    except Exception as e:
        results["errors"].append(f"Unexpected error parsing AST: {str(e)}")
        
    return results

def mock_llm_response(prompt_type: str, code: str):
    """
    Simulate LLM responses for demonstration without an API key.
    """
    if "INITIAL" in prompt_type:
        return "Initial Review: The code looks functional but lacks documentation and type hints. Indentation seems inconsistent."
    elif "CRITIQUE" in prompt_type:
        return "Critique: I missed that the variable naming is too brief (x, y) and I didn't mention PEP 8 specifics for the return statement whitespace."
    else:
        return "Final Refined Review:\n- **Naming**: Rename `x`, `y` to `a`, `b` or more descriptive names.\n- **PEP 8**: Use 4-space indentation.\n- **Docs**: Add a docstring explaining the function's purpose.\n- **Types**: Add `-> int` return type hint."

def get_llm_response(prompt: str, api_key: str, prompt_type: str = "INITIAL", code: str = ""):
    """
    Call OpenAI API or return a mock response if no key is provided.
    """
    if not api_key or api_key == "MOCK_MODE":
        return mock_llm_response(prompt_type, code)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

def run_reflection_loop(code: str, api_key: str):
    """
    The Agentic Reflection Loop: Review -> Critique -> Refine.
    """
    # 1. Initial Review
    initial_prompt = prompts.INITIAL_REVIEW_PROMPT.format(code=code)
    initial_review = get_llm_response(initial_prompt, api_key, "INITIAL", code)
    yield "initial", initial_review

    # 2. Self-Critique
    critique_prompt = prompts.CRITIQUE_PROMPT.format(code=code, initial_review=initial_review)
    critique = get_llm_response(critique_prompt, api_key, "CRITIQUE", code)
    yield "critique", critique

    # 3. Final Refined Review
    refine_prompt = prompts.REFINE_PROMPT.format(code=code, initial_review=initial_review, critique=critique)
    final_review = get_llm_response(refine_prompt, api_key, "REFINE", code)
    yield "final", final_review
