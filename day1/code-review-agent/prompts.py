# prompt_templates.py

SYSTEM_PROMPT = """You are an expert senior Python developer and code quality analyst. 
Your goal is to provide insightful, accurate, and actionable code reviews."""

INITIAL_REVIEW_PROMPT = """
Analyze the following Python code for:
1. Syntax errors or logical bugs.
2. Code smells and PEP 8 violations.
3. Suggestions for better performance or readability.

CODE TO REVIEW:
```python
{code}
```

Provide your review in a concise, bulleted format.
"""

CRITIQUE_PROMPT = """
You are performing a self-critique. Below is a code review you recently wrote.
Evaluate it critically for:
1. Hallucinations: Did you suggest a fix for a problem that doesn't exist?
2. Completeness: Did you miss a significant error or a better way to do things?
3. Clarity: Is the feedback actionable and easy to understand?

CODE REVIEW TO CRITIQUE:
{initial_review}

ORIGINAL CODE:
```python
{code}
```

Provide a brief list of points to improve or correct in the original review.
"""

REFINE_PROMPT = """
Based on the following self-critique, improve your original code review.
Ensure the final response is highly professional, accurate, and formatted clearly.

ORIGINAL REVIEW:
{initial_review}

SELF-CRITIQUE:
{critique}

ORIGINAL CODE:
```python
{code}
```

FINAL IMPROVED REVIEW:
"""
