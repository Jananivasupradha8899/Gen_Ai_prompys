import os
from openai import OpenAI
from dotenv import load_dotenv
import prompts

load_dotenv()

class ChatEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def call_llm(self, messages, temperature=0.7):
        if not self.client:
            return "ERROR: No API Key provided."
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    # --- Tool Mocks ---
    def get_order_status(self, order_id):
        # Simulated database lookup
        orders = {"123": "Delivered", "456": "In Transit", "789": "Processing"}
        return f"Order {order_id} status: {orders.get(order_id, 'Not Found')}"

    def get_refund_policy(self):
        return "Returns are accepted within 30 days of purchase for a full refund if items are in original condition."

    # --- Orchestration ---
    def process_query(self, query):
        reasoning_log = []

        # 1. Chain of Thought (Reasoning)
        reasoning_log.append("--- [Stage 1: Chain of Thought] ---")
        cot_messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": prompts.COT_PROMPT.format(query=query)}
        ]
        reasoning_output = self.call_llm(cot_messages)
        reasoning_log.append(reasoning_output)

        # 2. Decision Logic (Mock ReAct Trigger)
        # In a full demo, we'd use function calling. Here we'll simulate a check.
        draft_response = ""
        if "order" in query.lower() or "where" in query.lower():
            reasoning_log.append("\n--- [Stage 2: ReAct - Fetching Tool Info] ---")
            # Extract number for order id mock
            words = query.split()
            order_id = next((w for w in words if w.isdigit()), "123")
            tool_output = self.get_order_status(order_id)
            reasoning_log.append(f"Tool Action: get_order_status({order_id})\nObservation: {tool_output}")
            
            # Incorporate tool output into draft
            draft_messages = [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\nReasoning: {reasoning_output}\nTool Info: {tool_output}\nCompose a draft reply."}
            ]
            draft_response = self.call_llm(draft_messages)
        else:
            draft_messages = [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\nReasoning: {reasoning_output}\nCompose a draft reply."}
            ]
            draft_response = self.call_llm(draft_messages)

        reasoning_log.append(f"\nDraft Response Created: {draft_response[:100]}...")

        # 3. Self-Reflection
        reasoning_log.append("\n--- [Stage 3: Self-Reflection] ---")
        reflection_messages = [
            {"role": "system", "content": "You are a quality auditor."},
            {"role": "user", "content": prompts.REFLECTION_PROMPT.format(query=query, draft=draft_response)}
        ]
        reflection_output = self.call_llm(reflection_messages)
        reasoning_log.append(reflection_output)

        # Final Cleanup (Simple extraction of improved response)
        final_response = reflection_output if "NO CHANGES NEEDED" not in reflection_output else draft_response
        
        return {
            "final_answer": final_response,
            "reasoning_steps": "\n\n".join(reasoning_log)
        }
