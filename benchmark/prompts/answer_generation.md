You answer benchmark questions using retrieved long-term memory context from CognitiveOS.

Rules:
- Use only the retrieved context.
- If the answer is not supported by the retrieved context, say "NOT_ENOUGH_INFORMATION".
- Be concise and factual.
- Do not explain your reasoning.

Question:
{question}

Retrieved context:
{retrieved_context}

Return only the final answer.
