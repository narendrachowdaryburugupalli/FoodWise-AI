import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "qwen2.5:1.5b"


def generate_answer(
    question,
    retrieved_context,
    students,
    predicted_demand,
    food_prepared,
    surplus,
    risk
):
    """
    Generate a FoodWise AI response using a local LLM.

    The model receives:
    1. User question
    2. Retrieved RAG knowledge
    3. Current FoodWise ML prediction
    4. Current surplus/risk information
    """

    prompt = f"""
You are FoodWise AI, a campus food sustainability
decision-support assistant.

Your job is to provide concise, practical and
responsible answers about food demand forecasting,
food waste reduction, sustainability and redistribution.

IMPORTANT RULES:

- Use the provided knowledge context.
- Do not invent facts that are not supported by the context.
- Clearly distinguish estimates from measured values.
- Do not claim food is safe for redistribution.
- Do not make final operational decisions.
- Human staff must make final food preparation and
  redistribution decisions.
- If the knowledge context does not contain enough
  information, say so.
- Keep the answer easy to understand.

CURRENT FOODWISE DATA:

Students:
{students}

Predicted Demand:
{predicted_demand} meals

Food Prepared:
{food_prepared} meals

Potential Surplus:
{surplus} meals

Current Risk:
{risk}

RETRIEVED KNOWLEDGE:

{retrieved_context}

USER QUESTION:

{question}

Provide a concise answer using the retrieved knowledge
and current FoodWise data.
"""


    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )


        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "response",
            ""
        ).strip()


        if not answer:

            return (
                "The local AI model did not return "
                "an answer."
            )


        return answer


    except requests.exceptions.ConnectionError:

        return (
            "⚠️ FoodWise AI could not connect to Ollama. "
            "Please make sure Ollama is running."
        )


    except requests.exceptions.Timeout:

        return (
            "⚠️ The local AI model took too long to respond. "
            "Try asking a shorter question."
        )


    except Exception as e:

        return (
            f"⚠️ AI generation error: {str(e)}"
        )