import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv()

# ── LLM setup ─────────────────────────────────────────────────
llm = ChatGroq(
    api_key     = os.getenv("GROQ_API_KEY"),
    model_name  = "llama-3.3-70b-versatile",
    temperature = 0.7,
    max_tokens  = 1024
)

# ── Disease descriptions ───────────────────────────────────────
DISEASE_INFO ={
    "Apple_scab": "fungal disease causing dark, scabby lesions on leaves and fruit",
    "Apple_black_rot": "fungal disease causing black rot on fruit and leaf spots",
    "Apple_cedar_rust": "fungal disease causing bright orange rust spots on leaves",
    "Apple_healthy": "no disease detected, plant appears healthy",


    "Potato_early_blight": "fungal disease causing dark concentric spots with yellow halos on leaves",
    "Potato_late_blight": "serious disease causing water-soaked lesions on leaves and tubers that spread rapidly",
    "Potato_healthy": "no disease detected, plant appears healthy",

    "Tomato_bacterial_spot": "bacterial disease causing small water-soaked spots on leaves and fruit",
    "Tomato_Septoria_leaf_spot": "fungal disease causing small circular spots with dark borders and gray centers on leaves",
    "Tomato_late_blight": "destructive disease causing dark water-soaked lesions on leaves and stems",
    "Tomato_healthy": "no disease detected, plant appears healthy"
}

# ── System prompt template ─────────────────────────────────────
SYSTEM_TEMPLATE = """You are an expert agricultural assistant helping farmers 
diagnose and treat plant diseases. You speak in simple, practical language 
that farmers can easily understand and act on.

Current diagnosis context:
- Crop        : {crop}
- Disease     : {disease_label}
- Description : {disease_info}
- Confidence  : {confidence}

Your responsibilities:
- Answer questions about this specific disease clearly and practically
- Give actionable treatment and prevention advice
- Suggest organic and chemical treatment options where relevant
- Recommend when to consult a local agronomist
- Keep responses concise — farmers need quick, actionable answers

Important rules:
- Only discuss topics related to {crop} plant health
- If asked something unrelated, politely redirect to the crop topic
- If confidence is below 70%, mention the diagnosis may be uncertain
- If the plant is healthy, reassure the farmer and give prevention tips
"""

# ── Chat prompt template ───────────────────────────────────────
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{message}")
])

# ── Initial diagnosis message ──────────────────────────────────
def get_initial_message(crop: str, disease: str,
                         disease_label: str, confidence: float) -> str:
    disease_info = DISEASE_INFO.get(disease, "unknown condition")
    is_healthy   = "healthy" in disease.lower()

    if is_healthy:
        return (
            f"Good news! Your {crop} plant appears **healthy** "
            f"({confidence:.1%} confidence). "
            f"I can answer any questions about keeping it healthy "
            f"or preventing diseases."
        )
    else:
        uncertainty = " Note that confidence is below 70%, so consider a second opinion." \
                      if confidence < 0.70 else ""
        return (
            f"I've detected **{disease_label}** in your {crop} plant "
            f"({confidence:.1%} confidence). "
            f"This is a {disease_info}.{uncertainty} "
            f"Ask me anything about treatment, causes, or prevention."
        )


# ── Convert history to Langchain messages ─────────────────────
def build_history(history: list) -> list:
    langchain_messages = []
    for msg in history:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    return langchain_messages


# ── Chat response ──────────────────────────────────────────────
def get_chat_response(crop: str, disease: str, disease_label: str,
                       confidence: float, message: str, history: list) -> str:
    disease_info = DISEASE_INFO.get(disease, "unknown condition")

    system_prompt = SYSTEM_TEMPLATE.format(
        crop         = crop,
        disease_label= disease_label,
        disease_info = disease_info,
        confidence   = f"{confidence:.1%}"
    )

    # Build chain using pipe operator — Langchain's modern syntax
    chain = prompt_template | llm

    response = chain.invoke({
        "system_prompt": system_prompt,
        "history":       build_history(history),
        "message":       message
    })

    return response.content