# claude_ai.py
import anthropic
from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def ask_claude(context: str) -> str:
    """
    Invia il contesto di mercato a Claude e ottiene una conferma.
    Ritorna: 'BUY', 'SELL' o 'NO_SIGNAL'
    """
    prompt = f"""
    Sei un trader professionista di Forex con 20 anni di esperienza. 
    Analizza il seguente contesto di mercato e determina se c'è un'inversione di trend valida.

    Contesto:
    {context}

    Considera:
    - Market Structure Shift (MSS)
    - Change of Character (CHOCH)
    - Volume
    - Livelli chiave (support/resistenza)
    - Divergenze
    - Order Block / Fair Value Gap

    Rispondi SOLO con una delle seguenti opzioni:
    - BUY: Se c'è un forte segnale rialzista
    - SELL: Se c'è un forte segnale ribassista
    - NO_SIGNAL: Se il segnale non è confermato o è debole
    """
    
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=300,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    response = message.content[0].text.strip().upper()
    return response if response in ["BUY", "SELL", "NO_SIGNAL"] else "NO_SIGNAL"
