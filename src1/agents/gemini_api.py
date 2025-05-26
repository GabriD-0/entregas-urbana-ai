"""
Camada fininha para interagir com a API Gemini. Mantém tudo opcional
— se não houver chave, apenas imprime no console.
"""
import os
import json

try:
    import google.generativeai as genai
    GEN_AVAILABLE = True
except ImportError:
    GEN_AVAILABLE = False

def _init():
    if GEN_AVAILABLE and "GEMINI_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        return genai.GenerativeModel("gemini-1.0-pro")
    return None

_MODEL = _init()

def summarise_route(agent_id: str, path):
    prompt = (f"Agente {agent_id} percorreu {len(path)} células: {path}.\n"
              f"Escreva um resumo de até 2 linhas destacando o trajeto e "
              f"possíveis obstáculos superados.")
    if _MODEL:
        resp = _MODEL.generate_content(prompt)
        print(f"[Gemini] {resp.text}")
        return resp.text
    else:
        print(f"[DEBUG-Gemini-OFF] {prompt[:80]}...")
        return ""
