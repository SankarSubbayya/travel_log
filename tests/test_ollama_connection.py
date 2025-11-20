#!/usr/bin/env python3
"""Test Ollama connection and LLaVA model availability."""

import requests
import json

def test_ollama_connection():
    """Test connection to Ollama and check for LLaVA model."""
    ollama_url = "http://localhost:11434"

    print("Testing Ollama connection...")
    print(f"Ollama URL: {ollama_url}")
    print("-" * 50)

    try:
        # Check if Ollama is running
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        print(f"✅ Successfully connected to Ollama!")
        print(f"Found {len(models)} models:\n")

        llava_found = False
        for model in models:
            name = model.get("name", "")
            size = model.get("size", 0)
            size_gb = size / (1024**3)
            print(f"  - {name} ({size_gb:.1f} GB)")

            if "llava:7b" in name or name == "llava:7b":
                llava_found = True
                print(f"    ✅ This is the LLaVA model needed for caption generation!")

        print("-" * 50)
        if llava_found:
            print("✅ LLaVA model is ready for use!")
            print("\nYou can now run the Travel Log app and use caption generation:")
            print("  streamlit run app.py")
        else:
            print("⚠️ LLaVA model not found with exact name 'llava:7b'")
            print("But you have LLaVA models available that should work.")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama!")
        print("\nPlease ensure Ollama is running:")
        print("  ollama serve")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_ollama_connection()