"""
Quick setup helper for advanced dataset creation
"""

import os
import sys
from pathlib import Path


def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...\n")
    
    issues = []
    
    # Check processed books exist
    books_file = Path("data/processed/books/all_pdfs_combined.json")
    if not books_file.exists():
        issues.append("❌ Processed books not found. Run: python src/data_processing/extract_pdfs.py")
    else:
        print("✓ Processed books found")
    
    # Check for API keys (optional)
    providers = {
        "OpenAI (gpt-4o-mini)": "OPENAI_API_KEY",
        "Groq (FREE, llama-3.3-70b)": "GROQ_API_KEY",
        "Anthropic (claude)": "ANTHROPIC_API_KEY"
    }
    
    print("\n🔑 Checking API keys:")
    available_providers = []
    
    for provider, key_name in providers.items():
        if os.getenv(key_name):
            print(f"✓ {provider}: Found")
            available_providers.append(provider.split('(')[0].strip())
        else:
            print(f"○ {provider}: Not set (optional)")
    
    # Check for Ollama
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Ollama: Available (local, free)")
            available_providers.append("Ollama")
    except:
        print(f"○ Ollama: Not installed (optional)")
    
    print("\n" + "="*70)
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    
    if not available_providers:
        print("\n⚠️  No LLM providers available!")
        print("\nRecommended: Use Groq (FREE)")
        print("1. Visit: https://console.groq.com")
        print("2. Create account and get API key")
        print("3. Set key: $env:GROQ_API_KEY = 'gsk_...'")
        print("4. Run this script again")
        return False
    
    print(f"\n✅ Ready to generate dataset!")
    print(f"\nAvailable providers: {', '.join(available_providers)}")
    
    return True


def recommend_provider():
    """Recommend best available provider"""
    
    if os.getenv("GROQ_API_KEY"):
        return "groq", "Groq (FREE + FAST)", "llama-3.3-70b-versatile"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai", "OpenAI (HIGH QUALITY, ~$1-2)", "gpt-4o-mini"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic", "Anthropic (PREMIUM)", "claude-3-5-sonnet-20241022"
    else:
        # Check Ollama
        try:
            import subprocess
            subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
            return "ollama", "Ollama (LOCAL, FREE, SLOW)", "llama3.2"
        except:
            return None, None, None


def main():
    print("="*70)
    print("Advanced Dataset Creator - Quick Setup".center(70))
    print("="*70)
    print()
    
    if not check_requirements():
        return
    
    provider, desc, model = recommend_provider()
    
    if not provider:
        print("\n❌ No LLM provider available. See guide above.")
        return
    
    print("\n" + "="*70)
    print(f"📋 Recommended: {desc}")
    print("="*70)
    
    print(f"\n🧪 To test with 10 chunks (2-5 minutes):")
    print(f"   python src/data_processing/advanced_dataset_creator.py --provider {provider} --test")
    
    print(f"\n🚀 To generate full dataset (30-60 minutes):")
    print(f"   python src/data_processing/advanced_dataset_creator.py --provider {provider}")
    
    print(f"\n� Note: New dataset will be saved as train_llm.jsonl (keeps old dataset intact)")
    print(f"\n�📖 For more options, see: ADVANCED_DATASET_GUIDE.md")
    
    # Interactive mode
    print("\n" + "="*70)
    response = input("\nRun TEST mode now? (y/n): ").strip().lower()
    
    if response == 'y':
        import subprocess
        print("\n🧪 Running test with 10 chunks...\n")
        subprocess.run([
            sys.executable,
            "src/data_processing/advanced_dataset_creator.py",
            "--provider", provider,
            "--test"
        ])


if __name__ == "__main__":
    main()
