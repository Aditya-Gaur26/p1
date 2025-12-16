"""
Utility module for configuration management
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for the project"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.root_dir / "configs"
        
        # Paths
        self.data_dir = Path(os.getenv("DATA_DIR", "./data"))
        self.model_dir = Path(os.getenv("MODEL_DIR", "./models"))
        self.vectordb_dir = Path(os.getenv("VECTORDB_DIR", "./vectordb"))
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))
        
        # API Keys
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Model settings
        self.base_model_name = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model_upgraded = os.getenv("EMBEDDING_MODEL_UPGRADED", "sentence-transformers/all-mpnet-base-v2")
        
        # Feature flags
        self.enable_youtube = os.getenv("ENABLE_YOUTUBE_SUGGESTIONS", "true").lower() == "true"
        self.enable_papers = os.getenv("ENABLE_PAPER_SEARCH", "true").lower() == "true"
        self.enable_concepts = os.getenv("ENABLE_CONCEPT_MAPPING", "true").lower() == "true"
        
        # Enhanced RAG settings
        self.use_hybrid_retrieval = os.getenv("USE_HYBRID_RETRIEVAL", "true").lower() == "true"
        self.use_reranking = os.getenv("USE_RERANKING", "true").lower() == "true"
        self.use_query_expansion = os.getenv("USE_QUERY_EXPANSION", "true").lower() == "true"
        self.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache_size = int(os.getenv("CACHE_SIZE", "100"))
        
        # Load YAML configs
        self.training_config = self._load_yaml("training_config.yaml")
        self.model_config = self._load_yaml("model_config.yaml")
        self.api_config = self._load_yaml("api_config.yaml")
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        filepath = self.config_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.data_dir / "raw" / "slides",
            self.data_dir / "raw" / "books",
            self.data_dir / "raw" / "notes",
            self.data_dir / "processed",
            self.data_dir / "evaluation",
            self.model_dir / "base",
            self.model_dir / "fine_tuned",
            self.vectordb_dir / "course_materials",
            self.output_dir / "logs",
            self.output_dir / "results",
            self.output_dir / "responses",
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("✓ All directories created successfully")
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.training_config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.model_config
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.api_config


# Global config instance
config = Config()


if __name__ == "__main__":
    config.ensure_directories()
    print("\nConfiguration loaded successfully!")
    print(f"Root directory: {config.root_dir}")
    print(f"Data directory: {config.data_dir}")
    print(f"Model directory: {config.model_dir}")
