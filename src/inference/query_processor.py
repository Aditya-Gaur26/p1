"""
Main query processor - combines RAG, model, and enrichment features
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.inference.rag_system import RAGSystem
from src.inference.model_loader import ModelLoader
from src.enrichment.youtube_suggester import YouTubeSuggester
from src.enrichment.paper_search import PaperSearch


class QueryProcessor:
    """Process queries with RAG, model inference, and enrichment"""
    
    def __init__(self, use_rag: bool = True, use_enrichment: bool = True):
        print("Initializing Query Processor...")
        
        self.use_rag = use_rag
        self.use_enrichment = use_enrichment
        
        # Load model
        self.model_loader = ModelLoader()
        self.model_loader.load(use_adapter=True)
        
        # Initialize RAG if enabled
        if self.use_rag:
            try:
                self.rag_system = RAGSystem()
            except Exception as e:
                print(f"⚠️  RAG system not available: {str(e)}")
                self.use_rag = False
                self.rag_system = None
        
        # Initialize enrichment features if enabled
        if self.use_enrichment:
            self.youtube_suggester = YouTubeSuggester() if config.enable_youtube else None
            self.paper_search = PaperSearch() if config.enable_papers else None
        else:
            self.youtube_suggester = None
            self.paper_search = None
        
        print("✓ Query Processor ready\n")
    
    def answer(self, question: str, detail_level: str = "normal", 
               include_enrichment: bool = True) -> Dict[str, Any]:
        """
        Answer a question with RAG and enrichment
        
        Args:
            question: The question to answer
            detail_level: "brief", "normal", or "comprehensive"
            include_enrichment: Whether to include YouTube/papers
        
        Returns:
            Dictionary with answer and enrichment data
        """
        
        start_time = time.time()
        result = {
            'question': question,
            'answer': '',
            'sources': [],
            'youtube_videos': [],
            'research_papers': [],
            'context_used': '',
            'processing_time': 0
        }
        
        # Step 1: Retrieve context using RAG
        context = ""
        sources = []
        
        if self.use_rag and self.rag_system:
            try:
                context, sources = self.rag_system.get_context(question, n_results=3)
                result['sources'] = sources
                result['context_used'] = context
                print(f"📚 Retrieved context from {len(sources)} sources")
            except Exception as e:
                print(f"⚠️  RAG retrieval failed: {str(e)}")
        
        # Step 2: Generate answer with model
        prompt = self.model_loader.format_prompt(question, context=context if context else None)
        
        # Adjust max tokens based on detail level
        max_tokens = {
            'brief': 150,
            'normal': 300,
            'comprehensive': 512
        }.get(detail_level, 300)
        
        print("🤖 Generating answer...")
        answer = self.model_loader.generate(prompt, max_new_tokens=max_tokens)
        result['answer'] = answer
        
        # Step 3: Get enrichment features (in parallel if possible)
        if include_enrichment and self.use_enrichment:
            print("🌟 Fetching enrichment features...")
            
            # YouTube suggestions
            if self.youtube_suggester:
                try:
                    videos = self.youtube_suggester.search_videos(question, max_results=3)
                    result['youtube_videos'] = videos
                    print(f"  ✓ Found {len(videos)} YouTube videos")
                except Exception as e:
                    print(f"  ⚠️  YouTube search failed: {str(e)}")
            
            # Research papers
            if self.paper_search:
                try:
                    papers = self.paper_search.search_papers(question, max_results=3)
                    result['research_papers'] = papers
                    print(f"  ✓ Found {len(papers)} research papers")
                except Exception as e:
                    print(f"  ⚠️  Paper search failed: {str(e)}")
        
        # Calculate processing time
        result['processing_time'] = time.time() - start_time
        
        return result
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format result as readable text"""
        
        output = []
        output.append("=" * 70)
        output.append(f"Question: {result['question']}")
        output.append("=" * 70)
        
        # Main answer
        output.append("\n📝 ANSWER:")
        output.append(result['answer'])
        
        # Sources
        if result['sources']:
            output.append("\n\n📚 SOURCES:")
            for i, source in enumerate(result['sources'], 1):
                output.append(f"  {i}. {source}")
        
        # YouTube videos
        if result['youtube_videos']:
            output.append("\n\n🎥 RELATED VIDEOS:")
            for i, video in enumerate(result['youtube_videos'], 1):
                output.append(f"  {i}. {video['title']}")
                output.append(f"     Channel: {video['channel']}")
                output.append(f"     URL: {video['url']}")
        
        # Research papers
        if result['research_papers']:
            output.append("\n\n📄 RESEARCH PAPERS:")
            for i, paper in enumerate(result['research_papers'], 1):
                output.append(f"  {i}. {paper['title']}")
                if 'authors' in paper and paper['authors']:
                    output.append(f"     Authors: {', '.join(paper['authors'][:2])}")
                output.append(f"     URL: {paper['url']}")
        
        # Processing time
        output.append(f"\n⏱️  Processing time: {result['processing_time']:.2f}s")
        output.append("=" * 70)
        
        return '\n'.join(output)


def interactive_mode(processor: QueryProcessor):
    """Run in interactive Q&A mode"""
    
    print("\n" + "=" * 70)
    print("Interactive Q&A Mode - Operating Systems & Networks".center(70))
    print("=" * 70)
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'help' for more options")
    print("=" * 70 + "\n")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Ask any question about Operating Systems or Networks")
                print("  - 'quit' or 'exit' to stop")
                continue
            
            # Process question
            print()
            result = processor.answer(question)
            
            # Display result
            print("\n" + processor.format_response(result))
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def single_question_mode(processor: QueryProcessor, question: str):
    """Answer a single question"""
    
    result = processor.answer(question)
    print(processor.format_response(result))


def main():
    parser = argparse.ArgumentParser(description="Query Processor for OS & Networks")
    parser.add_argument("--question", "-q", type=str, help="Single question to answer")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--no-enrichment", action="store_true", help="Disable enrichment features")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = QueryProcessor(
        use_rag=not args.no_rag,
        use_enrichment=not args.no_enrichment
    )
    
    # Run mode
    if args.question:
        single_question_mode(processor, args.question)
    elif args.interactive:
        interactive_mode(processor)
    else:
        # Default to interactive
        interactive_mode(processor)


if __name__ == "__main__":
    main()
