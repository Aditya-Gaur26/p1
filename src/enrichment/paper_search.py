"""
Research paper search using arXiv
"""

import sys
import arxiv
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class PaperSearch:
    """Search for relevant research papers"""
    
    def __init__(self):
        # Load config
        arxiv_config = config.get_api_config().get('arxiv', {})
        self.max_results = arxiv_config.get('max_results', 5)
        self.sort_by = arxiv.SortCriterion.Relevance
        self.categories = arxiv_config.get('categories', ['cs.OS', 'cs.NI', 'cs.DC'])
        
        print("✓ arXiv search initialized")
    
    def search_papers(self, query: str, max_results: int = None) -> List[Dict]:
        """Search for papers on arXiv"""
        
        if max_results is None:
            max_results = self.max_results
        
        try:
            # Enhance query with categories
            search_query = query
            
            # Add category filter
            category_filter = ' OR '.join([f'cat:{cat}' for cat in self.categories])
            full_query = f"({search_query}) AND ({category_filter})"
            
            # Search arXiv
            search = arxiv.Search(
                query=full_query,
                max_results=max_results * 2,  # Get more to filter
                sort_by=self.sort_by
            )
            
            papers = []
            
            for result in search.results():
                # Check if relevant
                relevance_score = self._calculate_relevance(result, query)
                
                if relevance_score > 0.3:  # Threshold for relevance
                    papers.append({
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'abstract': result.summary[:300] + '...',
                        'url': result.entry_id,
                        'pdf_url': result.pdf_url,
                        'published': result.published.strftime('%Y-%m-%d'),
                        'categories': result.categories,
                        'relevance_score': relevance_score
                    })
                
                if len(papers) >= max_results:
                    break
            
            # Sort by relevance
            papers.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return papers[:max_results]
        
        except Exception as e:
            print(f"⚠️  Error searching arXiv: {str(e)}")
            return self._get_fallback_papers(query)
    
    def _calculate_relevance(self, result, query: str) -> float:
        """Calculate relevance score for paper"""
        score = 0.0
        query_lower = query.lower()
        
        # Title match
        if query_lower in result.title.lower():
            score += 0.5
        
        # Abstract match
        if query_lower in result.summary.lower():
            score += 0.3
        
        # Category match
        relevant_categories = ['cs.OS', 'cs.NI', 'cs.DC', 'cs.PF']
        for cat in result.categories:
            if cat in relevant_categories:
                score += 0.2
                break
        
        return min(score, 1.0)
    
    def _get_fallback_papers(self, query: str) -> List[Dict]:
        """Get fallback paper suggestions"""
        
        # Classic papers for OS/Networks
        classic_papers = {
            'process': [
                {
                    'title': 'The Structure of the "THE"-Multiprogramming System',
                    'authors': ['E. W. Dijkstra'],
                    'abstract': 'Classic paper on multiprogramming and process management...',
                    'url': 'https://dl.acm.org/doi/10.1145/363095.363143',
                    'year': '1968'
                }
            ],
            'thread': [
                {
                    'title': 'Why Threads Are A Bad Idea (for most purposes)',
                    'authors': ['John Ousterhout'],
                    'abstract': 'Discussion on threading models and alternatives...',
                    'url': 'https://web.stanford.edu/~ouster/cgi-bin/papers/threads.pdf',
                    'year': '1996'
                }
            ],
            'tcp': [
                {
                    'title': 'Transmission Control Protocol (RFC 793)',
                    'authors': ['Jon Postel'],
                    'abstract': 'The original TCP specification...',
                    'url': 'https://www.rfc-editor.org/rfc/rfc793',
                    'year': '1981'
                }
            ],
            'network': [
                {
                    'title': 'End-to-End Arguments in System Design',
                    'authors': ['J. H. Saltzer', 'D. P. Reed', 'D. D. Clark'],
                    'abstract': 'Fundamental paper on network design principles...',
                    'url': 'https://dl.acm.org/doi/10.1145/357401.357402',
                    'year': '1984'
                }
            ],
        }
        
        # Find matching papers
        query_lower = query.lower()
        for keyword, papers in classic_papers.items():
            if keyword in query_lower:
                return papers
        
        return [{
            'title': f'Search "{query}" on arXiv',
            'url': f'https://arxiv.org/search/?query={query.replace(" ", "+")}&searchtype=all',
            'note': 'Manual search link provided'
        }]
    
    def format_paper_citation(self, paper: Dict) -> str:
        """Format paper as citation"""
        authors_str = ', '.join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors_str += ' et al.'
        
        year = paper.get('published', paper.get('year', 'n.d.'))[:4]
        
        return f"{authors_str}. ({year}). {paper['title']}."


def main():
    """Test paper search"""
    print("=" * 60)
    print("Testing Research Paper Search".center(60))
    print("=" * 60)
    
    searcher = PaperSearch()
    
    test_queries = [
        "process scheduling algorithms",
        "TCP congestion control",
        "distributed systems"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        papers = searcher.search_papers(query, max_results=3)
        
        print(f"Found {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"\n  {i}. {paper['title']}")
            if 'authors' in paper and paper['authors']:
                print(f"     Authors: {', '.join(paper['authors'][:2])}")
            print(f"     URL: {paper['url']}")
            if 'relevance_score' in paper:
                print(f"     Relevance: {paper['relevance_score']:.2f}")


if __name__ == "__main__":
    main()
