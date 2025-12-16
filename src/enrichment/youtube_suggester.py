"""
YouTube video suggester
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class YouTubeSuggester:
    """Suggest relevant YouTube videos for topics"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.youtube_api_key
        
        if not self.api_key or self.api_key == "your_youtube_api_key_here":
            print("⚠️  YouTube API key not configured")
            self.youtube = None
        else:
            try:
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
                print("✓ YouTube API initialized")
            except Exception as e:
                print(f"⚠️  YouTube API initialization failed: {str(e)}")
                self.youtube = None
        
        # Load config
        youtube_config = config.get_api_config().get('youtube', {})
        self.max_results = youtube_config.get('max_results', 5)
        self.order_by = youtube_config.get('order_by', 'relevance')
        self.min_views = youtube_config.get('filters', {}).get('min_views', 1000)
    
    def search_videos(self, query: str, max_results: int = None) -> List[Dict]:
        """Search for YouTube videos"""
        
        if self.youtube is None:
            return self._get_fallback_suggestions(query)
        
        if max_results is None:
            max_results = self.max_results
        
        try:
            # Enhance query for educational content
            search_query = f"{query} tutorial explanation"
            
            # Search for videos
            search_response = self.youtube.search().list(
                q=search_query,
                type='video',
                part='id,snippet',
                maxResults=max_results * 2,  # Get more to filter
                order=self.order_by,
                relevanceLanguage='en',
                safeSearch='strict'
            ).execute()
            
            videos = []
            
            for item in search_response.get('items', []):
                video_id = item['id']['videoId']
                snippet = item['snippet']
                
                # Get video statistics
                stats = self._get_video_stats(video_id)
                
                # Filter by views
                if stats and int(stats.get('viewCount', 0)) >= self.min_views:
                    videos.append({
                        'title': snippet['title'],
                        'channel': snippet['channelTitle'],
                        'description': snippet['description'][:200] + '...',
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'thumbnail': snippet['thumbnails']['medium']['url'],
                        'published': snippet['publishedAt'],
                        'views': int(stats.get('viewCount', 0)),
                        'likes': int(stats.get('likeCount', 0)),
                        'relevance_score': self._calculate_relevance(snippet, query)
                    })
                
                if len(videos) >= max_results:
                    break
            
            # Sort by relevance score
            videos.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return videos[:max_results]
        
        except HttpError as e:
            print(f"⚠️  YouTube API error: {str(e)}")
            return self._get_fallback_suggestions(query)
        except Exception as e:
            print(f"⚠️  Error searching YouTube: {str(e)}")
            return self._get_fallback_suggestions(query)
    
    def _get_video_stats(self, video_id: str) -> Dict:
        """Get video statistics"""
        try:
            stats_response = self.youtube.videos().list(
                part='statistics',
                id=video_id
            ).execute()
            
            if stats_response['items']:
                return stats_response['items'][0]['statistics']
        except:
            pass
        
        return {}
    
    def _calculate_relevance(self, snippet: Dict, query: str) -> float:
        """Calculate relevance score for video"""
        score = 0.0
        query_lower = query.lower()
        
        # Title match
        if query_lower in snippet['title'].lower():
            score += 0.5
        
        # Description match
        if query_lower in snippet['description'].lower():
            score += 0.3
        
        # Educational keywords
        educational_keywords = ['tutorial', 'lecture', 'course', 'explained', 'introduction', 'guide']
        for keyword in educational_keywords:
            if keyword in snippet['title'].lower() or keyword in snippet['description'].lower():
                score += 0.1
                break
        
        # Quality channels (example - you can customize)
        quality_channels = ['Neso Academy', 'Gate Smashers', 'Abdul Bari', 'MIT OpenCourseWare']
        if snippet['channelTitle'] in quality_channels:
            score += 0.3
        
        return min(score, 1.0)
    
    def _get_fallback_suggestions(self, query: str) -> List[Dict]:
        """Get fallback suggestions when API is not available"""
        
        # Predefined quality videos for common OS/Networks topics
        fallback_videos = {
            'process': [
                {'title': 'Process Management in Operating Systems', 
                 'channel': 'Neso Academy', 
                 'url': 'https://www.youtube.com/watch?v=OrM7nZcxXZU'},
            ],
            'thread': [
                {'title': 'Process vs Thread', 
                 'channel': 'Gate Smashers', 
                 'url': 'https://www.youtube.com/watch?v=O3EyzlZxx3g'},
            ],
            'scheduling': [
                {'title': 'CPU Scheduling Algorithms', 
                 'channel': 'Abdul Bari', 
                 'url': 'https://www.youtube.com/watch?v=EWkQl0n0w5M'},
            ],
            'deadlock': [
                {'title': 'Deadlocks in Operating Systems', 
                 'channel': 'Neso Academy', 
                 'url': 'https://www.youtube.com/watch?v=UVo9mGARkhQ'},
            ],
            'memory': [
                {'title': 'Memory Management in OS', 
                 'channel': 'Gate Smashers', 
                 'url': 'https://www.youtube.com/watch?v=qdkxXygc3rE'},
            ],
            'tcp': [
                {'title': 'TCP/IP Protocol Explained', 
                 'channel': 'PowerCert Animated Videos', 
                 'url': 'https://www.youtube.com/watch?v=PpsEaqJV_A0'},
            ],
            'network': [
                {'title': 'Computer Networks Full Course', 
                 'channel': 'Gate Smashers', 
                 'url': 'https://www.youtube.com/watch?v=JFF2vJaN0Cw'},
            ],
        }
        
        # Find matching videos
        query_lower = query.lower()
        for keyword, videos in fallback_videos.items():
            if keyword in query_lower:
                return videos
        
        return [{
            'title': f'Search "{query}" on YouTube',
            'channel': 'Manual Search',
            'url': f'https://www.youtube.com/results?search_query={query.replace(" ", "+")}+operating+systems+networks',
            'note': 'API key not configured - manual search link provided'
        }]


def main():
    """Test YouTube suggester"""
    print("=" * 60)
    print("Testing YouTube Suggester".center(60))
    print("=" * 60)
    
    suggester = YouTubeSuggester()
    
    test_queries = [
        "process scheduling",
        "TCP protocol",
        "virtual memory"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        videos = suggester.search_videos(query, max_results=3)
        
        print(f"Found {len(videos)} videos:")
        for i, video in enumerate(videos, 1):
            print(f"\n  {i}. {video['title']}")
            print(f"     Channel: {video['channel']}")
            print(f"     URL: {video['url']}")
            if 'views' in video:
                print(f"     Views: {video['views']:,}")


if __name__ == "__main__":
    main()
