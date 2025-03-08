import requests
from bs4 import BeautifulSoup
import json
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import logging

class WebAccessManager:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    async def search_web(self, query: str, search_type: str = "general") -> Optional[Dict[str, Any]]:
        """Perform a web search using SearchAPI.io"""
        try:
            url = "https://www.searchapi.io/api/v1/search"
            
            # Clean and enhance query based on type
            location_terms = ["in", "for", "at"]
            clean_query = query.lower()
            for term in location_terms:
                if f" {term} " in clean_query:
                    location = clean_query.split(f" {term} ")[1].strip()
                    break
            else:
                location = ""

            # Modify query based on type
            if search_type == "weather":
                if location:
                    modified_query = f"current weather forecast {location} conditions temperature today"
                else:
                    modified_query = f"current weather forecast local conditions temperature today"
            
            elif search_type == "news":
                # Create multiple search variants to get diverse results
                search_variants = []
                if location:
                    search_variants = [
                        f"breaking news {location} today -site:news.google.com",
                        f"latest news headlines {location} today -site:news.google.com",
                        f"top news stories {location} now -site:news.google.com"
                    ]
                else:
                    search_variants = [
                        "breaking news worldwide today -site:news.google.com",
                        "latest news headlines today -site:news.google.com",
                        "top news stories now -site:news.google.com"
                    ]
                
                # Perform multiple searches and combine results
                all_results = {'top_stories': [], 'organic_results': []}
                for variant in search_variants:
                    params = {
                        "api_key": self.session.params["api_key"],
                        "q": variant,
                        "engine": "google",
                        "num": 5,
                        "gl": "us",
                        "hl": "en"
                    }
                    
                    response = self.session.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    variant_results = response.json()
                    
                    all_results['top_stories'].extend(variant_results.get('top_stories', []))
                    all_results['organic_results'].extend(variant_results.get('organic_results', []))
                
                # Remove duplicates based on URL
                seen_urls = set()
                unique_results = []
                for result in all_results['organic_results']:
                    url = result.get('link', '')
                    if url not in seen_urls:
                        seen_urls.add(url)
                        unique_results.append(result)
                
                all_results['organic_results'] = unique_results
                results = all_results
                
            else:
                # Standard single search for other types
                params = {
                    "api_key": self.session.params["api_key"],
                    "q": modified_query,
                    "engine": "google",
                    "num": 10,
                    "gl": "us",
                    "hl": "en"
                }
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                results = response.json()

            # Filter and clean results
            if search_type == "news":
                # Keep only recent and relevant news
                results['organic_results'] = [
                    r for r in results.get('organic_results', [])
                    if not any(term in r.get('title', '').lower() 
                             for term in ['old news', 'archive', 'history']) and
                    (r.get('date', '').endswith('ago') or 'today' in r.get('date', '').lower())
                ]

            formatted_results = {
                'query': query,
                'search_type': search_type,
                'location': location,
                'top_stories': results.get('top_stories', []),
                'organic_results': results.get('organic_results', []),
                'search_metadata': {
                    'total_results': len(results.get('organic_results', [])),
                    'time_taken': 0
                }
            }

            return formatted_results

        except Exception as e:
            logging.error(f"Search API error: {str(e)}")
            if isinstance(e, requests.exceptions.RequestException):
                logging.error(f"Request error details: {e.response.text if hasattr(e, 'response') else 'No response'}")
            return None

    async def process_web_query(self, query: str, search_type: str = "general") -> str:
        """Process queries that require web access"""
        try:
            results = await self.search_web(query, search_type)
            if not results:
                return f"I'm sorry, I couldn't find any information about {query}."

            response_lines = []
            
            if search_type == "news":
                # Combine and sort all news stories by date
                all_stories = []
                
                # Add top stories
                if results.get('top_stories'):
                    for story in results['top_stories']:
                        if not story.get('snippet'): continue
                        title = story.get('title', '').split('|')[0].split('-')[0].strip()
                        snippet = story.get('snippet', '').strip()
                        source = story.get('source', '')
                        date = story.get('date', '')
                        if title and snippet:
                            all_stories.append({
                                'title': title,
                                'snippet': snippet,
                                'source': source,
                                'date': date,
                                'is_top': True
                            })

                # Add organic results
                for result in results.get('organic_results', []):
                    if not result.get('snippet'): continue
                    title = result.get('title', '').split('|')[0].split('-')[0].strip()
                    snippet = result.get('snippet', '').strip()
                    source = result.get('source', '')
                    date = result.get('date', '')
                    if title and snippet:
                        all_stories.append({
                            'title': title,
                            'snippet': snippet,
                            'source': source,
                            'date': date,
                            'is_top': False
                        })

                # Remove duplicates and filter irrelevant content
                seen_titles = set()
                unique_stories = []
                for story in all_stories:
                    if story['title'] not in seen_titles and not any(term in story['title'].lower() for term in ['old news', 'archive', 'history']):
                        seen_titles.add(story['title'])
                        unique_stories.append(story)

                # Sort by date (prioritizing top stories and recent news)
                def sort_key(story):
                    date = story['date'].lower()
                    if story['is_top']:
                        return (0, date)
                    if 'minute' in date or 'hour' in date:
                        return (1, date)
                    if 'today' in date:
                        return (2, date)
                    return (3, date)

                unique_stories.sort(key=sort_key)

                # Format the response
                response_lines.append("\nLatest News:")
                for story in unique_stories[:5]:  # Limit to top 5 stories
                    response_lines.append(f"• {story['title']}")
                    response_lines.append(f"  {story['snippet']}")
                    if story['source'] and story['date']:
                        response_lines.append(f"  Source: {story['source']}, {story['date']}")
                    response_lines.append("")
            
            else:
                # Handle other types (weather, scores, etc.)
                # [Previous handling code remains the same]
                pass

            formatted_response = "\n".join(response_lines)
            
            if not formatted_response.strip():
                return "I found some information but couldn't extract the specific details. Could you try rephrasing your question?"

            return formatted_response

        except Exception as e:
            logging.error(f"Error in process_web_query: {str(e)}")
            return f"I encountered an error while searching. Please try again."