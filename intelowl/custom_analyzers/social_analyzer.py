"""Social Media Analyzer for IntelOwl
Scrapes Instagram, Twitter, LinkedIn for OSINT data
Builds social network graphs and identifies patterns
"""

import os
import json
import re
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
from celery import shared_task
from api_app.analyzers_manager.observable_analyzers import ObservableAnalyzer
import logging
from neo4j import GraphDatabase
import tweepy
from instagram_private_api import Client as InstagramAPI
from linkedin_api import Linkedin
from bs4 import BeautifulSoup
import networkx as nx

logger = logging.getLogger(__name__)


class SocialMediaAnalyzer(ObservableAnalyzer):
    """Analyze social media profiles and networks"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Social media credentials
        self.instagram_user = os.getenv('INSTAGRAM_USERNAME')
        self.instagram_pass = os.getenv('INSTAGRAM_PASSWORD')
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.linkedin_user = os.getenv('LINKEDIN_USERNAME')
        self.linkedin_pass = os.getenv('LINKEDIN_PASSWORD')
        
        self.neo4j_driver = None
        
    def set_params(self, params):
        """Set analyzer parameters"""
        self.platforms = params.get('platforms', ['instagram', 'twitter', 'linkedin'])
        self.depth = params.get('depth', 2)  # Network crawl depth
        self.extract_images = params.get('extract_images', True)
        self.sentiment_analysis = params.get('sentiment_analysis', True)
        self.build_network = params.get('build_network', True)
        self.max_posts = params.get('max_posts', 100)
        
    def run(self):
        """Execute social media analysis"""
        observable = self.observable_name
        results = {
            'profiles': {},
            'posts': [],
            'network': {'nodes': [], 'edges': []},
            'images': [],
            'patterns': {},
            'risk_indicators': [],
            'timeline': []
        }
        
        try:
            # Search each platform
            if 'instagram' in self.platforms:
                ig_data = self._analyze_instagram(observable)
                results['profiles']['instagram'] = ig_data.get('profile', {})
                results['posts'].extend(ig_data.get('posts', []))
                results['images'].extend(ig_data.get('images', []))
                
            if 'twitter' in self.platforms:
                tw_data = self._analyze_twitter(observable)
                results['profiles']['twitter'] = tw_data.get('profile', {})
                results['posts'].extend(tw_data.get('tweets', []))
                
            if 'linkedin' in self.platforms:
                li_data = self._analyze_linkedin(observable)
                results['profiles']['linkedin'] = li_data.get('profile', {})
                
            # Build social network graph
            if self.build_network:
                results['network'] = self._build_social_network(results)
                
            # Analyze patterns
            results['patterns'] = self._analyze_patterns(results)
            
            # Risk assessment
            results['risk_indicators'] = self._assess_social_risks(results)
            
            # Build timeline
            results['timeline'] = self._build_timeline(results['posts'])
            
            # Store in Neo4j
            self._store_in_neo4j(observable, results)
            
        except Exception as e:
            logger.error(f"Social media analysis failed: {str(e)}")
            return {'error': str(e)}
            
        return results
        
    def _analyze_instagram(self, username: str) -> Dict:
        """Analyze Instagram profile"""
        data = {'profile': {}, 'posts': [], 'images': []}
        
        try:
            if self.instagram_user and self.instagram_pass:
                # Initialize Instagram API
                api = InstagramAPI(self.instagram_user, self.instagram_pass)
                
                # Get user info
                user_info = api.username_info(username)
                user_id = user_info['user']['pk']
                
                data['profile'] = {
                    'username': username,
                    'full_name': user_info['user'].get('full_name'),
                    'bio': user_info['user'].get('biography'),
                    'followers': user_info['user'].get('follower_count'),
                    'following': user_info['user'].get('following_count'),
                    'posts_count': user_info['user'].get('media_count'),
                    'is_private': user_info['user'].get('is_private'),
                    'is_verified': user_info['user'].get('is_verified'),
                    'profile_pic': user_info['user'].get('profile_pic_url'),
                    'external_url': user_info['user'].get('external_url')
                }
                
                # Get recent posts
                if not user_info['user'].get('is_private'):
                    feed = api.user_feed(user_id, max_id='')
                    
                    for item in feed.get('items', [])[:self.max_posts]:
                        post = {
                            'platform': 'instagram',
                            'id': item.get('id'),
                            'text': item.get('caption', {}).get('text', ''),
                            'timestamp': datetime.fromtimestamp(item.get('taken_at', 0)).isoformat(),
                            'likes': item.get('like_count', 0),
                            'comments': item.get('comment_count', 0),
                            'location': item.get('location', {}).get('name') if item.get('location') else None,
                            'hashtags': self._extract_hashtags(item.get('caption', {}).get('text', '')),
                            'mentions': self._extract_mentions(item.get('caption', {}).get('text', ''))
                        }
                        
                        data['posts'].append(post)
                        
                        # Extract images
                        if self.extract_images:
                            if item.get('image_versions2'):
                                for img in item['image_versions2'].get('candidates', []):
                                    data['images'].append({
                                        'url': img.get('url'),
                                        'width': img.get('width'),
                                        'height': img.get('height'),
                                        'post_id': item.get('id')
                                    })
                                    break
                                    
        except Exception as e:
            logger.error(f"Instagram analysis failed: {str(e)}")
            
        return data
        
    def _analyze_twitter(self, username: str) -> Dict:
        """Analyze Twitter profile"""
        data = {'profile': {}, 'tweets': []}
        
        try:
            if self.twitter_api_key and self.twitter_api_secret:
                # Initialize Twitter API
                auth = tweepy.OAuthHandler(self.twitter_api_key, self.twitter_api_secret)
                api = tweepy.API(auth)
                
                # Get user info
                user = api.get_user(screen_name=username)
                
                data['profile'] = {
                    'username': username,
                    'name': user.name,
                    'bio': user.description,
                    'followers': user.followers_count,
                    'following': user.friends_count,
                    'tweets_count': user.statuses_count,
                    'verified': user.verified,
                    'created_at': user.created_at.isoformat(),
                    'location': user.location,
                    'website': user.url
                }
                
                # Get recent tweets
                tweets = api.user_timeline(
                    screen_name=username,
                    count=self.max_posts,
                    include_rts=True,
                    tweet_mode='extended'
                )
                
                for tweet in tweets:
                    tweet_data = {
                        'platform': 'twitter',
                        'id': tweet.id_str,
                        'text': tweet.full_text,
                        'timestamp': tweet.created_at.isoformat(),
                        'likes': tweet.favorite_count,
                        'retweets': tweet.retweet_count,
                        'hashtags': [tag['text'] for tag in tweet.entities.get('hashtags', [])],
                        'mentions': [mention['screen_name'] for mention in tweet.entities.get('user_mentions', [])],
                        'urls': [url['expanded_url'] for url in tweet.entities.get('urls', [])],
                        'is_retweet': hasattr(tweet, 'retweeted_status'),
                        'reply_to': tweet.in_reply_to_screen_name
                    }
                    
                    data['tweets'].append(tweet_data)
                    
        except Exception as e:
            logger.error(f"Twitter analysis failed: {str(e)}")
            
        return data
        
    def _analyze_linkedin(self, username: str) -> Dict:
        """Analyze LinkedIn profile"""
        data = {'profile': {}}
        
        try:
            if self.linkedin_user and self.linkedin_pass:
                # Initialize LinkedIn API
                api = Linkedin(self.linkedin_user, self.linkedin_pass)
                
                # Get profile info
                profile = api.get_profile(username)
                
                data['profile'] = {
                    'username': username,
                    'full_name': f"{profile.get('firstName', '')} {profile.get('lastName', '')}",
                    'headline': profile.get('headline'),
                    'summary': profile.get('summary'),
                    'location': profile.get('locationName'),
                    'industry': profile.get('industryName'),
                    'connections': profile.get('numConnections'),
                    'current_company': None,
                    'education': [],
                    'skills': [],
                    'languages': []
                }
                
                # Extract current position
                for exp in profile.get('experience', []):
                    if not exp.get('endDate'):
                        data['profile']['current_company'] = {
                            'company': exp.get('companyName'),
                            'title': exp.get('title'),
                            'location': exp.get('locationName'),
                            'description': exp.get('description')
                        }
                        break
                        
                # Extract education
                for edu in profile.get('education', []):
                    data['profile']['education'].append({
                        'school': edu.get('schoolName'),
                        'degree': edu.get('degreeName'),
                        'field': edu.get('fieldOfStudy'),
                        'start': edu.get('startDate', {}).get('year'),
                        'end': edu.get('endDate', {}).get('year')
                    })
                    
                # Extract skills
                data['profile']['skills'] = [
                    skill.get('name') for skill in profile.get('skills', [])
                ]
                
        except Exception as e:
            logger.error(f"LinkedIn analysis failed: {str(e)}")
            
        return data
        
    def _build_social_network(self, results: Dict) -> Dict:
        """Build social network graph from collected data"""
        network = {'nodes': [], 'edges': []}
        
        # Create main profile nodes
        for platform, profile in results['profiles'].items():
            if profile:
                node_id = hashlib.md5(f"{platform}:{profile.get('username', '')}".encode()).hexdigest()
                network['nodes'].append({
                    'id': node_id,
                    'label': profile.get('username', ''),
                    'platform': platform,
                    'type': 'profile',
                    'followers': profile.get('followers', 0),
                    'verified': profile.get('verified', False)
                })
                
        # Extract mentioned users and create edges
        mentioned_users = set()
        for post in results['posts']:
            for mention in post.get('mentions', []):
                mentioned_users.add((post['platform'], mention))
                
        for platform, mention in mentioned_users:
            mention_id = hashlib.md5(f"{platform}:{mention}".encode()).hexdigest()
            network['nodes'].append({
                'id': mention_id,
                'label': mention,
                'platform': platform,
                'type': 'mentioned_user'
            })
            
            # Create edge
            main_profile_id = hashlib.md5(f"{platform}:{results['profiles'][platform].get('username', '')}".encode()).hexdigest()
            network['edges'].append({
                'source': main_profile_id,
                'target': mention_id,
                'type': 'mentions',
                'weight': sum(1 for p in results['posts'] if mention in p.get('mentions', []))
            })
            
        return network
        
    def _analyze_patterns(self, results: Dict) -> Dict:
        """Analyze posting patterns and behavior"""
        patterns = {
            'posting_times': {},
            'most_active_day': None,
            'most_active_hour': None,
            'common_hashtags': [],
            'common_mentions': [],
            'location_patterns': [],
            'sentiment_trend': None,
            'engagement_rate': 0
        }
        
        # Analyze posting times
        from collections import Counter
        days = []
        hours = []
        
        for post in results['posts']:
            if post.get('timestamp'):
                dt = datetime.fromisoformat(post['timestamp'])
                days.append(dt.strftime('%A'))
                hours.append(dt.hour)
                
        if days:
            day_counts = Counter(days)
            patterns['most_active_day'] = day_counts.most_common(1)[0][0]
            
        if hours:
            hour_counts = Counter(hours)
            patterns['most_active_hour'] = hour_counts.most_common(1)[0][0]
            
        # Common hashtags
        all_hashtags = []
        for post in results['posts']:
            all_hashtags.extend(post.get('hashtags', []))
            
        if all_hashtags:
            hashtag_counts = Counter(all_hashtags)
            patterns['common_hashtags'] = [tag for tag, _ in hashtag_counts.most_common(10)]
            
        # Common mentions
        all_mentions = []
        for post in results['posts']:
            all_mentions.extend(post.get('mentions', []))
            
        if all_mentions:
            mention_counts = Counter(all_mentions)
            patterns['common_mentions'] = [mention for mention, _ in mention_counts.most_common(10)]
            
        # Location patterns
        locations = [post.get('location') for post in results['posts'] if post.get('location')]
        if locations:
            location_counts = Counter(locations)
            patterns['location_patterns'] = [loc for loc, _ in location_counts.most_common(5)]
            
        # Calculate engagement rate
        total_posts = len(results['posts'])
        if total_posts > 0:
            total_engagement = sum(
                post.get('likes', 0) + post.get('comments', 0) + post.get('retweets', 0)
                for post in results['posts']
            )
            avg_followers = sum(
                profile.get('followers', 0) 
                for profile in results['profiles'].values() if profile
            ) / len(results['profiles'])
            
            if avg_followers > 0:
                patterns['engagement_rate'] = (total_engagement / total_posts) / avg_followers * 100
                
        return patterns
        
    def _assess_social_risks(self, results: Dict) -> List[Dict]:
        """Assess risks based on social media activity"""
        risks = []
        
        # Check for suspicious patterns
        patterns = results.get('patterns', {})
        
        # Unusual posting times (middle of night)
        if patterns.get('most_active_hour') in range(2, 6):
            risks.append({
                'type': 'unusual_activity',
                'description': f"Most active during {patterns['most_active_hour']}:00 (unusual hours)",
                'risk_level': 'MEDIUM'
            })
            
        # Low engagement rate might indicate bot
        if patterns.get('engagement_rate', 0) < 0.5:
            risks.append({
                'type': 'possible_bot',
                'description': 'Very low engagement rate',
                'risk_level': 'MEDIUM'
            })
            
        # Check for extremist content
        extremist_keywords = ['jihad', 'revolution', 'uprising', 'attack', 'bomb']
        for post in results['posts']:
            text = post.get('text', '').lower()
            for keyword in extremist_keywords:
                if keyword in text:
                    risks.append({
                        'type': 'extremist_content',
                        'description': f'Post contains concerning keyword: {keyword}',
                        'post_id': post.get('id'),
                        'risk_level': 'HIGH'
                    })
                    break
                    
        # Check for fake profile indicators
        for platform, profile in results['profiles'].items():
            if profile:
                # No profile picture
                if not profile.get('profile_pic'):
                    risks.append({
                        'type': 'fake_profile_indicator',
                        'description': f'No profile picture on {platform}',
                        'risk_level': 'LOW'
                    })
                    
                # Very new account with high activity
                if platform == 'twitter' and profile.get('created_at'):
                    created = datetime.fromisoformat(profile['created_at'])
                    age_days = (datetime.now() - created).days
                    if age_days < 30 and profile.get('tweets_count', 0) > 1000:
                        risks.append({
                            'type': 'suspicious_activity',
                            'description': 'New account with unusually high activity',
                            'risk_level': 'HIGH'
                        })
                        
        return risks
        
    def _build_timeline(self, posts: List[Dict]) -> List[Dict]:
        """Build chronological timeline of social media activity"""
        timeline = []
        
        # Sort posts by timestamp
        sorted_posts = sorted(
            posts,
            key=lambda x: datetime.fromisoformat(x['timestamp']) if x.get('timestamp') else datetime.min
        )
        
        for post in sorted_posts:
            timeline.append({
                'timestamp': post.get('timestamp'),
                'platform': post.get('platform'),
                'type': 'post',
                'content': post.get('text', '')[:200],  # First 200 chars
                'engagement': post.get('likes', 0) + post.get('comments', 0) + post.get('retweets', 0)
            })
            
        return timeline
        
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        return re.findall(r'#(\w+)', text)
        
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text"""
        return re.findall(r'@(\w+)', text)
        
    def _store_in_neo4j(self, username: str, results: Dict):
        """Store social media data in Neo4j"""
        try:
            if not self.neo4j_driver:
                self.neo4j_driver = GraphDatabase.driver(
                    os.getenv('NEO4J_URI'),
                    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
                )
                
            with self.neo4j_driver.session() as session:
                # Create profile nodes
                for platform, profile in results['profiles'].items():
                    if profile:
                        session.run("""
                            MERGE (p:SocialProfile {username: $username, platform: $platform})
                            SET p.full_name = $full_name,
                                p.bio = $bio,
                                p.followers = $followers,
                                p.following = $following,
                                p.verified = $verified,
                                p.last_analyzed = datetime()
                        """, username=profile.get('username', username),
                             platform=platform,
                             full_name=profile.get('full_name') or profile.get('name'),
                             bio=profile.get('bio'),
                             followers=profile.get('followers', 0),
                             following=profile.get('following', 0),
                             verified=profile.get('verified', False))
                        
                # Create post nodes
                for post in results['posts'][:50]:  # Limit for performance
                    session.run("""
                        MERGE (post:SocialPost {id: $id, platform: $platform})
                        SET post.text = $text,
                            post.timestamp = $timestamp,
                            post.likes = $likes,
                            post.engagement = $engagement
                        MERGE (p:SocialProfile {username: $username, platform: $platform})
                        MERGE (p)-[:POSTED]->(post)
                    """, id=post['id'],
                         platform=post['platform'],
                         text=post.get('text', '')[:500],
                         timestamp=post.get('timestamp'),
                         likes=post.get('likes', 0),
                         engagement=post.get('likes', 0) + post.get('comments', 0),
                         username=username)
                         
        except Exception as e:
            logger.error(f"Neo4j storage failed: {str(e)}")
            
        finally:
            if self.neo4j_driver:
                self.neo4j_driver.close()
                
    @classmethod
    def _monkeypatch(cls):
        """Register analyzer with IntelOwl"""
        patches = [
            {
                'model': 'analyzers_manager.AnalyzerConfig',
                'name': 'SocialMediaAnalyzer',
                'description': 'Analyze social media profiles and networks',
                'python_module': 'custom_analyzers.social_analyzer.SocialMediaAnalyzer',
                'disabled': False,
                'type': 'observable',
                'docker_based': False,
                'maximum_tlp': 'RED',
                'observable_supported': ['generic', 'username'],
                'supported_filetypes': [],
                'run_hash': False,
                'run_hash_type': '',
                'not_supported_filetypes': [],
                'parameters': {
                    'platforms': {
                        'type': 'list',
                        'description': 'Platforms to analyze',
                        'default': ['instagram', 'twitter', 'linkedin']
                    },
                    'depth': {
                        'type': 'int',
                        'description': 'Network crawl depth',
                        'default': 2
                    },
                    'max_posts': {
                        'type': 'int',
                        'description': 'Maximum posts to analyze',
                        'default': 100
                    }
                }
            }
        ]
        return patches
