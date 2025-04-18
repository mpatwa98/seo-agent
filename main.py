# blog_agent.py
import os
import json
import time
import asyncio
import argparse
import functools
import datetime
from typing import Dict, List
import logging
import aiohttp
from google import genai
from dotenv import load_dotenv
import readability

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("blog_agent")

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

# Configure Google Gemini
if GEMINI_API_KEY:
    genai.Client(api_key=GEMINI_API_KEY)
    gemini_model = genai.("gemini-pro")
else:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("Missing API key for Google Gemini")


class TopicAnalysisAgent:
    """Agent responsible for analyzing the topic and determining content strategy."""

    @staticmethod
    async def analyze_topic(topic: str, tone: str = "educational") -> Dict:
        """
        Analyzes a topic to determine subtopics and content strategy.

        Args:
            topic: The main blog topic
            tone: Desired tone for the blog (educational, formal, creative, etc.)

        Returns:
            Dict with analyzed topic information
        """
        logger.info(f"Analyzing topic: {topic} with tone: {tone}")

        # Create Gemini model instance
        model = genai.GenerativeModel("gemini-pro")

        prompt = f"""
        As a blog content strategist, analyze the topic "{topic}" and provide the following:
        
        1. A comprehensive breakdown of 5-7 subtopics that would make good H2 headings
        2. The target audience for this content
        3. The level of technical depth appropriate for this topic
        4. 10 keywords that would be relevant for SEO
        
        Format your response as a JSON object with the following structure:
        {{
            "main_topic": "The main topic",
            "subtopics": ["Subtopic 1", "Subtopic 2", ...],
            "target_audience": "Description of target audience",
            "technical_depth": "beginner|intermediate|advanced",
            "tone": "{tone}",
            "keywords": ["keyword1", "keyword2", ...]
        }}
        
        Only return the JSON object, no explanations.
        """

        try:
            response = await asyncio.to_thread(
                lambda: model.generate_content(prompt).text
            )

            # Extract JSON from response
            try:
                # Find JSON within the response if it's not pure JSON
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    result = json.loads(response)

                # Add tone if not present
                if "tone" not in result:
                    result["tone"] = tone

                logger.info(
                    f"Topic analysis complete: identified {len(result['subtopics'])} subtopics"
                )
                return result

            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from Gemini response: {response}")
                # Fallback handling
                return {
                    "main_topic": topic,
                    "subtopics": [
                        f"Aspect of {topic} 1",
                        f"Aspect of {topic} 2",
                        f"Aspect of {topic} 3",
                    ],
                    "target_audience": "General audience interested in this topic",
                    "technical_depth": "intermediate",
                    "tone": tone,
                    "keywords": [topic.lower()]
                    + [
                        f"{topic.lower()} {suffix}"
                        for suffix in ["guide", "tutorial", "explained", "examples"]
                    ],
                }

        except Exception as e:
            logger.error(f"Error in topic analysis: {str(e)}")
            raise


class ResearchAgent:
    """Agent responsible for gathering research data from various sources."""

    # Cache for API responses
    _news_cache = {}
    _keywords_cache = {}
    _quotes_cache = {}

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def get_cached_keywords(keyword: str) -> List[Dict]:
        """Cached function to get related keywords."""
        return ResearchAgent._keywords_cache.get(keyword, [])

    @staticmethod
    async def get_news_articles(topic: str, max_results: int = 5) -> List[Dict]:
        """
        Fetch recent news articles related to the topic using NewsData.io API.

        Args:
            topic: The topic to search for
            max_results: Maximum number of articles to return

        Returns:
            List of article dictionaries
        """
        # Check cache first
        cache_key = f"{topic}_{max_results}"
        if cache_key in ResearchAgent._news_cache:
            logger.info(f"Using cached news results for: {topic}")
            return ResearchAgent._news_cache[cache_key]

        if not NEWSDATA_API_KEY:
            logger.warning("NewsData API key not found, using mock data")
            # Return mock data
            mock_data = [
                {
                    "title": f"Recent Developments in {topic}",
                    "description": f"This article discusses the latest advancements in {topic}.",
                    "url": "https://example.com/news/1",
                    "published_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                },
                {
                    "title": f"How {topic} is Changing Industries",
                    "description": f"An analysis of how {topic} is transforming various sectors.",
                    "url": "https://example.com/news/2",
                    "published_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                },
            ]
            ResearchAgent._news_cache[cache_key] = mock_data
            return mock_data

        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": NEWSDATA_API_KEY,
            "q": topic,
            "language": "en",
            "size": max_results,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data.get("status") == "success":
                            results = []
                            for article in data.get("results", [])[:max_results]:
                                results.append(
                                    {
                                        "title": article.get("title"),
                                        "description": article.get("description"),
                                        "url": article.get("link"),
                                        "published_date": article.get("pubDate"),
                                    }
                                )

                            # Cache the results
                            ResearchAgent._news_cache[cache_key] = results
                            logger.info(
                                f"Retrieved {len(results)} news articles for: {topic}"
                            )
                            return results
                        else:
                            logger.warning(f"NewsData API returned error: {data}")
                            return []
                    else:
                        logger.error(f"NewsData API HTTP error: {response.status}")
                        # Implement retry logic
                        return await ResearchAgent._retry_news_request(
                            topic, max_results
                        )
        except Exception as e:
            logger.error(f"Error fetching news articles: {str(e)}")
            return []

    @staticmethod
    async def _retry_news_request(
        topic: str, max_results: int = 5, max_retries: int = 3
    ) -> List[Dict]:
        """Retry logic for NewsData API requests."""
        for i in range(max_retries):
            logger.info(f"Retrying NewsData API request ({i + 1}/{max_retries})")
            try:
                # Wait before retrying (exponential backoff)
                await asyncio.sleep(2**i)

                url = "https://newsdata.io/api/1/news"
                params = {
                    "apikey": NEWSDATA_API_KEY,
                    "q": topic,
                    "language": "en",
                    "size": max_results,
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()

                            if data.get("status") == "success":
                                results = []
                                for article in data.get("results", [])[:max_results]:
                                    results.append(
                                        {
                                            "title": article.get("title"),
                                            "description": article.get("description"),
                                            "url": article.get("link"),
                                            "published_date": article.get("pubDate"),
                                        }
                                    )

                                # Cache the results
                                cache_key = f"{topic}_{max_results}"
                                ResearchAgent._news_cache[cache_key] = results
                                return results
            except Exception as e:
                logger.error(f"Error in retry attempt {i + 1}: {str(e)}")

        # Return empty list if all retries fail
        return []

    @staticmethod
    async def get_related_keywords(keyword: str, max_results: int = 15) -> List[Dict]:
        """
        Find related keywords using Datamuse API.

        Args:
            keyword: The base keyword
            max_results: Maximum number of related keywords to return

        Returns:
            List of related keywords with scores
        """
        # Check cache first
        cache_key = f"{keyword}_{max_results}"
        if cache_key in ResearchAgent._keywords_cache:
            logger.info(f"Using cached keyword results for: {keyword}")
            return ResearchAgent._keywords_cache[cache_key]

        url = "https://api.datamuse.com/words"
        params = {
            "ml": keyword,  # Words with similar meaning
            "max": max_results,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Cache the results
                        ResearchAgent._keywords_cache[cache_key] = data
                        logger.info(
                            f"Retrieved {len(data)} related keywords for: {keyword}"
                        )
                        return data
                    else:
                        logger.error(f"Datamuse API HTTP error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching related keywords: {str(e)}")
            return []

    @staticmethod
    async def get_quotes(keyword: str, max_results: int = 3) -> List[Dict]:
        """
        Fetch relevant quotes using Quotable.io API.

        Args:
            keyword: The keyword to search for quotes
            max_results: Maximum number of quotes to return

        Returns:
            List of quote dictionaries
        """
        # Check cache first
        cache_key = f"{keyword}_{max_results}"
        if cache_key in ResearchAgent._quotes_cache:
            logger.info(f"Using cached quotes for: {keyword}")
            return ResearchAgent._quotes_cache[cache_key]

        # Quotable.io doesn't have a direct search by keyword feature,
        # so we'll query by tags that might relate to our keyword
        url = "https://api.quotable.io/quotes/random"
        params = {"tags": keyword, "limit": max_results}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Cache the results
                        ResearchAgent._quotes_cache[cache_key] = data
                        logger.info(f"Retrieved {len(data)} quotes for: {keyword}")
                        return data
                    elif response.status == 404:
                        # Try a more general approach with no tags
                        async with session.get(
                            "https://api.quotable.io/quotes/random",
                            params={"limit": max_results},
                        ) as fallback_response:
                            if fallback_response.status == 200:
                                data = await fallback_response.json()
                                # Cache the results
                                ResearchAgent._quotes_cache[cache_key] = data
                                logger.info(
                                    f"Retrieved {len(data)} general quotes (no keyword match)"
                                )
                                return data
                            else:
                                logger.error(
                                    f"Quotable API fallback HTTP error: {fallback_response.status}"
                                )
                                return []
                    else:
                        logger.error(f"Quotable API HTTP error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching quotes: {str(e)}")
            return []

    @staticmethod
    async def gather_research(topic_analysis: Dict) -> Dict:
        """
        Gather all research materials needed for the blog post.

        Args:
            topic_analysis: The output from TopicAnalysisAgent.analyze_topic()

        Returns:
            Dict with all research materials
        """
        main_topic = topic_analysis["main_topic"]
        keywords = topic_analysis["keywords"]

        # Run multiple API calls concurrently
        tasks = [
            ResearchAgent.get_news_articles(main_topic),
            ResearchAgent.get_quotes(main_topic),
        ]

        # Add keyword research tasks
        for keyword in keywords[:3]:  # Limit to top 3 keywords
            tasks.append(ResearchAgent.get_related_keywords(keyword))

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling any exceptions
        news_articles = results[0] if not isinstance(results[0], Exception) else []
        quotes = results[1] if not isinstance(results[1], Exception) else []

        # Process keyword results
        keyword_results = {}
        for i, keyword in enumerate(keywords[:3]):
            if i + 2 < len(results) and not isinstance(results[i + 2], Exception):
                keyword_results[keyword] = results[i + 2]
            else:
                keyword_results[keyword] = []

        # Flatten all related keywords into a single list
        all_related_keywords = []
        for keyword_list in keyword_results.values():
            all_related_keywords.extend(keyword_list)

        # Sort by score (if available) and get unique words
        unique_keywords = {}
        for kw in all_related_keywords:
            word = kw.get("word")
            score = kw.get("score", 0)
            if word and word not in unique_keywords:
                unique_keywords[word] = score

        # Create sorted list of unique keywords
        sorted_keywords = sorted(
            unique_keywords.items(), key=lambda x: x[1], reverse=True
        )
        related_keywords = [
            {"word": word, "score": score} for word, score in sorted_keywords[:20]
        ]

        logger.info(
            f"Research complete: {len(news_articles)} news articles, {len(quotes)} quotes, {len(related_keywords)} related keywords"
        )

        return {
            "news_articles": news_articles,
            "quotes": quotes,
            "related_keywords": related_keywords,
        }


class ContentGenerationAgent:
    """Agent responsible for generating the blog content."""

    @staticmethod
    async def generate_blog_post(topic_analysis: Dict, research_data: Dict) -> Dict:
        """
        Generate a complete blog post based on the topic analysis and research data.

        Args:
            topic_analysis: The output from TopicAnalysisAgent.analyze_topic()
            research_data: The output from ResearchAgent.gather_research()

        Returns:
            Dict with the generated blog content
        """
        # Extract relevant information
        main_topic = topic_analysis["main_topic"]
        subtopics = topic_analysis["subtopics"]
        tone = topic_analysis.get("tone", "educational")
        technical_depth = topic_analysis.get("technical_depth", "intermediate")
        target_audience = topic_analysis.get("target_audience", "General audience")

        # Extract research data
        news_articles = research_data.get("news_articles", [])
        quotes = research_data.get("quotes", [])
        related_keywords = research_data.get("related_keywords", [])

        # Create Gemini model instance
        model = genai.GenerativeModel("gemini-pro")

        # Generate a compelling title
        title = await ContentGenerationAgent._generate_title(model, main_topic, tone)

        # Generate an introduction
        introduction = await ContentGenerationAgent._generate_introduction(
            model,
            main_topic,
            tone,
            technical_depth,
            target_audience,
            news_articles,
            quotes,
        )

        # Generate content for each subtopic
        sections = []
        for subtopic in subtopics:
            section_content = await ContentGenerationAgent._generate_section(
                model,
                subtopic,
                main_topic,
                tone,
                technical_depth,
                target_audience,
                news_articles,
                related_keywords,
            )
            sections.append({"heading": subtopic, "content": section_content})

        # Generate a conclusion
        conclusion = await ContentGenerationAgent._generate_conclusion(
            model, main_topic, tone, technical_depth
        )

        # Calculate reading time (average reading speed: 200-250 words per minute)
        total_words = (
            len(introduction.split())
            + sum(len(section["content"].split()) for section in sections)
            + len(conclusion.split())
        )
        reading_time_minutes = max(1, round(total_words / 225))

        logger.info(f"Blog post generated: {title} ({reading_time_minutes} min read)")

        return {
            "title": title,
            "introduction": introduction,
            "sections": sections,
            "conclusion": conclusion,
            "reading_time_minutes": reading_time_minutes,
            "word_count": total_words,
        }

    @staticmethod
    async def _generate_title(model, topic: str, tone: str) -> str:
        """Generate a compelling title for the blog post."""
        prompt = f"""
        Create an engaging, SEO-optimized title for a blog post about "{topic}".
        The title should be:
        - Attention-grabbing and click-worthy
        - Between 40-60 characters long
        - Include the main keyword "{topic}"
        - Match a {tone} tone
        - Not use clickbait tactics
        
        Return only the title text with no quotation marks or explanation.
        """

        try:
            response = await asyncio.to_thread(
                lambda: model.generate_content(prompt).text
            )

            # Clean up the response
            title = response.strip().strip('"').strip("'")
            logger.info(f"Generated title: {title}")
            return title

        except Exception as e:
            logger.error(f"Error generating title: {str(e)}")
            # Fallback title
            return f"{topic.title()}: A Comprehensive Guide"

    @staticmethod
    async def _generate_introduction(
        model,
        topic: str,
        tone: str,
        technical_depth: str,
        target_audience: str,
        news_articles: List[Dict],
        quotes: List[Dict],
    ) -> str:
        """Generate an engaging introduction for the blog post."""
        # Create a summarized news context if available
        news_context = ""
        if news_articles:
            news_context = f"""
            Recent news context to reference:
            {news_articles[0].get("title", "")}: {news_articles[0].get("description", "")}
            """

        # Add a relevant quote if available
        quote_context = ""
        if quotes:
            quote = quotes[0]
            quote_context = f"""
            Consider working in this quote if relevant:
            "{quote.get("content", "")}" - {quote.get("author", "Unknown")}
            """

        prompt = f"""
        Write an engaging introduction (about 150 words) for a blog post about "{topic}".
        
        Target audience: {target_audience}
        Technical depth: {technical_depth}
        Tone: {tone}
        
        {news_context}
        {quote_context}
        
        The introduction should:
        1. Hook the reader with an engaging opening
        2. Clearly state what the blog post will cover
        3. Establish your credibility on the subject
        4. Include the main keyword "{topic}" naturally
        5. Be about 150 words in length
        
        Return only the introduction text with no added context or explanation.
        """

        try:
            response = await asyncio.to_thread(
                lambda: model.generate_content(prompt).text
            )

            # Clean up the response
            introduction = response.strip()
            logger.info(f"Generated introduction: {len(introduction.split())} words")
            return introduction

        except Exception as e:
            logger.error(f"Error generating introduction: {str(e)}")
            # Fallback introduction
            return f"In this comprehensive guide, we'll explore {topic} and its various aspects. This post will walk you through everything you need to know about {topic}, from basic concepts to practical applications."

    @staticmethod
    async def _generate_section(
        model,
        subtopic: str,
        main_topic: str,
        tone: str,
        technical_depth: str,
        target_audience: str,
        news_articles: List[Dict],
        related_keywords: List[Dict],
    ) -> str:
        """Generate content for a blog section based on the subtopic."""
        # Get relevant keywords for this section
        relevant_keywords = [kw["word"] for kw in related_keywords[:5]]
        keywords_str = ", ".join(relevant_keywords)

        # Include a news reference if available
        news_context = ""
        for article in news_articles:
            if (
                subtopic.lower() in article.get("title", "").lower()
                or subtopic.lower() in article.get("description", "").lower()
            ):
                news_context = f"""
                Consider referencing this news:
                {article.get("title", "")}: {article.get("description", "")}
                """
                break

        prompt = f"""
        Write a comprehensive section (about 250-300 words) for a blog post about "{main_topic}".
        
        Section topic: "{subtopic}"
        Target audience: {target_audience}
        Technical depth: {technical_depth}
        Tone: {tone}
        
        {news_context}
        
        Consider naturally incorporating some of these related keywords: {keywords_str}
        
        The section should:
        1. Start with a clear explanation of the subtopic
        2. Include practical examples or applications
        3. Provide valuable insights or tips
        4. Use bullet points or numbered lists where appropriate
        5. Be between 250-300 words
        
        Return only the section content with Markdown formatting. Do not include the heading.
        """

        try:
            response = await asyncio.to_thread(
                lambda: model.generate_content(prompt).text
            )

            # Clean up the response
            section_content = response.strip()
            logger.info(
                f"Generated section '{subtopic}': {len(section_content.split())} words"
            )
            return section_content

        except Exception as e:
            logger.error(f"Error generating section '{subtopic}': {str(e)}")
            # Fallback section content
            return f"This section covers {subtopic} as it relates to {main_topic}. Understanding this aspect is crucial for mastering the subject matter. Let's explore the key concepts and practical applications."

    @staticmethod
    async def _generate_conclusion(
        model, topic: str, tone: str, technical_depth: str
    ) -> str:
        """Generate a conclusion with a call-to-action."""
        prompt = f"""
        Write a strong conclusion (about 150 words) for a blog post about "{topic}".
        
        Tone: {tone}
        Technical depth: {technical_depth}
        
        The conclusion should:
        1. Summarize the key points covered in the blog
        2. Reinforce the main value proposition
        3. End with a clear call-to-action
        4. Include the main keyword "{topic}" naturally
        5. Be about 150 words in length
        
        Return only the conclusion text with no added context or explanation.
        """

        try:
            response = await asyncio.to_thread(
                lambda: model.generate_content(prompt).text
            )

            # Clean up the response
            conclusion = response.strip()
            logger.info(f"Generated conclusion: {len(conclusion.split())} words")
            return conclusion

        except Exception as e:
            logger.error(f"Error generating conclusion: {str(e)}")
            # Fallback conclusion
            return f"In conclusion, we've covered the essential aspects of {topic}. By understanding these concepts, you can effectively leverage {topic} in your work. We hope this guide has been helpful. If you have any questions or experiences with {topic}, feel free to share them in the comments below!"


class SEOOptimizationAgent:
    """Agent responsible for SEO optimization of the blog content."""

    @staticmethod
    async def optimize_blog(
        topic_analysis: Dict, research_data: Dict, blog_content: Dict
    ) -> Dict:
        """
        Optimize the blog post for SEO.

        Args:
            topic_analysis: The output from TopicAnalysisAgent.analyze_topic()
            research_data: The output from ResearchAgent.gather_research()
            blog_content: The output from ContentGenerationAgent.generate_blog_post()

        Returns:
            Dict with SEO metadata
        """
        # Extract relevant information
        main_topic = topic_analysis["main_topic"]
        keywords = topic_analysis["keywords"]
        related_keywords = research_data.get("related_keywords", [])

        # Create Gemini model instance
        model = genai.GenerativeModel("gemini-pro")

        # Generate meta description
        meta_description = await SEOOptimizationAgent._generate_meta_description(
            model, main_topic, blog_content["title"], keywords
        )

        # Generate URL slug
        slug = await SEOOptimizationAgent._generate_slug(
            model, main_topic, blog_content["title"]
        )

        # Generate SEO tags
        tags = await SEOOptimizationAgent._generate_tags(
            model, main_topic, keywords, related_keywords
        )

        logger.info(f"SEO optimization complete: {len(tags)} tags generated")

        return {
            "meta_description": meta_description,
            "slug": slug,
            "tags": tags,
            "primary_keyword": main_topic,
            "secondary_keywords": keywords,
        }

    @staticmethod
    async def _generate_meta_description(
        model, topic: str, title: str, keywords: List[str]
    ) -> str:
        """Generate an SEO-optimized meta description."""
        prompt = f"""
        Create an SEO-optimized meta description for a blog post titled "{title}" about "{topic}".
        
        The meta description should:
        1. Be exactly 150-160 characters long (critical for SEO)
        2. Include the primary keyword "{topic}" naturally
        3. Include at least one secondary keyword from this list: {", ".join(keywords[:3])}
        4. Contain a clear value proposition
        5. End with a subtle call-to-action
        
        Return only the meta description text with no added context or explanation.
        """

        try:
            response = await asyncio.to_thread(
                lambda: model.generate_content(prompt).text
            )

            # Clean up the response
            description = response.strip().strip('"').strip("'")

            # Ensure it's not too long
            if len(description) > 160:
                description = description[:157] + "..."

            logger.info(f"Generated meta description: {len(description)} chars")
            return description

        except Exception as e:
            logger.error(f"Error generating meta description: {str(e)}")
            # Fallback meta description
            return f"Learn everything you need to know about {topic} in this comprehensive guide. Discover key concepts, practical applications, and expert tips."[
                :160
            ]

    @staticmethod
    async def _generate_slug(model, topic: str, title: str) -> str:
        """Generate an SEO-friendly URL slug."""
        prompt = f"""
        Create an SEO-friendly URL slug for a blog post titled "{title}" about "{topic}".
        
        The slug should:
        1. Be 3-5 words maximum
        2. Include the primary keyword "{topic}" or a close variation
        3. Use hyphens between words
        4. Contain only lowercase letters, numbers, and hyphens
        5. No special characters or spaces
        
        Return only the slug text with no added context or explanation.
        """

        try:
            response = await asyncio.to_thread(
                lambda: model.generate_content(prompt).text
            )

            # Clean up and format the response
            slug = response.strip().strip('"').strip("'").lower()

            # Replace spaces with hyphens and remove non-alphanumeric characters
            import re

            slug = re.sub(r"[^a-z0-9\s-]", "", slug)
            slug = re.sub(r"\s+", "-", slug)
            slug = re.sub(r"-+", "-", slug)

            logger.info(f"Generated slug: {slug}")
            return slug

        except Exception as e:
            logger.error(f"Error generating slug: {str(e)}")
            # Fallback slug
            topic_slug = topic.lower().replace(" ", "-")
            return f"guide-to-{topic_slug}"

    @staticmethod
    async def _generate_tags(
        model, topic: str, keywords: List[str], related_keywords: List[Dict]
    ) -> List[str]:
        """Generate SEO tags for the blog post."""
        # Combine direct keywords with related keywords
        all_keywords = keywords.copy()
        for kw in related_keywords:
            all_keywords.append(kw.get("word", ""))

        # Remove duplicates and limit to top 15 keywords
        unique_keywords = list(set([k.lower() for k in all_keywords if k]))[:15]

        prompt = f"""
        Create a list of 8-10 SEO tags for a blog post about "{topic}".
        
        Use these keywords as inspiration: {", ".join(unique_keywords)}
        
        The tags should:
        1. Include the primary keyword "{topic}"
        2. Include relevant secondary keywords
        3. Be succinct (1-3 words each)
        4. Use lowercase format
        5. No hashtags or special characters
        
        Return the tags as a comma-separated list with no explanation.
        """

        try:
            response = await asyncio.to_thread(
                lambda: model.generate_content(prompt).text
            )

            # Clean up and format the response
            tags = [tag.strip().lower() for tag in response.strip().split(",")]

            # Ensure there are enough tags
            if len(tags) < 5:
                tags.extend(
                    [topic.lower(), "guide", "tutorial", "tips", "best practices"][
                        : 5 - len(tags)
                    ]
                )

            logger.info(f"Generated tags: {len(tags)} tags")
            return tags

        except Exception as e:
            logger.error(f"Error generating tags: {str(e)}")
            # Fallback tags
            return [topic.lower(), "guide", "tutorial", "tips", "best practices"]


class ExportAgent:
    """Agent responsible for exporting the blog content to various formats."""

    @staticmethod
    def export_blog(
        topic_analysis: Dict,
        blog_content: Dict,
        seo_data: Dict,
        output_dir: str = "output",
    ) -> Dict:
        """
        Export the blog content to various formats.

        Args:
            topic_analysis: The output from TopicAnalysisAgent.analyze_topic()
            blog_content: The output from ContentGenerationAgent.generate_blog_post()
            seo_data: The output from SEOOptimizationAgent.optimize_blog()
            output_dir: Directory to save the output files

        Returns:
            Dict with paths to exported files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for file naming
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_filename = f"{seo_data['slug']}-{timestamp}"

        # Export as Markdown
        md_path = os.path.join(output_dir, f"{base_filename}.md")
        markdown_content = ExportAgent._generate_markdown(blog_content, seo_data)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Export metadata as JSON
        json_path = os.path.join(output_dir, f"{base_filename}-metadata.json")
        metadata = {
            "title": blog_content["title"],
            "meta_description": seo_data["meta_description"],
            "slug": seo_data["slug"],
            "tags": seo_data["tags"],
            "primary_keyword": seo_data["primary_keyword"],
            "secondary_keywords": seo_data["secondary_keywords"],
            "reading_time_minutes": blog_content["reading_time_minutes"],
            "word_count": blog_content["word_count"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topic_analysis": {
                "main_topic": topic_analysis["main_topic"],
                "target_audience": topic_analysis.get("target_audience", ""),
                "technical_depth": topic_analysis.get("technical_depth", ""),
                "tone": topic_analysis.get("tone", "educational"),
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Calculate readability score
        try:
            readability_score = readability.flesch_kincaid_grade(markdown_content)
        except:
            readability_score = None

        logger.info(f"Blog exported to: {md_path}")
        logger.info(f"Metadata exported to: {json_path}")

        return {
            "markdown_path": md_path,
            "metadata_path": json_path,
            "readability_score": readability_score,
        }

    @staticmethod
    def _generate_markdown(blog_content: Dict, seo_data: Dict) -> str:
        """Generate Markdown content for the blog post."""
        # Start with YAML front matter
        markdown = f"""---
title: "{blog_content["title"]}"
description: "{seo_data["meta_description"]}"
slug: "{seo_data["slug"]}"
tags: {json.dumps(seo_data["tags"])}
reading_time: {blog_content["reading_time_minutes"]} min
---

# {blog_content["title"]}

{blog_content["introduction"]}

"""

        # Add each section
        for section in blog_content["sections"]:
            markdown += f"## {section['heading']}\n\n{section['content']}\n\n"

        # Add conclusion
        markdown += f"## Conclusion\n\n{blog_content['conclusion']}\n"

        return markdown


class BlogAgentController:
    """Main controller class for the blog writing agent."""

    def __init__(self, output_dir: str = "output"):
        """Initialize the controller."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    async def generate_blog(self, topic: str, tone: str = "educational") -> Dict:
        """
        Generate a complete blog post based on the provided topic.

        Args:
            topic: The main topic of the blog post
            tone: The desired tone of the blog post

        Returns:
            Dict with the paths to the exported files
        """
        logger.info(f"Starting blog generation for topic: {topic} with tone: {tone}")

        try:
            # Step 1: Analyze the topic
            logger.info("Step 1: Analyzing topic...")
            topic_analysis = await TopicAnalysisAgent.analyze_topic(topic, tone)

            # Step 2: Gather research data
            logger.info("Step 2: Gathering research data...")
            research_data = await ResearchAgent.gather_research(topic_analysis)

            # Step 3: Generate blog content
            logger.info("Step 3: Generating blog content...")
            blog_content = await ContentGenerationAgent.generate_blog_post(
                topic_analysis, research_data
            )

            # Step 4: Optimize for SEO
            logger.info("Step 4: Optimizing for SEO...")
            seo_data = await SEOOptimizationAgent.optimize_blog(
                topic_analysis, research_data, blog_content
            )

            # Step 5: Export the blog
            logger.info("Step 5: Exporting blog...")
            export_result = ExportAgent.export_blog(
                topic_analysis, blog_content, seo_data, self.output_dir
            )

            # Step 6: Generate a summary
            summary = {
                "title": blog_content["title"],
                "word_count": blog_content["word_count"],
                "reading_time": blog_content["reading_time_minutes"],
                "sections": len(blog_content["sections"]),
                "readability_score": export_result.get("readability_score"),
                "files": {
                    "markdown": export_result["markdown_path"],
                    "metadata": export_result["metadata_path"],
                },
            }

            logger.info(f"Blog generation complete for: {topic}")
            return summary

        except Exception as e:
            logger.error(f"Error generating blog: {str(e)}")
            raise

    async def process_batch(
        self, topics: List[str], tone: str = "educational"
    ) -> List[Dict]:
        """
        Process multiple topics in batch mode.

        Args:
            topics: List of topics to generate blogs for
            tone: The desired tone for all blogs

        Returns:
            List of summaries for each generated blog
        """
        results = []

        for topic in topics:
            try:
                logger.info(f"Processing batch topic: {topic}")
                result = await self.generate_blog(topic, tone)
                results.append({"topic": topic, "status": "success", "summary": result})
            except Exception as e:
                logger.error(f"Error processing batch topic '{topic}': {str(e)}")
                results.append({"topic": topic, "status": "error", "error": str(e)})

        return results


# Define readability module (simplified)
class readability:
    @staticmethod
    def flesch_kincaid_grade(text: str) -> float:
        """
        Calculate the Flesch-Kincaid Grade Level for the given text.

        A score between 8-12 is ideal for most blog content.

        Args:
            text: The text to analyze

        Returns:
            The Flesch-Kincaid Grade Level score
        """
        # Remove markdown formatting
        import re

        clean_text = re.sub(r"#+ ", "", text)  # Remove headings
        clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", clean_text)  # Remove bold
        clean_text = re.sub(r"\*(.*?)\*", r"\1", clean_text)  # Remove italics
        clean_text = re.sub(
            r"```.*?```", "", clean_text, flags=re.DOTALL
        )  # Remove code blocks
        clean_text = re.sub(r"`(.*?)`", r"\1", clean_text)  # Remove inline code
        clean_text = re.sub(r"!\[(.*?)\]\((.*?)\)", "", clean_text)  # Remove images
        clean_text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", clean_text)  # Remove links

        # Split into sentences and words
        sentences = re.split(r"[.!?]+", clean_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r"\b\w+\b", clean_text)

        if not sentences or not words:
            return 0

        # Count syllables (simplified)
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1

            # Remove e at the end
            if word.endswith("e"):
                word = word[:-1]

            # Count vowel groups
            vowels = "aeiouy"
            count = 0
            prev_is_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel

            return max(1, count)

        # Calculate syllables
        syllables = sum(count_syllables(word) for word in words)

        # Calculate Flesch-Kincaid Grade Level
        if len(sentences) == 0 or len(words) == 0:
            return 0

        grade = (
            0.39 * (len(words) / len(sentences))
            + 11.8 * (syllables / len(words))
            - 15.59
        )
        return round(grade, 1)


async def main():
    """Main entry point for the blog writing agent."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AI Blog Writing Agent")
    parser.add_argument("--topic", type=str, help="The main topic of the blog post")
    parser.add_argument(
        "--topics", type=str, help="Comma-separated list of topics for batch processing"
    )
    parser.add_argument(
        "--tone",
        type=str,
        default="educational",
        help="The tone of the blog post (educational, formal, creative, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for the generated files",
    )
    args = parser.parse_args()

    # Initialize the controller
    controller = BlogAgentController(output_dir=args.output)

    # Check if we're in batch mode
    if args.topics:
        topics = [topic.strip() for topic in args.topics.split(",")]
        print(f"Starting batch processing for {len(topics)} topics...")
        results = await controller.process_batch(topics, args.tone)

        # Print summary
        print("\n===== BATCH PROCESSING RESULTS =====")
        for result in results:
            status = "✅" if result["status"] == "success" else "❌"
            print(f"{status} {result['topic']}")
            if result["status"] == "success":
                summary = result["summary"]
                print(f"   Title: {summary['title']}")
                print(
                    f"   Words: {summary['word_count']} ({summary['reading_time']} min read)"
                )
                print(f"   Files: {os.path.basename(summary['files']['markdown'])}")
            else:
                print(f"   Error: {result['error']}")
        print("===================================")

    elif args.topic:
        # Single topic mode
        print(f"Generating blog for topic: {args.topic}")
        try:
            result = await controller.generate_blog(args.topic, args.tone)

            # Print summary
            print("\n===== BLOG GENERATION COMPLETE =====")
            print(f"Title: {result['title']}")
            print(f"Word Count: {result['word_count']} words")
            print(f"Reading Time: {result['reading_time']} minutes")
            print(f"Sections: {result['sections']}")
            if result.get("readability_score"):
                print(
                    f"Readability Score: {result['readability_score']} (Flesch-Kincaid Grade Level)"
                )
            print(f"Files:")
            print(f"  - Markdown: {result['files']['markdown']}")
            print(f"  - Metadata: {result['files']['metadata']}")
            print("====================================")

        except Exception as e:
            print(f"Error generating blog: {str(e)}")
    else:
        print("Please provide a topic using --topic or multiple topics using --topics")


if __name__ == "__main__":
    asyncio.run(main())
