import random
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class Personality:
    """Defines a bot's personality, writing style, and behavior."""
    
    def __init__(
        self,
        name: str,
        style: str,
        genre: str,
        behavior: str,
        category: str = "",
        tags: List[str] = None,
        target_audience: str = "general",
        language: str = "en",
        copyright: str = "All rights reserved",
        cover_image_url: str = "",
    ):
        """
        Initialize personality.
        
        Args:
            name: Bot's name
            style: Writing style description
            genre: Preferred genre
            behavior: Behavioral description
            category: Story category (defaults to genre if not provided)
            tags: List of tags for stories
            target_audience: Target audience (e.g., "young adult", "adult", "general")
            language: Language code (default: "en")
            copyright: Copyright notice
            cover_image_url: Default cover image URL
        """
        self.name = name
        self.style = style
        self.genre = genre
        self.behavior = behavior
        self.category = category or genre
        self.tags = tags or []
        self.target_audience = target_audience
        self.language = language
        self.copyright = copyright
        self.cover_image_url = cover_image_url
    
    def build_prompt(self) -> str:
        """Build a prompt describing the personality."""
        return f"""
You are {self.name}.
Your writing style: {self.style}
Your preferred genre: {self.genre}
Behavior: {self.behavior}
Focus on writing stories consistent with these traits.
"""
    
    def generate_story_metadata(self) -> Dict[str, Any]:
        """
        Generate metadata for a new story based on personality.
        
        Returns:
            Dictionary with story metadata
        """
        return {
            "category": self.category,
            "tags": self.tags.copy() if self.tags else [],
            "targetAudience": self.target_audience,
            "language": self.language,
            "copyright": self.copyright,
            "coverImageUrl": self.cover_image_url,
            "botPersonality": self.name, 
        }
    
    def generate_story_title(self, theme: str = "", llm_provider=None, plot_template_name: str = "") -> str:
        """
        Generate a story title based on personality and optional theme.
        Uses LLM if provider is available, otherwise falls back to simple generation.
        
        Args:
            theme: Optional theme or prompt for the story
            llm_provider: Optional LLM provider for enhanced generation
            plot_template_name: Optional plot template name for context
            
        Returns:
            Story title
        """
        # Use LLM if available
        if llm_provider:
            try:
                prompt = f"""You are {self.name}, a writer specializing in {self.genre} stories.
Your writing style: {self.style}
Your behavior: {self.behavior}

Generate a compelling, creative story title that:
1. Reflects your {self.style} writing style
2. Fits the {self.genre} genre
3. Is intriguing and memorable
4. Captures the essence of a {self.genre} story

"""
                if plot_template_name:
                    prompt += f"Plot archetype: {plot_template_name}\n\n"
                if theme:
                    prompt += f"Theme/prompt: {theme}\n\n"
                
                prompt += """Generate ONLY the title itself - no quotes, no explanations, just the title.
Make it creative, unique, and fitting for your personality and genre.

Title:"""
                
                generated_title = llm_provider.generate_content(prompt).strip()
                
                # Clean up the response (remove quotes, extra text, etc.)
                generated_title = generated_title.replace('"', '').replace("'", "").strip()
                # Take first line if multiple lines
                generated_title = generated_title.split('\n')[0].strip()
                # Remove "Title:" prefix if present
                if generated_title.lower().startswith("title:"):
                    generated_title = generated_title[6:].strip()
                
                if generated_title and len(generated_title) > 3:
                    logger.info(f"Generated LLM title: {generated_title}")
                    return generated_title
                else:
                    logger.warning(f"LLM generated invalid title, falling back to simple generation")
            except Exception as e:
                logger.warning(f"LLM title generation failed: {e}, falling back to simple generation")
        
        # Fallback to simple title generation
        if theme:
            return f"{self.name}'s {theme}"
        if plot_template_name:
            return f"{self.name}'s {plot_template_name}"
        return f"{self.name}'s {self.genre.title()} Tale"
    
    def generate_story_description(self, title: str = "", llm_provider=None, plot_template_name: str = "", plot_context: str = "") -> str:
        """
        Generate a story description based on personality.
        Uses LLM if provider is available, otherwise falls back to simple generation.
        
        Args:
            title: Story title (optional)
            llm_provider: Optional LLM provider for enhanced generation
            plot_template_name: Optional plot template name for context
            plot_context: Optional plot context/theme
            
        Returns:
            Story description
        """
        # Use LLM if available
        if llm_provider:
            try:
                prompt = f"""You are {self.name}, a writer specializing in {self.genre} stories.
Your writing style: {self.style}
Your behavior: {self.behavior}

"""
                if title:
                    prompt += f"Story Title: {title}\n\n"
                if plot_template_name:
                    prompt += f"Plot Archetype: {plot_template_name}\n\n"
                if plot_context:
                    prompt += f"Plot Context: {plot_context}\n\n"
                
                prompt += f"""Write a compelling, engaging story description (2-3 sentences) that:
1. Reflects your {self.style} writing style
2. Captures the essence of a {self.genre} story
3. Is intriguing and makes readers want to read more
4. Incorporates your personality: {self.behavior}

Write ONLY the description - no labels, no quotes, just the description text.

Description:"""
                
                generated_description = llm_provider.generate_content(prompt).strip()
                
                # Clean up the response
                generated_description = generated_description.replace('"', '').replace("'", "").strip()
                # Remove "Description:" prefix if present
                if generated_description.lower().startswith("description:"):
                    generated_description = generated_description[12:].strip()
                # Take first paragraph if multiple paragraphs
                generated_description = generated_description.split('\n\n')[0].strip()
                
                if generated_description and len(generated_description) > 20:
                    logger.info(f"Generated LLM description: {generated_description[:100]}...")
                    return generated_description
                else:
                    logger.warning(f"LLM generated invalid description, falling back to simple generation")
            except Exception as e:
                logger.warning(f"LLM description generation failed: {e}, falling back to simple generation")
        
        # Fallback to simple description generation
        if title:
            return f"A {self.genre} story by {self.name}. {self.behavior}."
        return f"A {self.genre} story written in a {self.style} style. {self.behavior}."
    
    @staticmethod
    def get_random_bot():
        """Get a random personality bot."""
        return random.choice([SCI_FI_BOT, FANTASY_BOT, NOIR_BOT, ROMANCE_BOT, HORROR_BOT])
    
    @staticmethod
    def get_all_bots():
        """Get all available personality bots."""
        return [SCI_FI_BOT, FANTASY_BOT, NOIR_BOT, ROMANCE_BOT, HORROR_BOT]
    
    @staticmethod
    def get_bot_by_name(name: str):
        """Get a personality bot by name."""
        bots = {
            "Zara-7": SCI_FI_BOT,
            "Eloria the Whisperer": FANTASY_BOT,
            "Grayson Vale": NOIR_BOT,
            "Seraphine Bloom": ROMANCE_BOT,
            "Umbra": HORROR_BOT,
        }
        return bots.get(name)


SCI_FI_BOT = Personality(
    name="Zara-7",
    style="cold, logical, futuristic",
    genre="science fiction",
    behavior="writes short analytical paragraphs and responds to other characters with precision",
    category="Science Fiction",
    tags=["sci-fi", "futuristic", "technology", "space", "100% AI"],
    target_audience="adult",
)

FANTASY_BOT = Personality(
    name="Eloria the Whisperer",
    style="mystical, poetic",
    genre="high fantasy",
    behavior="creates long descriptive worldbuilding passages",
    category="Fantasy",
    tags=["fantasy", "magic", "adventure", "medieval", "100% AI"],
    target_audience="young adult",
)

NOIR_BOT = Personality(
    name="Grayson Vale",
    style="gritty, terse, atmospheric",
    genre="noir crime",
    behavior="delivers sharp, mood-heavy lines and narrates like a hardboiled detective",
    category="Crime / Noir",
    tags=["noir", "detective", "crime", "urban", "100% AI"],
    target_audience="adult",
)

ROMANCE_BOT = Personality(
    name="Seraphine Bloom",
    style="warm, emotional, lyrical",
    genre="romance",
    behavior="focuses on feelings, tension, and sensory-rich interactions",
    category="Romance",
    tags=["romance", "love", "emotion", "relationships", "100% AI"],
    target_audience="adult",
)

HORROR_BOT = Personality(
    name="Umbra",
    style="unsettling, whispery, ominous",
    genre="psychological horror",
    behavior="builds slow dread through eerie descriptions and quiet tension",
    category="Horror",
    tags=["horror", "fear", "dark", "supernatural", "100% AI"],
    target_audience="adult",
)