"""Agent runner for bot operations."""
import logging
import os
import re
from typing import Optional
import random
from datetime import datetime, timezone
from bots.templates import PLOT_TEMPLATES
# Handle imports for both direct execution and module import
try:
    from agents.storyAgent.agent import StoryAgent
    from agents.storyAgent.llm_provider import get_llm_provider
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from agents.storyAgent.agent import StoryAgent
    from agents.storyAgent.llm_provider import get_llm_provider

from .firebase_client import FirebaseClient
from .personality import Personality, Personality as PersonalityModule
from .scheduler import Scheduler

logger = logging.getLogger(__name__)


class AgentRunner:
    """Runs bot actions using StoryAgent tools."""
    
    def __init__(
        self,
        firebase_client: FirebaseClient,
        personality: Personality,
        project_id: Optional[str] = None,
        story_length: str = "short",
        max_chapters_per_story: int = 10,
    ):
        """
        Initialize agent runner.
        
        Args:
            firebase_client: FirebaseClient instance
            personality: Personality instance
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
            story_length: Story length preference (short/medium/long)
            max_chapters_per_story: Maximum chapters before story is considered complete
        """
        self.firebase_client = firebase_client
        self.personality = personality
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("project_id must be provided or set GOOGLE_CLOUD_PROJECT")
        
        self.story_length = story_length
        self.max_chapters_per_story = max_chapters_per_story
        
        # Initialize StoryAgent
        self.story_agent = StoryAgent(project_id=self.project_id)
        # Initialize LLM provider for personality-based generation
        self.llm_provider = get_llm_provider(project_id=self.project_id)
        self.user_id: Optional[str] = None
        self.scheduler: Optional[Scheduler] = None
    
    async def authenticate(self, email: str, password: str) -> None:
        """
        Authenticate bot user.
        
        Args:
            email: Bot email
            password: Bot password
        """
        try:
            token, user_id = self.firebase_client.login(email, password)
            self.user_id = user_id
            logger.info(f"Authenticated as {email} (user_id: {user_id})")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def _extract_title_from_content(self, content: str) -> str:
        """
        Extract title from generated content if present.
        
        Args:
            content: Generated content that may contain "Title: ..." format
            
        Returns:
            Extracted title or default title
        """
        # Look for "Title: ..." pattern
        title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        
        
        return "Untitled Story"
    
    def _extract_chapter_title_from_content(self, content: str, chapter_number: int) -> str:
        """
        Extract chapter title from generated content.
        
        Args:
            content: Generated chapter content
            chapter_number: Chapter number
            
        Returns:
            Extracted title or default title
        """
        # Look for "Title: ..." pattern
        title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        
        # Fallback to default chapter title
        return f"Chapter {chapter_number}"
    
    def _extract_story_content(self, content: str) -> str:
        """
        Extract story content, removing title and summary if present.
        
        Args:
            content: Full generated content
            
        Returns:
            Clean story content
        """
        # Remove title line if present
        content = re.sub(r"Title:\s*.+?\n", "", content, flags=re.IGNORECASE)
        # Remove summary section if present
        content = re.sub(r"Summary:.*", "", content, flags=re.IGNORECASE | re.DOTALL)
        # Remove "Story:" label if present
        content = re.sub(r"^Story:\s*", "", content, flags=re.IGNORECASE | re.MULTILINE)
        return content.strip()
    
    def _extract_chapter_content(self, content: str) -> str:
        """
        Extract chapter content, removing title and metadata if present.
        
        Args:
            content: Full generated chapter content
            
        Returns:
            Clean chapter content
        """
        # Remove title line if present
        content = re.sub(r"Title:\s*.+?\n", "", content, flags=re.IGNORECASE)
        # Remove chapter number label if present
        content = re.sub(r"Chapter Number:\s*\d+\s*\n", "", content, flags=re.IGNORECASE)
        # Remove "Content:" label if present
        content = re.sub(r"^Content:\s*", "", content, flags=re.IGNORECASE | re.MULTILINE)
        return content.strip()
    
    async def create_new_story(self) -> Optional[str]:
        """
        Create a new story and generate its first chapter.
        Uses a random personality for each new story.
        
        Returns:
            Story ID if successful, None otherwise
        """
        if not self.user_id:
            logger.error("Not authenticated. Call authenticate() first.")
            return None
        
        try:
            # Select a random personality for this new story
            story_personality = PersonalityModule.get_random_bot()
            logger.info(f"Creating new story with personality: {story_personality.name}")
            
            # Generate story title and description
            selected_template = random.choice(PLOT_TEMPLATES)
            first_stage = selected_template["events"][0] if selected_template.get("events") else None
            plot_context = f"Stage 1: {first_stage['name']}. {first_stage['content']}" if first_stage else ""
            
            # Use LLM to generate title and description
            story_title = story_personality.generate_story_title(
                plot_template_name=selected_template["name"],
                llm_provider=self.llm_provider
            )
            story_description = story_personality.generate_story_description(
                title=story_title,
                plot_template_name=selected_template["name"],
                plot_context=plot_context,
                llm_provider=self.llm_provider
            )
            metadata = story_personality.generate_story_metadata()

            metadata["plotTemplateId"] = selected_template["id"]
            metadata["plotArchetype"] = selected_template["name"]
            metadata["totalPlotStages"] = len(selected_template["events"])            
            metadata["genre"] = story_personality.genre
            metadata["tone"] = story_personality.style
            metadata["botPersonality"] = story_personality.name
            
            # Ensure botPersonality is set (safety check)
            if not metadata.get("botPersonality"):
                logger.error(f"botPersonality not set in metadata! Setting to {story_personality.name}")
                metadata["botPersonality"] = story_personality.name
            
            # Create story document
            story_id = self.firebase_client.create_story(
                user_id=self.user_id,
                title=story_title,
                description=story_description,
                metadata=metadata,
            )
            
            logger.info(f"Created story {story_id}: {story_title} with personality: {story_personality.name} (botPersonality: {metadata.get('botPersonality')})")
            
            # save plot to firebase
            plot_id = self.firebase_client.add_plot(
                story_id=story_id,
                plot_name=selected_template["name"],
                events=selected_template["events"]
            )
            logger.info(f"Added plot {plot_id} ({selected_template['name']}) to story {story_id}")
            
            first_stage = selected_template["events"][0]
            
            # Generate story content using StoryAgent with the selected personality
            result = await self.story_agent.generate_story(
                    story_id=story_id,
                    genre=story_personality.genre,
                    tone=story_personality.style,
                    length=self.story_length,
                    generate_first_chapter_only=True,
                    plot_context=f"Stage 1: {first_stage['name']}. {first_stage['content']}"
                )
            
            if "error" in result:
                logger.error(f"Story generation failed: {result['error']}")
                return None
            
            # Extract and save first chapter
            content = result.get("content", "")
            if content:
                # Extract title from content if present
                chapter_title = self._extract_chapter_title_from_content(content, 1)
                story_content = self._extract_story_content(content)
                
                # Update story title if one was generated
                extracted_title = self._extract_title_from_content(content)
                if extracted_title and extracted_title != story_title:
                    self.firebase_client.update_story(story_id, title=extracted_title)
                
                # Add first chapter
                chapter_id = self.firebase_client.add_chapter(
                    story_id=story_id,
                    chapter_title=chapter_title,
                    content=story_content,
                    user_id=self.user_id,
                )
                
                logger.info(f"Added first chapter {chapter_id} to story {story_id}")
            else:
                # Create empty first chapter if no content generated
                self.firebase_client.add_chapter(
                    story_id=story_id,
                    chapter_title="Chapter 1",
                    content="",
                    user_id=self.user_id,
                )
            
            return story_id
            
        except Exception as e:
            logger.error(f"Error creating new story: {e}", exc_info=True)
            return None
    
    async def continue_story(self, story_id: str) -> bool:
        """
        Continue an existing story by generating a new chapter.
        
        Args:
            story_id: Story document ID to continue
            
        Returns:
            True if successful, False otherwise
        """
        if not self.user_id:
            logger.error("Not authenticated. Call authenticate() first.")
            return False
        
        try:
            # Get story and existing chapters
            story = self.firebase_client.get_story(story_id)
            if not story:
                logger.error(f"Story {story_id} not found")
                return False
            
            
            # Check if story belongs to this bot
            if story.get("userId") != self.user_id:
                logger.warning(f"Story {story_id} does not belong to this bot")
                return False
            
            # Check chapter limit
            current_chapter_count = story.get("chapterCount", 0)
            if current_chapter_count >= self.max_chapters_per_story:
                logger.info(f"Story {story_id} has reached max chapters ({current_chapter_count})")
                return False
            
            # Get the personality that created this story
            # Check both top-level (where it's stored when created) and nested metadata (for backward compatibility)
            bot_personality_name = story.get("botPersonality") or story.get("metadata", {}).get("botPersonality")
            
            if not bot_personality_name:
                logger.warning(f"Story {story_id} has no botPersonality metadata, using default personality")
                story_personality = self.personality
            else:
                # Load the personality that created this story to maintain consistency
                story_personality = PersonalityModule.get_bot_by_name(bot_personality_name)
                if not story_personality:
                    logger.warning(f"Story {story_id} has unknown botPersonality '{bot_personality_name}', using default personality")
                    story_personality = self.personality
                else:
                    logger.info(f"Continuing story {story_id} with original personality: {story_personality.name}")
            
            # Ensure genre and tone are set in story metadata (for chapter generation)
            # They should already be set from story creation, but update if missing
            if not story.get("genre") or not story.get("tone"):
                update_data = {"updatedAt": datetime.now(timezone.utc)}
                if not story.get("genre"):
                    update_data["genre"] = story_personality.genre
                if not story.get("tone"):
                    update_data["tone"] = story_personality.style
                if len(update_data) > 1:  # More than just updatedAt
                    self.firebase_client.db.collection("stories").document(story_id).update(update_data)
                    logger.info(f"Updated story {story_id} with genre/tone from personality")
            
            # Get plot template ID (check both top-level and nested metadata)
            metadata = story.get("metadata", {})
            template_id = story.get("plotTemplateId") or metadata.get("plotTemplateId")
            
            # Find the actual template data
            plot_template = next((t for t in PLOT_TEMPLATES if t["id"] == template_id), None)
            
            current_chapter_count = story.get("chapterCount", 0)
            next_chapter_number = current_chapter_count + 1
            
            # 2. CALCULATE CURRENT PLOT STAGE
            # We need to map 'next_chapter_number' to one of the 5 plot stages.
            # Logic: Distribute chapters evenly across stages.
            plot_instruction = ""
            
            if plot_template:
                total_planned_chapters = self.max_chapters_per_story
                total_stages = len(plot_template["events"])
                
                # Math to find which stage index (0-4) we are in
                # e.g., if max_chapters=10, chapter 1-2 = Stage 0, 3-4 = Stage 1...
                stage_index = int((next_chapter_number - 1) / total_planned_chapters * total_stages)
                
                # Clamp index just in case
                stage_index = min(stage_index, total_stages - 1)
                
                current_stage_data = plot_template["events"][stage_index]
                
                # 3. CREATE THE CONTEXT INSTRUCTION
                plot_instruction = (
                    f"NARRATIVE ARC INSTRUCTION: This story follows the '{plot_template['name']}' archetype. "
                    f"We are currently in Chapter {next_chapter_number} of {total_planned_chapters}. "
                    f"This corresponds to Stage {stage_index + 1}: '{current_stage_data['name']}'. "
                    f"The events in this chapter must align with: {current_stage_data['content']}"
                )
                
                logger.info(f"Story {story_id}: Applying plot stage '{current_stage_data['name']}'")

            # Get existing chapters
            chapters = self.firebase_client.get_chapters(story_id)
            
            # Get existing chapters for context
            chapters = self.firebase_client.get_chapters(story_id)
            previous_chapters = [
                {
                    "chapterNumber": ch.get("order", idx + 1),
                    "title": ch.get("title", ""),
                    "content": ch.get("content", ""),
                }
                for idx, ch in enumerate(chapters)
            ]

            # 4. PASS PLOT INSTRUCTION TO AGENT
            # The chapter tool will get genre and tone from story metadata
            result = await self.story_agent.generate_chapter(
                story_id=story_id,
                chapter_number=next_chapter_number,
                previous_chapters=previous_chapters,                
                plot_context=plot_instruction
            )
            
            if "error" in result:
                logger.error(f"Chapter generation failed: {result['error']}")
                return False
            
            # Extract and save chapter
            content = result.get("content", "")
            if content:
                chapter_title = self._extract_chapter_title_from_content(content, next_chapter_number)
                chapter_content = self._extract_chapter_content(content)
                
                chapter_id = self.firebase_client.add_chapter(
                    story_id=story_id,
                    chapter_title=chapter_title,
                    content=chapter_content,
                    user_id=self.user_id,
                )
                
                logger.info(f"Added chapter {chapter_id} ({next_chapter_number}) to story {story_id}")
                
                # Check if story is finished (reached max chapters) and publish it
                updated_story = self.firebase_client.get_story(story_id)
                if updated_story:
                    new_chapter_count = updated_story.get("chapterCount", 0)
                    if new_chapter_count >= self.max_chapters_per_story:
                        self.firebase_client.publish_story(story_id)
                        logger.info(f"Story {story_id} has reached max chapters ({new_chapter_count}) and has been published")
                
                return True
            else:
                logger.warning(f"No content generated for chapter {next_chapter_number}")
                return False
                
        except Exception as e:
            logger.error(f"Error continuing story {story_id}: {e}", exc_info=True)
            return False
    
    async def execute_action(self) -> None:
        """Execute a single bot action (create new story or continue existing)."""
        if not self.user_id:
            logger.error("Not authenticated. Call authenticate() first.")
            return
        
        if not self.scheduler:
            logger.error("Scheduler not configured. Call set_scheduler() first.")
            return
        
        # Decide action
        should_continue = self.scheduler.should_continue_existing()
        
        if should_continue:
            # Try to continue an existing story
            # The bot can continue stories from any personality, but will use the story's original personality
            stories = self.firebase_client.get_user_stories(self.user_id)
            if stories:
                story = self.scheduler.select_story_for_continuation(
                    stories,
                    max_chapters=self.max_chapters_per_story,
                )
                if story:
                    await self.continue_story(story["id"])
                    return
            
            # Fallback to creating new story if no continuable stories
            logger.info("No continuable stories found, creating new story instead")
            await self.create_new_story()
        else:
            # Create new story
            await self.create_new_story()
    
    def set_scheduler(self, scheduler: Scheduler) -> None:
        """Set the scheduler for this runner."""
        self.scheduler = scheduler
    
    async def run(self) -> None:
        """Run the bot continuously using the scheduler."""
        if not self.scheduler:
            logger.error("Scheduler not configured. Call set_scheduler() first.")
            return
        
        await self.scheduler.run_daily(self.execute_action)
