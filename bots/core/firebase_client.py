import os
import requests
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from google.cloud import firestore

logger = logging.getLogger(__name__)


class FirebaseClient:
    """Firebase client for bot operations with emulator support."""
    
    def __init__(self, project_id: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Firebase client.
        
        Args:
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
            api_key: Firebase API key (for auth REST API, defaults to 'fake-api-key' for emulator)
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("project_id must be provided or set GOOGLE_CLOUD_PROJECT")
        
        self.api_key = api_key or "fake-api-key"  # Emulator accepts any key
        
        # Check if using emulator
        self.emulator_host = os.getenv("FIRESTORE_EMULATOR_HOST")
        self.auth_emulator_host = os.getenv("FIREBASE_AUTH_EMULATOR_HOST", "localhost:9099")
        self.use_emulator = bool(self.emulator_host)
        
        # Initialize Firestore client
        # Firestore client automatically uses emulator when FIRESTORE_EMULATOR_HOST is set
        self.db = firestore.Client(project=self.project_id)
        
        logger.info(f"FirebaseClient initialized: project={self.project_id}, emulator={self.use_emulator}")
    
    def login(self, email: str, password: str) -> Tuple[str, str]:
        """
        Authenticate user and return token and user ID.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            Tuple of (id_token, user_id)
        """
        if self.use_emulator:
            auth_url = f"http://{self.auth_emulator_host}/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.api_key}"
        else:
            auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.api_key}"
        
        try:
            response = requests.post(
                auth_url,
                json={
                    "email": email,
                    "password": password,
                    "returnSecureToken": True,
                },
                timeout=10,
            )
            response.raise_for_status()
            auth_data = response.json()
            
            if "idToken" in auth_data:
                user_id = auth_data.get("localId")
                logger.info(f"Authentication successful for {email}")
                return auth_data["idToken"], user_id
            else:
                raise ValueError(f"Authentication failed: {auth_data.get('error', {}).get('message', 'Unknown error')}")
        except requests.RequestException as e:
            logger.error(f"Authentication request failed: {e}")
            raise
    
    def get_user_info(self, user_id: str) -> str:
        """
        Get username from user document.
        
        Args:
            user_id: User document ID
            
        Returns:
            Username string, or empty string if not found
        """
        try:
            user_ref = self.db.collection("users").document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                return user_data.get("username", "")
            return ""
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return ""
    
    def create_story(
        self,
        user_id: str,
        title: str,
        description: str,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Create a new story document in Firestore.
        
        Args:
            user_id: User ID who owns the story
            title: Story title
            description: Story description
            metadata: Story metadata (category, tags, targetAudience, language, copyright, coverImageUrl)
            
        Returns:
            Story document ID
        """
        try:
            # Get author username
            author = self.get_user_info(user_id)
            
            # Create story document
            story_ref = self.db.collection("stories").document()
            story_id = story_ref.id
            
            now = datetime.now(timezone.utc)
            story_data = {
                "id": story_id,
                "title": title,
                "description": description,
                "userId": user_id,
                "isPublished": False,
                "createdAt": now,
                "updatedAt": now,
                "chapterCount": 0,
                "author": author,
                "views": 0,
                "likes": 0,
                **metadata,
            }
            
            story_ref.set(story_data)
            logger.info(f"Created story {story_id}: {title}")
            return story_id
        except Exception as e:
            logger.error(f"Error creating story: {e}")
            raise
    
    def add_chapter(
        self,
        story_id: str,
        chapter_title: str,
        content: str = "",
        user_id: Optional[str] = None,
    ) -> str:
        """
        Add a chapter to a story.
        
        Args:
            story_id: Story document ID
            chapter_title: Chapter title
            content: Chapter content (defaults to empty)
            user_id: User ID (optional, will fetch from story if not provided)
            
        Returns:
            Chapter document ID
        """
        try:
            # Get story to check chapter count and get user_id
            story_ref = self.db.collection("stories").document(story_id)
            story_doc = story_ref.get()
            
            if not story_doc.exists:
                raise ValueError(f"Story {story_id} not found")
            
            story_data = story_doc.to_dict()
            if not user_id:
                user_id = story_data.get("userId")
            
            chapter_count = story_data.get("chapterCount", 0)
            
            # Create chapter document
            chapter_ref = story_ref.collection("chapters").document()
            chapter_id = chapter_ref.id
            
            # Calculate word count
            word_count = len(content.split()) if content else 0
            
            chapter_data = {
                "id": chapter_id,
                "title": chapter_title,
                "content": content,
                "order": chapter_count,
                "wordCount": word_count,
                "userId": user_id,
            }
            
            chapter_ref.set(chapter_data)
            
            # Update story's chapter count and updatedAt
            story_ref.update({
                "chapterCount": chapter_count + 1,
                "updatedAt": datetime.now(timezone.utc),
            })
            
            logger.info(f"Added chapter {chapter_id} to story {story_id}")
            return chapter_id
        except Exception as e:
            logger.error(f"Error adding chapter: {e}")
            raise
    
    def update_chapter(
        self,
        story_id: str,
        chapter_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
    ) -> None:
        """
        Update an existing chapter.
        
        Args:
            story_id: Story document ID
            chapter_id: Chapter document ID
            title: New chapter title (optional)
            content: New chapter content (optional)
        """
        try:
            chapter_ref = (
                self.db.collection("stories")
                .document(story_id)
                .collection("chapters")
                .document(chapter_id)
            )
            
            update_data = {}
            if title is not None:
                update_data["title"] = title
            if content is not None:
                update_data["content"] = content
                update_data["wordCount"] = len(content.split())
            
            if update_data:
                chapter_ref.update(update_data)
                
                # Update story's updatedAt
                self.db.collection("stories").document(story_id).update({
                    "updatedAt": datetime.now(timezone.utc),
                })
                
                logger.info(f"Updated chapter {chapter_id} in story {story_id}")
        except Exception as e:
            logger.error(f"Error updating chapter: {e}")
            raise
    
    def get_story(self, story_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a story document.
        
        Args:
            story_id: Story document ID
            
        Returns:
            Story data dictionary or None if not found
        """
        try:
            story_ref = self.db.collection("stories").document(story_id)
            story_doc = story_ref.get()
            
            if story_doc.exists:
                return story_doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting story: {e}")
            return None
    
    def get_user_stories(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all stories owned by a user.
        
        Args:
            user_id: User ID
            limit: Optional limit on number of stories
            
        Returns:
            List of story dictionaries
        """
        try:
            query = (
                self.db.collection("stories")
                .where("userId", "==", user_id)
                .order_by("updatedAt", direction=firestore.Query.DESCENDING)
            )
            
            if limit:
                query = query.limit(limit)
            
            docs = query.stream()
            stories = []
            for doc in docs:
                story_data = doc.to_dict()
                story_data["id"] = doc.id
                stories.append(story_data)
            
            return stories
        except Exception as e:
            logger.error(f"Error getting user stories: {e}")
            return []
    
    def get_chapters(self, story_id: str) -> List[Dict[str, Any]]:
        """
        Get all chapters for a story, ordered by order field.
        
        Args:
            story_id: Story document ID
            
        Returns:
            List of chapter dictionaries
        """
        try:
            chapters_ref = (
                self.db.collection("stories")
                .document(story_id)
                .collection("chapters")
            )
            
            query = chapters_ref.order_by("order")
            docs = query.stream()
            
            chapters = []
            for doc in docs:
                chapter_data = doc.to_dict()
                chapter_data["id"] = doc.id
                chapters.append(chapter_data)
            
            return chapters
        except Exception as e:
            logger.error(f"Error getting chapters: {e}")
            return []
    
    def update_story(
        self,
        story_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Update story title and/or description.
        
        Args:
            story_id: Story document ID
            title: New title (optional)
            description: New description (optional)
        """
        try:
            update_data = {"updatedAt": datetime.now(timezone.utc)}
            if title is not None:
                update_data["title"] = title
            if description is not None:
                update_data["description"] = description
            
            self.db.collection("stories").document(story_id).update(update_data)
            logger.info(f"Updated story {story_id}")
        except Exception as e:
            logger.error(f"Error updating story: {e}")
            raise
    
    def publish_story(self, story_id: str) -> None:
        """
        Publish a story by setting isPublished to True.
        
        Args:
            story_id: Story document ID
        """
        try:
            story_ref = self.db.collection("stories").document(story_id)
            story_doc = story_ref.get()
            
            if not story_doc.exists:
                raise ValueError(f"Story {story_id} not found")
            
            story_ref.update({
                "isPublished": True,
                "updatedAt": datetime.now(timezone.utc),
            })
            logger.info(f"Published story {story_id}")
        except Exception as e:
            logger.error(f"Error publishing story: {e}")
            raise

    def add_plot(self, story_id: str, plot_name: str, events: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Add a new plot to a story.
        
        Args:
            story_id: Story document ID
            plot_name: Name of the plot
            events: Optional list of events, each with 'name' and 'content' keys
            
        Returns:
            Plot document ID
        """
        try:
            # Verify story exists
            story_ref = self.db.collection("stories").document(story_id)
            story_doc = story_ref.get()
            
            if not story_doc.exists:
                raise ValueError(f"Story {story_id} not found")
            
            # Create plot document in subcollection
            plots_collection = story_ref.collection("plots")
            plot_ref = plots_collection.document()
            plot_id = plot_ref.id
            
            # Convert template events to PlotEvent format if provided
            plot_events = []
            if events:
                plot_events = [
                    {
                        "id": f"{plot_id}-event-{idx}",
                        "name": event.get("name", ""),
                        "content": event.get("content", ""),
                    }
                    for idx, event in enumerate(events)
                ]
            
            plot_data = {
                "id": plot_id,
                "name": plot_name,
                "description": "",
                "events": plot_events,
            }
            
            plot_ref.set(plot_data)
            logger.info(f"Added plot {plot_id} ({plot_name}) with {len(plot_events)} events to story {story_id}")
            return plot_id
        except Exception as e:
            logger.error(f"Error adding plot: {e}")
            raise