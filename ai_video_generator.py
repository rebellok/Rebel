import streamlit as st
import cv2
import numpy as np
from moviepy.editor import *
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from moviepy.video.fx.all import speedx,fadein,fadeout
import pandas as pd
import io
import tempfile
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
from pathlib import Path
import pillow_heif
import zipfile
from typing import List, Dict, Tuple, Optional
import json
import hashlib
import gc
import psutil
import multiprocessing
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import queue
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Enable HEIC support
try:
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

# Streamlit configuration
st.set_page_config(
    page_title="AI Video Generator Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    .feature-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    .status-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .status-processing {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .status-error {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .preview-container {
        background: #f8f9fa;
        border: 2px dashed #6c757d;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('''
<div class="main-header">
    <h1>🎬 AI Video Generator Pro</h1>
    <p>Create stunning videos with AI-powered effects, transitions, and optimization</p>
    <p><small>Supports Images, Videos, Audio, HEIC • 5s to 13min • Professional Quality</small></p>
</div>
''', unsafe_allow_html=True)

# Enums and Data Classes
class VideoQuality(Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    ULTRA = "ultra"

class TextPosition(Enum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"
    CUSTOM = "custom"

class TextEffect(Enum):
    NONE = "none"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    TYPE_WRITE = "type_write"
    GLOW = "glow"
    BOUNCE = "bounce"

@dataclass
class CustomTextOverlay:
    """Comprehensive text overlay configuration"""
    text: str = ""
    enabled: bool = False
    font_family: str = "Arial"
    font_size: int = 48
    font_color: Tuple[int, int, int] = (255, 255, 255)  # RGB
    outline_enabled: bool = True
    outline_color: Tuple[int, int, int] = (0, 0, 0)     # RGB
    outline_width: int = 2
    position: TextPosition = TextPosition.CENTER
    custom_x: float = 0.5  # 0.0 to 1.0 (percentage of width)
    custom_y: float = 0.5  # 0.0 to 1.0 (percentage of height)
    start_time: float = 0.0
    duration: float = 0.0  # 0 = full clip duration
    opacity: float = 1.0
    effect: TextEffect = TextEffect.NONE
    background_enabled: bool = False
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 128)  # RGBA
    shadow_enabled: bool = False
    shadow_offset_x: int = 2
    shadow_offset_y: int = 2
    shadow_color: Tuple[int, int, int] = (0, 0, 0)
    rotation: float = 0.0  # degrees
    scale: float = 1.0

@dataclass
class MediaItem:
    """Enhanced media item with text overlay support"""
    file_path: str
    media_type: str
    duration: float
    start_time: float
    effects: List[str]
    transition: str
    volume: float = 1.0
    speed: float = 1.0
    text_overlay: CustomTextOverlay = field(default_factory=CustomTextOverlay)

@dataclass
class VideoProject:
    title: str
    total_duration: float
    quality: VideoQuality
    fps: int
    resolution: Tuple[int, int]
    background_music: Optional[str]
    media_items: List[MediaItem]
    global_effects: List[str]

class TextOverlayGenerator:
    """Generate text overlays using PIL instead of ImageMagick"""
    
    def __init__(self):
        self.default_font_size = 48
        self.default_color = (255, 255, 255, 255)  # White with alpha
        self.outline_color = (0, 0, 0, 255)  # Black outline
        
    def create_text_overlay(self, text: str, size: Tuple[int, int], 
                          position: str = 'top-right', font_size: int = None,
                          color: Tuple[int, int, int, int] = None) -> np.ndarray:
        """Create text overlay using PIL"""
        
        try:
            # Set defaults
            if font_size is None:
                font_size = self.default_font_size
            if color is None:
                color = self.default_color
            
            # Create transparent image
            img = Image.new('RGBA', size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Try to load a system font
            try:
                # Try common system fonts
                font_paths = [
                    "C:/Windows/Fonts/arial.ttf",  # Windows
                    "C:/Windows/Fonts/calibri.ttf",  # Windows
                    "/System/Library/Fonts/Arial.ttf",  # macOS
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
                    "/usr/share/fonts/TTF/arial.ttf"  # Linux alternative
                ]
                
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                
                if font is None:
                    font = ImageFont.load_default()
                    
            except Exception:
                font = ImageFont.load_default()
            
            # Get text dimensions
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position
            margin = 20
            if position == 'top-right':
                x = size[0] - text_width - margin
                y = margin
            elif position == 'top-left':
                x = margin
                y = margin
            elif position == 'bottom-right':
                x = size[0] - text_width - margin
                y = size[1] - text_height - margin
            elif position == 'bottom-left':
                x = margin
                y = size[1] - text_height - margin
            elif position == 'center':
                x = (size[0] - text_width) // 2
                y = (size[1] - text_height) // 2
            else:  # default to top-right
                x = size[0] - text_width - margin
                y = margin
            
            # Draw text with outline for better visibility
            outline_width = 2
            for adj_x in range(-outline_width, outline_width + 1):
                for adj_y in range(-outline_width, outline_width + 1):
                    if adj_x != 0 or adj_y != 0:
                        draw.text((x + adj_x, y + adj_y), text, font=font, fill=self.outline_color)
            
            # Draw main text
            draw.text((x, y), text, font=font, fill=color)
            
            # Convert to numpy array
            return np.array(img)
            
        except Exception as e:
            # Fallback: create simple colored rectangle with text info
            img = np.zeros((50, 200, 4), dtype=np.uint8)
            img[:, :, 3] = 128  # Semi-transparent
            return img

class EffectsEngine:
    """Apply various effects using OpenCV and PIL"""
    
    @staticmethod
    def apply_blur(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Apply blur effect using OpenCV"""
        try:
            kernel_size = int(5 * intensity)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(1, kernel_size)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        except Exception:
            return image
    
    @staticmethod
    def apply_sharpen(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Apply sharpen effect using OpenCV"""
        try:
            kernel = np.array([[-1, -1, -1],
                              [-1, 9 * intensity, -1],
                              [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        except Exception:
            return image
    
    @staticmethod
    def apply_vintage(image: np.ndarray) -> np.ndarray:
        """Apply vintage effect using OpenCV"""
        try:
            # Convert to float for processing
            img_float = image.astype(np.float32) / 255.0
            
            # Apply sepia tone using OpenCV
            sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])
            
            sepia_img = cv2.transform(img_float, sepia_kernel)
            
            # Add vignette effect
            h, w = image.shape[:2]
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            center_x, center_y = w // 2, h // 2
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
            vignette = 1 - (distance / max_distance) * 0.7
            
            sepia_img = sepia_img * vignette[:, :, np.newaxis]
            
            return (np.clip(sepia_img, 0, 1) * 255).astype(np.uint8)
        except Exception:
            return image
    
    @staticmethod
    def apply_glitch(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """Apply glitch effect using NumPy"""
        try:
            h, w, c = image.shape
            result = image.copy()
            
            # Random horizontal shifts
            for _ in range(int(10 * intensity)):
                y = np.random.randint(0, h)
                shift = np.random.randint(-int(w * intensity), int(w * intensity))
                if shift > 0 and shift < w:
                    result[y, shift:] = result[y, :-shift]
                    result[y, :shift] = np.random.randint(0, 255, (shift, c))
                elif shift < 0 and abs(shift) < w:
                    result[y, :shift] = result[y, -shift:]
                    result[y, shift:] = np.random.randint(0, 255, (-shift, c))
            
            return result
        except Exception:
            return image
    
    @staticmethod
    def apply_neon(image: np.ndarray) -> np.ndarray:
        """Apply neon effect using OpenCV"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Create neon glow
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Create colored neon effect
            neon = np.zeros_like(image)
            neon[:, :, 0] = dilated  # Red channel
            neon[:, :, 1] = dilated * 0.5  # Green channel
            neon[:, :, 2] = dilated  # Blue channel
            
            # Blend with original
            result = cv2.addWeighted(image, 0.7, neon, 0.3, 0)
            
            return result
        except Exception:
            return image
    
    @staticmethod
    def apply_black_white(image: np.ndarray) -> np.ndarray:
        """Convert to black and white using OpenCV"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        except Exception:
            return image
    
    @staticmethod
    def apply_brightness(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Adjust brightness using NumPy"""
        try:
            brightened = image.astype(np.float32) * factor
            return np.clip(brightened, 0, 255).astype(np.uint8)
        except Exception:
            return image
    
    @staticmethod
    def apply_contrast(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Adjust contrast using NumPy"""
        try:
            # Convert to float and normalize
            img_float = image.astype(np.float32) / 255.0
            
            # Apply contrast
            contrasted = (img_float - 0.5) * factor + 0.5
            
            # Convert back
            return (np.clip(contrasted, 0, 1) * 255).astype(np.uint8)
        except Exception:
            return image

class CustomVideoClip:
    """Custom video clip handler without ImageMagick dependencies"""
    
    @staticmethod
    def create_image_clip_with_overlay(image_array: np.ndarray, duration: float, 
                                     overlay_text: str = None, 
                                     overlay_position: str = 'top-right') -> VideoFileClip:
        """Create image clip with optional text overlay using PIL"""
        
        try:
            if overlay_text:
                # Create text overlay
                text_gen = TextOverlayGenerator()
                overlay = text_gen.create_text_overlay(
                    overlay_text, 
                    (image_array.shape[1], image_array.shape[0]),
                    overlay_position
                )
                
                # Blend overlay with image
                if overlay.shape[2] == 4:  # RGBA
                    alpha = overlay[:, :, 3:4] / 255.0
                    overlay_rgb = overlay[:, :, :3]
                    
                    # Blend
                    blended = image_array * (1 - alpha) + overlay_rgb * alpha
                    image_array = blended.astype(np.uint8)
            
            # Create clip using MoviePy's ImageClip
            return ImageClip(image_array, duration=duration)
            
        except Exception:
            # Fallback without overlay
            return ImageClip(image_array, duration=duration)

class PreviewGenerator:
    """Generate quick previews for testing"""
    
    def __init__(self):
        self.preview_fps = 15
        self.preview_duration = 10
        self.preview_resolution = (640, 360)
        self.text_overlay = TextOverlayGenerator()
        self.effects_engine = EffectsEngine()
        
    def generate_quick_preview(self, project: VideoProject, progress_callback=None, 
                             status_callback=None) -> str:
        """Generate a quick, low-quality preview"""
        
        try:
            if status_callback:
                status_callback("🔄 Generating quick preview...")
            
            temp_dir = tempfile.mkdtemp()
            
            # Process only first few clips for preview
            max_clips = min(3, len(project.media_items))
            preview_items = project.media_items[:max_clips]
            
            clips = []
            total_preview_duration = 0
            
            for i, media_item in enumerate(preview_items):
                if progress_callback:
                    progress_callback(i / max_clips * 0.8)
                
                if status_callback:
                    status_callback(f"📁 Processing preview clip {i+1}/{max_clips}...")
                
                preview_duration = min(media_item.duration, 3.0)
                total_preview_duration += preview_duration
                
                if total_preview_duration > self.preview_duration:
                    preview_duration = self.preview_duration - (total_preview_duration - preview_duration)
                    if preview_duration <= 0:
                        break
                
                clip = self._process_preview_media_item(media_item, preview_duration)
                if clip:
                    clips.append(clip)
                
                if total_preview_duration >= self.preview_duration:
                    break
            
            if not clips:
                raise Exception("No valid clips for preview")
            
            if status_callback:
                status_callback("🎬 Assembling preview...")
            
            # Combine clips
            final_clip = concatenate_videoclips(clips)
            
            # Add preview overlay using custom method
            preview_overlay = self._create_preview_overlay(final_clip.size, final_clip.duration)
            if preview_overlay:
                final_clip = CompositeVideoClip([final_clip, preview_overlay])
            
            if status_callback:
                status_callback("💾 Rendering preview...")
            
            output_path = os.path.join(temp_dir, f"preview_{project.title}_{int(time.time())}.mp4")
            
            final_clip.write_videofile(
                output_path,
                fps=self.preview_fps,
                codec='libx264',
                bitrate='500k',
                audio_codec='aac',
                verbose=False,
                logger=None,
                preset='ultrafast'
            )
            
            # Clean up
            final_clip.close()
            for clip in clips:
                clip.close()
            
            if progress_callback:
                progress_callback(1.0)
            
            if status_callback:
                status_callback("✅ Preview generated!")
            
            return output_path
            
        except Exception as e:
            if status_callback:
                status_callback(f"❌ Preview error: {str(e)}")
            raise e
    
    def _create_preview_overlay(self, size: Tuple[int, int], duration: float) -> VideoFileClip:
        """Create preview text overlay using PIL"""
        
        try:
            # Create text overlay image
            overlay_img = self.text_overlay.create_text_overlay(
                "PREVIEW", 
                size, 
                position='top-right',
                font_size=min(48, size[0] // 15),
                color=(255, 255, 0, 200)  # Yellow with transparency
            )
            
            # Convert RGBA to RGB with background
            if overlay_img.shape[2] == 4:
                alpha = overlay_img[:, :, 3:4] / 255.0
                rgb = overlay_img[:, :, :3]
                # Create transparent effect
                background = np.zeros_like(rgb)
                blended = background * (1 - alpha) + rgb * alpha
                overlay_img = blended.astype(np.uint8)
            
            # Create video clip from overlay
            overlay_clip = ImageClip(overlay_img, duration=duration)
            overlay_clip = overlay_clip.set_opacity(0.8)
            
            return overlay_clip
            
        except Exception:
            return None
    
    def _process_preview_media_item(self, media_item: MediaItem, duration: float) -> VideoFileClip:
        """Process media item for preview"""
        
        try:
            if media_item.media_type == 'image':
                return self._process_preview_image(media_item, duration)
            elif media_item.media_type == 'video':
                return self._process_preview_video(media_item, duration)
            else:
                return None
        except Exception:
            return None
    
    def _process_preview_image(self, media_item: MediaItem, duration: float) -> VideoFileClip:
        """Process image for preview"""
        
        try:
            # Load image
            if media_item.file_path.lower().endswith(('.heic', '.heif')) and HEIC_SUPPORT:
                image = Image.open(media_item.file_path)
            else:
                image = Image.open(media_item.file_path)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for preview
            image.thumbnail(self.preview_resolution, Image.Resampling.LANCZOS)
            
            # Create preview-sized canvas
            canvas = Image.new('RGB', self.preview_resolution, (0, 0, 0))
            x = (self.preview_resolution[0] - image.width) // 2
            y = (self.preview_resolution[1] - image.height) // 2
            canvas.paste(image, (x, y))
            
            # Convert to array
            image_array = np.array(canvas)
            
            # Apply basic effects
            for effect in media_item.effects:
                if effect == 'blur':
                    image_array = self.effects_engine.apply_blur(image_array)
                elif effect == 'sharpen':
                    image_array = self.effects_engine.apply_sharpen(image_array)
                elif effect == 'vintage':
                    image_array = self.effects_engine.apply_vintage(image_array)
                elif effect == 'glitch':
                    image_array = self.effects_engine.apply_glitch(image_array)
                elif effect == 'neon':
                    image_array = self.effects_engine.apply_neon(image_array)
                elif effect == 'black_white':
                    image_array = self.effects_engine.apply_black_white(image_array)
            
            # Create clip
            clip = ImageClip(image_array, duration=duration)
            
            return clip
            
        except Exception:
            return None
    
    def _process_preview_video(self, media_item: MediaItem, duration: float) -> VideoFileClip:
        """Process video for preview"""
        
        try:
            clip = VideoFileClip(media_item.file_path)
            clip = clip.resize(self.preview_resolution)
            clip = clip.set_duration(min(duration, clip.duration))
            
            if media_item.speed != 1.0:
                clip = clip.fx(speedx, media_item.speed)
            
            return clip
            
        except Exception:
            return None

class FontManager:
    """Manages system fonts and provides font utilities"""
    
    def __init__(self):
        self.font_cache = {}
        self.available_fonts = self._discover_system_fonts()
    
    def _discover_system_fonts(self) -> Dict[str, str]:
        """Discover available system fonts"""
        fonts = {}
        
        # Common font locations by OS
        font_directories = [
            "C:/Windows/Fonts/",  # Windows
            "/System/Library/Fonts/",  # macOS
            "/Library/Fonts/",  # macOS user fonts
            "/usr/share/fonts/",  # Linux
            "/usr/local/share/fonts/",  # Linux local
        ]
        
        # Common font mappings
        font_files = {
            "Arial": ["arial.ttf", "Arial.ttf"],
            "Times New Roman": ["times.ttf", "Times.ttf"],
            "Helvetica": ["Helvetica.ttc", "helvetica.ttf"],
            "Calibri": ["calibri.ttf", "Calibri.ttf"],
            "Comic Sans MS": ["comic.ttf", "ComicSansMS.ttf"],
            "Courier New": ["cour.ttf", "CourierNew.ttf"],
            "Georgia": ["georgia.ttf", "Georgia.ttf"],
            "Impact": ["impact.ttf", "Impact.ttf"],
            "Tahoma": ["tahoma.ttf", "Tahoma.ttf"],
            "Verdana": ["verdana.ttf", "Verdana.ttf"],
            "Open Sans": ["OpenSans-Regular.ttf"],
            "Roboto": ["Roboto-Regular.ttf"],
        }
        
        for font_name, filenames in font_files.items():
            for directory in font_directories:
                if not os.path.exists(directory):
                    continue
                
                for filename in filenames:
                    font_path = os.path.join(directory, filename)
                    if os.path.exists(font_path):
                        fonts[font_name] = font_path
                        break
                        
                if font_name in fonts:
                    break
        
        # Add fallback
        if not fonts:
            fonts["Default"] = None
            
        return fonts
    
    def get_font(self, font_name: str, size: int) -> ImageFont.FreeTypeFont:
        """Get font object with caching"""
        cache_key = f"{font_name}_{size}"
        
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        try:
            font_path = self.available_fonts.get(font_name)
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        self.font_cache[cache_key] = font
        return font
    
    def get_available_fonts(self) -> List[str]:
        """Get list of available font names"""
        return list(self.available_fonts.keys())

class TextRenderer:
    """Handles text rendering with advanced formatting"""
    
    def __init__(self):
        self.font_manager = FontManager()
        self._render_cache = {}  # Cache for rendered text overlays
        self._cache_max_size = 100
    
    def _get_cache_key(self, config: CustomTextOverlay, canvas_size: Tuple[int, int], frame_time: float) -> str:
        """Generate cache key for text overlay"""
        
        # Round frame time to reduce cache misses
        rounded_time = round(frame_time, 2)
        
        key_parts = [
            config.text,
            config.font_family,
            str(config.font_size),
            str(config.font_color),
            str(config.position.value),
            str(canvas_size),
            str(rounded_time),
            str(config.effect.value)
        ]
        
        return "|".join(key_parts)
    
    def render_text_overlay(self, config: CustomTextOverlay, canvas_size: Tuple[int, int], 
                          frame_time: float = 0.0) -> Optional[np.ndarray]:
        """Render text overlay with caching for performance"""
        
        if not config.enabled or not config.text.strip():
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(config, canvas_size, frame_time)
        if cache_key in self._render_cache:
            return self._render_cache[cache_key].copy()
        
        # Render new overlay
        overlay = self._render_text_overlay_internal(config, canvas_size, frame_time)
        
        # Cache the result (with size limit)
        if len(self._render_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._render_cache))
            del self._render_cache[oldest_key]
        
        if overlay is not None:
            self._render_cache[cache_key] = overlay.copy()
        
        return overlay
    
    def _render_text_overlay_internal(self, config: CustomTextOverlay, canvas_size: Tuple[int, int], 
                                    frame_time: float = 0.0) -> Optional[np.ndarray]:
        """Internal method for actual text rendering (original render_text_overlay logic)"""
        
        if not config.enabled or not config.text.strip():
            return None
        
        try:
            # Check timing
            if config.duration > 0:
                if frame_time < config.start_time or frame_time > config.start_time + config.duration:
                    return None
            elif frame_time < config.start_time:
                return None
            
            # Create transparent canvas
            canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            
            # Get font
            font = self.font_manager.get_font(config.font_family, 
                                            int(config.font_size * config.scale))
            
            # Calculate text dimensions and position
            text_bbox = draw.textbbox((0, 0), config.text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x, y = self._calculate_position(config, canvas_size, text_width, text_height)
            
            # Apply effects
            effect_offset, current_opacity = self._calculate_effects(
                config, frame_time, canvas_size, text_width, text_height
            )
            
            x += effect_offset[0]
            y += effect_offset[1]
            
            # Draw background if enabled
            if config.background_enabled:
                self._draw_background(draw, config, x, y, text_width, text_height, current_opacity)
            
            # Draw shadow if enabled
            if config.shadow_enabled:
                self._draw_shadow(draw, config, x, y, font, current_opacity)
            
            # Draw outline if enabled
            if config.outline_enabled:
                self._draw_outline(draw, config, x, y, font, current_opacity)
            
            # Draw main text
            self._draw_main_text(draw, config, x, y, font, current_opacity, frame_time)
            
            # Apply rotation if needed
            if config.rotation != 0:
                canvas = self._apply_rotation(canvas, config.rotation, canvas_size)
            
            return np.array(canvas)
            
        except Exception as e:
            st.warning(f"Error rendering text overlay: {str(e)}")
            return None
    
    def _calculate_position(self, config: CustomTextOverlay, canvas_size: Tuple[int, int],
                          text_width: int, text_height: int) -> Tuple[int, int]:
        """Calculate text position based on configuration"""
        
        width, height = canvas_size
        margin = 20
        
        if config.position == TextPosition.CUSTOM:
            x = int(config.custom_x * width)
            y = int(config.custom_y * height)
            return (x, y)
        
        position_map = {
            TextPosition.TOP_LEFT: (margin, margin),
            TextPosition.TOP_CENTER: ((width - text_width) // 2, margin),
            TextPosition.TOP_RIGHT: (width - text_width - margin, margin),
            TextPosition.CENTER_LEFT: (margin, (height - text_height) // 2),
            TextPosition.CENTER: ((width - text_width) // 2, (height - text_height) // 2),
            TextPosition.CENTER_RIGHT: (width - text_width - margin, (height - text_height) // 2),
            TextPosition.BOTTOM_LEFT: (margin, height - text_height - margin),
            TextPosition.BOTTOM_CENTER: ((width - text_width) // 2, height - text_height - margin),
            TextPosition.BOTTOM_RIGHT: (width - text_width - margin, height - text_height - margin),
        }
        
        return position_map.get(config.position, position_map[TextPosition.CENTER])
    
    def _calculate_effects(self, config: CustomTextOverlay, frame_time: float, 
                         canvas_size: Tuple[int, int], text_width: int, text_height: int) -> Tuple[Tuple[float, float], float]:
        """Calculate effects offsets and current opacity"""
        
        effect_offset = (0, 0)
        current_opacity = config.opacity
        
        # Calculate time progress within the text duration
        if config.duration > 0:
            text_end_time = config.start_time + config.duration
            if frame_time < config.start_time or frame_time > text_end_time:
                return effect_offset, 0.0
            
            progress = (frame_time - config.start_time) / config.duration
        else:
            progress = min(1.0, (frame_time - config.start_time) / 2.0) if frame_time >= config.start_time else 0.0
        
        progress = max(0.0, min(1.0, progress))
        
        # Handle different effects
        if config.effect == TextEffect.FADE_IN:
            current_opacity = config.opacity * progress
            
        elif config.effect == TextEffect.FADE_OUT:
            current_opacity = config.opacity * (1 - progress)
            
        elif config.effect == TextEffect.SLIDE_LEFT:
            slide_distance = canvas_size[0] * 0.2  # 20% of width
            effect_offset = (-slide_distance * (1 - progress), 0)
            
        elif config.effect == TextEffect.SLIDE_RIGHT:
            slide_distance = canvas_size[0] * 0.2
            effect_offset = (slide_distance * (1 - progress), 0)
            
        elif config.effect == TextEffect.SLIDE_UP:
            slide_distance = canvas_size[1] * 0.2  # 20% of height
            effect_offset = (0, -slide_distance * (1 - progress))
            
        elif config.effect == TextEffect.SLIDE_DOWN:
            slide_distance = canvas_size[1] * 0.2
            effect_offset = (0, slide_distance * (1 - progress))
            
        elif config.effect == TextEffect.TYPE_WRITE:
            # Progressive text reveal
            chars_to_show = int(len(config.text) * progress)
            if hasattr(config, '_typewriter_text'):
                config._typewriter_text = config.text[:chars_to_show]
            
        elif config.effect == TextEffect.GLOW:
            # Pulsing glow effect
            glow_intensity = 0.5 + 0.5 * np.sin(frame_time * 4)  # 4 Hz pulse
            current_opacity = config.opacity * glow_intensity
            
        elif config.effect == TextEffect.BOUNCE:
            # Bouncing effect
            bounce_height = 20 * np.abs(np.sin(frame_time * 6))  # 6 Hz bounce
            effect_offset = (0, -bounce_height)
        
        return effect_offset, current_opacity
    
    def _draw_background(self, draw: ImageDraw.Draw, config: CustomTextOverlay, 
                       x: float, y: float, text_width: float, text_height: float, opacity: float):
        """Draw background rectangle behind text"""
        
        try:
            if not config.background_enabled:
                return
                
            color = config.background_color
            bg_alpha = int(color[3] * opacity / 255.0 * 255)
            bg_color = (color[0], color[1], color[2], bg_alpha)
            
            # Add padding around text
            padding = 10
            bg_rect = [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding
            ]
            
            # Create a separate image for background with transparency
            bg_img = Image.new('RGBA', (int(bg_rect[2] - bg_rect[0]), int(bg_rect[3] - bg_rect[1])), bg_color)
            
            # This would need to be composited properly - simplified for now
            draw.rectangle(bg_rect, fill=bg_color)
            
        except Exception as e:
            pass
    
    def _draw_shadow(self, draw: ImageDraw.Draw, config: CustomTextOverlay, 
                   x: float, y: float, font: ImageFont.FreeTypeFont, opacity: float):
        """Draw shadow effect for text"""
        
        try:
            if not config.shadow_enabled:
                return
                
            shadow_color = config.shadow_color
            shadow_alpha = int(255 * opacity * 0.7)  # Shadows are slightly transparent
            shadow_color_with_alpha = (shadow_color[0], shadow_color[1], shadow_color[2], shadow_alpha)
            
            shadow_x = x + config.shadow_offset_x
            shadow_y = y + config.shadow_offset_y
            
            # Draw shadow text
            text_to_draw = config.text
            if config.effect == TextEffect.TYPE_WRITE and hasattr(config, '_typewriter_text'):
                text_to_draw = config._typewriter_text
                
            draw.text((shadow_x, shadow_y), text_to_draw, font=font, fill=shadow_color_with_alpha)
            
        except Exception:
            pass
    
    def _draw_outline(self, draw: ImageDraw.Draw, config: CustomTextOverlay, 
                    x: float, y: float, font: ImageFont.FreeTypeFont, opacity: float):
        """Draw outline effect for text"""
        
        try:
            if not config.outline_enabled:
                return
                
            outline_color = config.outline_color
            outline_alpha = int(255 * opacity)
            outline_color_with_alpha = (outline_color[0], outline_color[1], outline_color[2], outline_alpha)
            
            text_to_draw = config.text
            if config.effect == TextEffect.TYPE_WRITE and hasattr(config, '_typewriter_text'):
                text_to_draw = config._typewriter_text
            
            # Draw outline by drawing text in multiple positions
            outline_width = config.outline_width
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), text_to_draw, font=font, fill=outline_color_with_alpha)
                    
        except Exception:
            pass
    
    def _draw_main_text(self, draw: ImageDraw.Draw, config: CustomTextOverlay, 
                      x: float, y: float, font: ImageFont.FreeTypeFont, opacity: float, frame_time: float):
        """Draw the main text with effects"""
        
        try:
            text_color = config.font_color
            text_alpha = int(255 * opacity)
            text_color_with_alpha = (text_color[0], text_color[1], text_color[2], text_alpha)
            
            text_to_draw = config.text
            if config.effect == TextEffect.TYPE_WRITE:
                # Calculate characters to show based on progress
                if config.duration > 0:
                    progress = (frame_time - config.start_time) / config.duration
                else:
                    progress = min(1.0, (frame_time - config.start_time) / 2.0)
                progress = max(0.0, min(1.0, progress))
                chars_to_show = int(len(config.text) * progress)
                text_to_draw = config.text[:chars_to_show]
            
            # Apply glow effect if enabled
            if config.effect == TextEffect.GLOW:
                # Draw multiple layers for glow
                glow_color = (min(255, text_color[0] + 50), min(255, text_color[1] + 50), min(255, text_color[2] + 50), int(text_alpha * 0.5))
                for i in range(1, 4):
                    draw.text((x + i, y + i), text_to_draw, font=font, fill=glow_color)
                    draw.text((x - i, y - i), text_to_draw, font=font, fill=glow_color)
            
            # Draw main text
            draw.text((x, y), text_to_draw, font=font, fill=text_color_with_alpha)
            
        except Exception:
            pass
    
    def _apply_rotation(self, image: Image.Image, angle: float, canvas_size: Tuple[int, int]) -> Image.Image:
        """Apply rotation to image"""
        try:
            if angle == 0:
                return image
                
            # Rotate image
            rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False)
            
            # If image was expanded, crop back to original size
            if rotated.size != canvas_size:
                # Center crop
                left = (rotated.width - canvas_size[0]) // 2
                top = (rotated.height - canvas_size[1]) // 2
                right = left + canvas_size[0]
                bottom = top + canvas_size[1]
                
                rotated = rotated.crop((left, top, right, bottom))
            
            return rotated
            
        except Exception:
            return image

class PerformanceMonitor:
    """Monitor system performance during video generation"""
    
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        self.is_monitoring = False
        
    def start_monitoring(self):
        self.is_monitoring = True
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.timestamps.clear()
        
    def stop_monitoring(self):
        self.is_monitoring = False
        
    def record_metrics(self):
        if self.is_monitoring:
            try:
                self.cpu_usage.append(psutil.cpu_percent())
                self.memory_usage.append(psutil.virtual_memory().percent)
                self.timestamps.append(datetime.now())
            except Exception:
                pass
def render_text_overlay_ui(media_item: MediaItem, item_index: int) -> CustomTextOverlay:
    """Render comprehensive text overlay configuration UI for a media item"""

    # Initialize text overlay if not exists
    if not hasattr(media_item, 'text_overlay') or media_item.text_overlay is None:
        media_item.text_overlay = CustomTextOverlay()

    text_config = media_item.text_overlay

    # Text Overlay Section (single expander only)
    with st.expander(f"📝 Text Overlay for {Path(media_item.file_path).stem}", expanded=False):
        col_text1, col_text2 = st.columns(2)

        with col_text1:
            # Enable/Disable Text Overlay
            text_config.enabled = st.checkbox(
                "Enable Text Overlay",
                value=text_config.enabled,
                key=f"text_enabled_{item_index}",
                help="Add custom text overlay to this clip"
            )

            if text_config.enabled:
                # Text Content
                text_config.text = st.text_area(
                    "Text Content",
                    value=text_config.text,
                    height=80,
                    key=f"text_content_{item_index}",
                    help="Enter the text to display on this clip",
                    placeholder="Enter your text here..."
                )

                # Font Settings
                st.markdown("**🔤 Font Settings**")
                font_manager = FontManager()
                available_fonts = font_manager.get_available_fonts()
                if available_fonts:
                    try:
                        font_index = available_fonts.index(text_config.font_family)
                    except ValueError:
                        font_index = 0
                    text_config.font_family = st.selectbox(
                        "Font Family",
                        options=available_fonts,
                        index=font_index,
                        key=f"font_family_{item_index}",
                        help="Choose from available system fonts"
                    )
                else:
                    text_config.font_family = "Arial"
                    st.info("Using default font (system fonts not detected)")

                text_config.font_size = st.slider(
                    "Font Size",
                    min_value=12,
                    max_value=200,
                    value=text_config.font_size,
                    key=f"font_size_{item_index}",
                    help="Size of the text in pixels"
                )

                # Color Settings
                st.markdown("**🎨 Color Settings**")
                font_color_hex = st.color_picker(
                    "Text Color",
                    value=f"#{text_config.font_color[0]:02x}{text_config.font_color[1]:02x}{text_config.font_color[2]:02x}",
                    key=f"font_color_{item_index}",
                    help="Primary color of the text"
                )
                text_config.font_color = tuple(int(font_color_hex[i:i+2], 16) for i in (1, 3, 5))

                # Outline Settings (no nested columns)
                text_config.outline_enabled = st.checkbox(
                    "Enable Outline",
                    value=text_config.outline_enabled,
                    key=f"outline_enabled_{item_index}",
                    help="Add an outline around the text for better visibility"
                )
                if text_config.outline_enabled:
                    outline_color_hex = st.color_picker(
                        "Outline Color",
                        value=f"#{text_config.outline_color[0]:02x}{text_config.outline_color[1]:02x}{text_config.outline_color[2]:02x}",
                        key=f"outline_color_{item_index}"
                    )
                    text_config.outline_color = tuple(int(outline_color_hex[i:i+2], 16) for i in (1, 3, 5))
                    text_config.outline_width = st.slider(
                        "Outline Width",
                        min_value=1,
                        max_value=10,
                        value=text_config.outline_width,
                        key=f"outline_width_{item_index}",
                        help="Thickness of the outline"
                    )

        with col_text2:
            if text_config.enabled:
                # Position Settings
                st.markdown("**📍 Position Settings**")
                position_options = {
                    "Top Left": TextPosition.TOP_LEFT,
                    "Top Center": TextPosition.TOP_CENTER,
                    "Top Right": TextPosition.TOP_RIGHT,
                    "Center Left": TextPosition.CENTER_LEFT,
                    "Center": TextPosition.CENTER,
                    "Center Right": TextPosition.CENTER_RIGHT,
                    "Bottom Left": TextPosition.BOTTOM_LEFT,
                    "Bottom Center": TextPosition.BOTTOM_CENTER,
                    "Bottom Right": TextPosition.BOTTOM_RIGHT,
                    "Custom Position": TextPosition.CUSTOM
                }
                current_position_name = next(
                    (name for name, pos in position_options.items() if pos == text_config.position),
                    "Center"
                )
                selected_position = st.selectbox(
                    "Text Position",
                    options=list(position_options.keys()),
                    index=list(position_options.keys()).index(current_position_name),
                    key=f"text_position_{item_index}",
                    help="Where to place the text on the screen"
                )
                text_config.position = position_options[selected_position]

                # Custom Position Controls (no nested columns)
                if text_config.position == TextPosition.CUSTOM:
                    text_config.custom_x = st.slider(
                        "X Position (%)",
                        min_value=0.0,
                        max_value=1.0,
                        value=text_config.custom_x,
                        step=0.01,
                        key=f"custom_x_{item_index}",
                        help="0.0 = left edge, 1.0 = right edge"
                    )
                    text_config.custom_y = st.slider(
                        "Y Position (%)",
                        min_value=0.0,
                        max_value=1.0,
                        value=text_config.custom_y,
                        step=0.01,
                        key=f"custom_y_{item_index}",
                        help="0.0 = top edge, 1.0 = bottom edge"
                    )

                # Timing Settings (no nested columns)
                st.markdown("**⏰ Timing Settings**")
                text_config.start_time = st.number_input(
                    "Start Time (s)",
                    min_value=0.0,
                    max_value=media_item.duration,
                    value=text_config.start_time,
                    step=0.1,
                    key=f"text_start_{item_index}",
                    help="When the text should appear"
                )
                text_config.duration = st.number_input(
                    "Text Duration (s)",
                    min_value=0.0,
                    max_value=media_item.duration,
                    value=text_config.duration if text_config.duration > 0 else media_item.duration,
                    step=0.1,
                    key=f"text_duration_{item_index}",
                    help="How long text is visible (0 = full clip)"
                )

                # Opacity
                text_config.opacity = st.slider(
                    "Text Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    value=text_config.opacity,
                    step=0.05,
                    key=f"text_opacity_{item_index}",
                    help="Transparency of the text (1.0 = fully opaque)"
                )

        # Advanced Settings (NO expander, just a heading)
        if text_config.enabled:
            st.markdown("### 🔧 Advanced Text Settings")
            col_adv1, col_adv2, col_adv3 = st.columns(3)

            with col_adv1:
                # Text Effects
                st.markdown("**✨ Animation Effects**")
                effect_options = {
                    "None": TextEffect.NONE,
                    "Fade In": TextEffect.FADE_IN,
                    "Fade Out": TextEffect.FADE_OUT,
                    "Slide Left": TextEffect.SLIDE_LEFT,
                    "Slide Right": TextEffect.SLIDE_RIGHT,
                    "Slide Up": TextEffect.SLIDE_UP,
                    "Slide Down": TextEffect.SLIDE_DOWN,
                    "Type Writer": TextEffect.TYPE_WRITE,
                    "Glow": TextEffect.GLOW,
                    "Bounce": TextEffect.BOUNCE
                }
                current_effect_name = next(
                    (name for name, effect in effect_options.items() if effect == text_config.effect),
                    "None"
                )
                selected_effect = st.selectbox(
                    "Text Effect",
                    options=list(effect_options.keys()),
                    index=list(effect_options.keys()).index(current_effect_name),
                    key=f"text_effect_{item_index}",
                    help="Animation effect for the text appearance"
                )
                text_config.effect = effect_options[selected_effect]
                text_config.scale = st.slider(
                    "Scale",
                    min_value=0.1,
                    max_value=3.0,
                    value=text_config.scale,
                    step=0.1,
                    key=f"text_scale_{item_index}",
                    help="Size multiplier for the text"
                )
                text_config.rotation = st.slider(
                    "Rotation (degrees)",
                    min_value=-180.0,
                    max_value=180.0,
                    value=text_config.rotation,
                    step=1.0,
                    key=f"text_rotation_{item_index}",
                    help="Rotate the text (0 = horizontal)"
                )

            with col_adv2:
                # Background Settings
                st.markdown("**📦 Background**")
                text_config.background_enabled = st.checkbox(
                    "Enable Background",
                    value=text_config.background_enabled,
                    key=f"bg_enabled_{item_index}",
                    help="Add a colored background behind the text"
                )
                if text_config.background_enabled:
                    bg_color_hex = st.color_picker(
                        "Background Color",
                        value=f"#{text_config.background_color[0]:02x}{text_config.background_color[1]:02x}{text_config.background_color[2]:02x}",
                        key=f"bg_color_{item_index}",
                        help="Color of the background rectangle"
                    )
                    bg_rgb = tuple(int(bg_color_hex[i:i+2], 16) for i in (1, 3, 5))
                    bg_alpha = st.slider(
                        "Background Opacity",
                        min_value=0,
                        max_value=255,
                        value=text_config.background_color[3],
                        key=f"bg_alpha_{item_index}",
                        help="Transparency of the background (255 = opaque)"
                    )
                    text_config.background_color = bg_rgb + (bg_alpha,)

            with col_adv3:
                # Shadow Settings (no nested columns)
                st.markdown("**🌑 Shadow**")
                text_config.shadow_enabled = st.checkbox(
                    "Enable Shadow",
                    value=text_config.shadow_enabled,
                    key=f"shadow_enabled_{item_index}",
                    help="Add a drop shadow effect to the text"
                )
                if text_config.shadow_enabled:
                    shadow_color_hex = st.color_picker(
                        "Shadow Color",
                        value=f"#{text_config.shadow_color[0]:02x}{text_config.shadow_color[1]:02x}{text_config.shadow_color[2]:02x}",
                        key=f"shadow_color_{item_index}",
                        help="Color of the drop shadow"
                    )
                    text_config.shadow_color = tuple(int(shadow_color_hex[i:i+2], 16) for i in (1, 3, 5))
                    text_config.shadow_offset_x = st.slider(
                        "Shadow X Offset",
                        min_value=-20,
                        max_value=20,
                        value=text_config.shadow_offset_x,
                        key=f"shadow_x_{item_index}",
                        help="Horizontal shadow displacement"
                    )
                    text_config.shadow_offset_y = st.slider(
                        "Shadow Y Offset",
                        min_value=-20,
                        max_value=20,
                        value=text_config.shadow_offset_y,
                        key=f"shadow_y_{item_index}",
                        help="Vertical shadow displacement"
                    )

        # Preview Section
        if text_config.enabled and text_config.text.strip():
            st.markdown("**👀 Text Preview**")
            preview_style = f"""
            <div style='
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                text-align: center;
                font-family: {text_config.font_family}, sans-serif;
                font-size: {min(text_config.font_size, 24)}px;
                position: relative;
            '>
                <h4 style='margin: 0; color: white;'>📝 "{text_config.text}"</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 12px; opacity: 0.8;'>
                    Font: {text_config.font_family} • Size: {text_config.font_size}px
                    {f" • Position: {selected_position}" if 'selected_position' in locals() else ""}
                </p>
            </div>
            """
            st.markdown(preview_style, unsafe_allow_html=True)
            settings_info = []
            if text_config.effect != TextEffect.NONE:
                settings_info.append(f"Effect: {text_config.effect.value.replace('_', ' ').title()}")
            if text_config.outline_enabled:
                settings_info.append(f"Outline: {text_config.outline_width}px")
            if text_config.background_enabled:
                settings_info.append("Background: Enabled")
            if text_config.shadow_enabled:
                settings_info.append("Shadow: Enabled")
            if text_config.rotation != 0:
                settings_info.append(f"Rotation: {text_config.rotation}°")
            if text_config.scale != 1.0:
                settings_info.append(f"Scale: {text_config.scale}x")
            if settings_info:
                st.caption(" • ".join(settings_info))
            if text_config.duration > 0:
                timing_info = f"⏰ Appears at {text_config.start_time:.1f}s for {text_config.duration:.1f}s"
            else:
                timing_info = f"⏰ Appears at {text_config.start_time:.1f}s for full clip duration"
            st.caption(timing_info)
            # Quick preset buttons (side by side)
            st.markdown("**⚡ Quick Presets**")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🏷️ Title Style", key=f"preset_title_{item_index}", help="Large, centered, bold style"):
                    text_config.font_size = 72
                    text_config.position = TextPosition.TOP_CENTER
                    text_config.outline_enabled = True
                    text_config.outline_width = 3
                    text_config.background_enabled = True
                    text_config.background_color = (0, 0, 0, 128)
                    st.rerun()
            with c2:
                if st.button("💬 Subtitle Style", key=f"preset_subtitle_{item_index}", help="Medium, bottom, readable style"):
                    text_config.font_size = 36
                    text_config.position = TextPosition.BOTTOM_CENTER
                    text_config.outline_enabled = True
                    text_config.outline_width = 2
                    text_config.background_enabled = True
                    text_config.background_color = (0, 0, 0, 180)
                    st.rerun()
            with c3:
                if st.button("🏆 Watermark Style", key=f"preset_watermark_{item_index}", help="Small, corner, transparent style"):
                    text_config.font_size = 24
                    text_config.position = TextPosition.BOTTOM_RIGHT
                    text_config.opacity = 0.7
                    text_config.outline_enabled = False
                    text_config.background_enabled = False
                    st.rerun()

    return text_config
class MediaProcessor:
    """Handle media file processing and optimization"""
    
    def __init__(self):
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic', '.heif'}
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        self.supported_audio_formats = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'}
        self.effects_engine = EffectsEngine()
        
    def detect_media_type(self, file_path: str) -> str:
        """Detect media type from file extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext in self.supported_image_formats:
            return 'image'
        elif ext in self.supported_video_formats:
            return 'video'
        elif ext in self.supported_audio_formats:
            return 'audio'
        else:
            return 'unknown'
    
    def process_heic_image(self, heic_path: str) -> np.ndarray:
        """Convert HEIC image to numpy array"""
        try:
            if not HEIC_SUPPORT:
                st.error("HEIC support not available. Please install pillow-heif.")
                return None
                
            image = Image.open(heic_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            st.error(f"Error processing HEIC file: {str(e)}")
            return None
    
    def extract_video_info(self, video_path: str) -> Dict:
        """Extract video information"""
        try:
            clip = VideoFileClip(video_path)
            info = {
                'duration': clip.duration,
                'width': clip.w,
                'height': clip.h,
                'fps': clip.fps,
                'codec': 'unknown'
            }
            clip.close()
            return info
        except Exception:
            return {'duration': 10.0, 'width': 1920, 'height': 1080, 'fps': 30, 'codec': 'unknown'}

class VideoGenerator:
    """Main video generation engine"""
    
    def __init__(self):
        self.media_processor = MediaProcessor()
        self.performance_monitor = PerformanceMonitor()
        self.preview_generator = PreviewGenerator()
        self.effects_engine = EffectsEngine()
        
    def generate_video(self, project: VideoProject, progress_callback=None, 
                      status_callback=None) -> str:
        """Generate video from project configuration"""
        
        try:
            self.performance_monitor.start_monitoring()
            
            if status_callback:
                status_callback("🚀 Starting video generation...")
            
            temp_dir = tempfile.mkdtemp()
            
            clips = []
            total_items = len(project.media_items)
            
            for i, media_item in enumerate(project.media_items):
                if progress_callback:
                    progress_callback(i / total_items * 0.8)
                
                if status_callback:
                    status_callback(f"📁 Processing {Path(media_item.file_path).name}...")
                
                clip = self._process_media_item(media_item, project)
                if clip:
                    clips.append(clip)
                
                self.performance_monitor.record_metrics()
            
            if not clips:
                raise Exception("No valid media clips found")
            
            if status_callback:
                status_callback("🎬 Assembling video clips...")
            
            final_clip = concatenate_videoclips(clips)
            
            if status_callback:
                status_callback("🎵 Adding background music...")
            
            if project.background_music:
                final_clip = self._add_background_music(final_clip, project.background_music)
            
            if status_callback:
                status_callback("🎨 Applying global effects...")
            
            final_clip = self._apply_global_effects(final_clip, project.global_effects)
            
            if status_callback:
                status_callback("💾 Rendering final video...")
            
            output_path = os.path.join(temp_dir, f"{project.title}_{int(time.time())}.mp4")
            codec_settings = self._get_codec_settings(project.quality)
            
            final_clip.write_videofile(
                output_path,
                fps=project.fps,
                codec=codec_settings['codec'],
                bitrate=codec_settings['bitrate'],
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Clean up
            final_clip.close()
            for clip in clips:
                clip.close()
            
            self.performance_monitor.stop_monitoring()
            
            if progress_callback:
                progress_callback(1.0)
            
            if status_callback:
                status_callback("✅ Video generation completed!")
            
            return output_path
            
        except Exception as e:
            self.performance_monitor.stop_monitoring()
            if status_callback:
                status_callback(f"❌ Error: {str(e)}")
            raise e
    
    def _process_media_item(self, media_item: MediaItem, project: VideoProject) -> VideoFileClip:
        """Process individual media item with text overlay integration"""
        
        media_type = self.media_processor.detect_media_type(media_item.file_path)
        
        if media_type == 'image':
            base_clip = self._process_image(media_item, project)
        elif media_type == 'video':
            base_clip = self._process_video(media_item, project)
        else:
            return None
        
        if base_clip is None:
            return None
        
        # Add text overlay if enabled
        if media_item.text_overlay.enabled and media_item.text_overlay.text.strip():
            base_clip = self._add_text_overlay_to_clip(base_clip, media_item.text_overlay)
        
        return base_clip
    
    def _add_text_overlay_to_clip(self, clip: VideoFileClip, text_config: CustomTextOverlay) -> VideoFileClip:
        """Add text overlay to a video clip"""
        
        try:
            text_renderer = TextRenderer()
            
            def make_frame_with_text(get_frame, t):
                """Generate frame with text overlay"""
                
                # Get original frame
                frame = get_frame(t)
                
                # Render text overlay for current time
                text_overlay = text_renderer.render_text_overlay(
                    text_config,
                    (frame.shape[1], frame.shape[0]),  # (width, height)
                    frame_time=t
                )
                
                if text_overlay is not None:
                    # Blend text overlay with frame
                    if text_overlay.shape[2] == 4:  # RGBA overlay
                        # Extract alpha channel
                        alpha = text_overlay[:, :, 3:4] / 255.0
                        text_rgb = text_overlay[:, :, :3]
                        
                        # Ensure frame is the right shape
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            # Blend using alpha compositing
                            frame_float = frame.astype(np.float32)
                            text_float = text_rgb.astype(np.float32)
                            
                            blended = frame_float * (1 - alpha) + text_float * alpha
                            frame = np.clip(blended, 0, 255).astype(np.uint8)
                
                return frame
            
            # Create new clip with text overlay
            new_clip = clip.fl(lambda gf, t: make_frame_with_text(gf, t))
            
            return new_clip
            
        except Exception as e:
            st.warning(f"Error adding text overlay: {str(e)}")
            return clip
    
    def _process_image(self, media_item: MediaItem, project: VideoProject) -> VideoFileClip:
        """Process image file"""
        try:
            # Load image
            if media_item.file_path.lower().endswith(('.heic', '.heif')):
                image_array = self.media_processor.process_heic_image(media_item.file_path)
            else:
                image = Image.open(media_item.file_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_array = np.array(image)
            
            if image_array is None:
                return None
            
            # Resize to target resolution
            pil_image = Image.fromarray(image_array)
            pil_image.thumbnail(project.resolution, Image.Resampling.LANCZOS)
            
            # Create canvas
            canvas = Image.new('RGB', project.resolution, (0, 0, 0))
            x = (project.resolution[0] - pil_image.width) // 2
            y = (project.resolution[1] - pil_image.height) // 2
            canvas.paste(pil_image, (x, y))
            
            optimized_image = np.array(canvas)
            
            # Apply effects using our custom effects engine
            for effect in media_item.effects:
                if effect == 'blur':
                    optimized_image = self.effects_engine.apply_blur(optimized_image)
                elif effect == 'sharpen':
                    optimized_image = self.effects_engine.apply_sharpen(optimized_image)
                elif effect == 'vintage':
                    optimized_image = self.effects_engine.apply_vintage(optimized_image)
                elif effect == 'glitch':
                    optimized_image = self.effects_engine.apply_glitch(optimized_image)
                elif effect == 'neon':
                    optimized_image = self.effects_engine.apply_neon(optimized_image)
                elif effect == 'black_white':
                    optimized_image = self.effects_engine.apply_black_white(optimized_image)
            
            # Create clip
            clip = ImageClip(optimized_image, duration=media_item.duration)
            
            if media_item.speed != 1.0:
                clip = clip.fx(speedx, media_item.speed)
            
            return clip
        except Exception as e:
            st.warning(f"Error processing image {media_item.file_path}: {str(e)}")
            return None
    
    def _process_video(self, media_item: MediaItem, project: VideoProject) -> VideoFileClip:
        """Process video file"""
        try:
            clip = VideoFileClip(media_item.file_path)
            clip = clip.resize(project.resolution)
            
            if media_item.duration > 0:
                clip = clip.set_duration(min(media_item.duration, clip.duration))
            
            if media_item.speed != 1.0:
                clip = clip.fx(speedx, media_item.speed)
            
            if hasattr(clip, 'audio') and clip.audio:
                clip = clip.set_audio(clip.audio.fx(afx.volumex, media_item.volume))
            
            return clip
        except Exception as e:
            st.warning(f"Error processing video {media_item.file_path}: {str(e)}")
            return None
    
    def _add_background_music(self, video_clip: VideoFileClip, music_path: str) -> VideoFileClip:
        """Add background music to video"""
        try:
            audio_clip = AudioFileClip(music_path)
            
            if audio_clip.duration < video_clip.duration:
                loops_needed = int(video_clip.duration / audio_clip.duration) + 1
                audio_clip = concatenate_audioclips([audio_clip] * loops_needed)
            
            audio_clip = audio_clip.set_duration(video_clip.duration)
            
            if hasattr(video_clip, 'audio') and video_clip.audio:
                final_audio = CompositeAudioClip([video_clip.audio, audio_clip.fx(afx.volumex, 0.3)])
            else:
                final_audio = audio_clip.fx(afx.volumex, 0.5)
            
            return video_clip.set_audio(final_audio)
            
        except Exception as e:
            st.warning(f"Could not add background music: {str(e)}")
            return video_clip
    
    def _apply_global_effects(self, video_clip: VideoFileClip, effects: List[str]) -> VideoFileClip:
        """Apply global effects to entire video"""
        
        result_clip = video_clip
        
        try:
            for effect in effects:
                if effect == 'fade_in':
                    result_clip = result_clip.fadein(1.0)
                elif effect == 'fade_out':
                    result_clip = result_clip.fadeout(1.0)
                elif effect == 'brightness':
                    # Apply brightness using custom function
                    def apply_brightness_frame(frame):
                        return self.effects_engine.apply_brightness(frame, 1.2)
                    result_clip = result_clip.fl_image(apply_brightness_frame)
                elif effect == 'contrast':
                    # Apply contrast using custom function
                    def apply_contrast_frame(frame):
                        return self.effects_engine.apply_contrast(frame, 1.2)
                    result_clip = result_clip.fl_image(apply_contrast_frame)
        except Exception as e:
            st.warning(f"Error applying global effects: {str(e)}")
        
        return result_clip
    
    def _get_codec_settings(self, quality: VideoQuality) -> Dict:
        """Get codec settings based on quality"""
        
        if isinstance(quality, VideoQuality):
            quality_str = quality.value
        else:
            quality_str = str(quality).lower()
        
        settings = {
            'draft': {
                'codec': 'libx264',
                'bitrate': '1000k'
            },
            'standard': {
                'codec': 'libx264',
                'bitrate': '2000k'
            },
            'high': {
                'codec': 'libx264',
                'bitrate': '5000k'
            },
            'ultra': {
                'codec': 'libx264',
                'bitrate': '10000k'
            }
        }
        
        return settings.get(quality_str, settings['standard'])

# Initialize session state
if 'video_generator' not in st.session_state:
    st.session_state.video_generator = VideoGenerator()

if 'project' not in st.session_state:
    st.session_state.project = None

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Display system information
with st.sidebar.expander("💻 System Info", expanded=False):
    try:
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        st.metric("CPU Cores", cpu_count)
        st.metric("Total RAM", f"{memory.total / (1024**3):.1f} GB")
        st.metric("Available RAM", f"{memory.available / (1024**3):.1f} GB")
        
        if HEIC_SUPPORT:
            st.success("✅ HEIC Support Available")
        else:
            st.warning("⚠️ HEIC Support Missing")
            
        st.info("🎨 Using PIL/OpenCV (No ImageMagick)")
    except Exception:
        st.info("System info unavailable")

# Sidebar Configuration
st.sidebar.title("🎛️ Video Configuration")

# Project Settings
with st.sidebar.expander("📋 Project Settings", expanded=True):
    project_title = st.text_input("Project Title", value="My AI Video", key="project_title")
    
    quality_options = {
        "Draft (Fast)": VideoQuality.DRAFT,
        "Standard": VideoQuality.STANDARD,
        "High Quality": VideoQuality.HIGH,
        "Ultra HD": VideoQuality.ULTRA
    }
    
    selected_quality = st.selectbox(
        "Video Quality",
        options=list(quality_options.keys()),
        index=1,
        help="Higher quality takes longer to process"
    )
    
    resolution_options = {
        "720p (HD)": (1280, 720),
        "1080p (Full HD)": (1920, 1080),
        "1440p (2K)": (2560, 1440),
        "2160p (4K)": (3840, 2160)
    }
    
    selected_resolution = st.selectbox(
        "Resolution",
        options=list(resolution_options.keys()),
        index=1
    )
    
    fps = st.selectbox("Frame Rate (FPS)", [24, 30, 60], index=1)
    duration_minutes = st.slider("Target Duration (minutes)", 0.08, 13.0, 1.0, 0.08)
    total_duration = duration_minutes * 60

# Global Effects
with st.sidebar.expander("🎨 Global Effects"):
    global_effects = st.multiselect(
        "Apply to Entire Video",
        ["fade_in", "fade_out", "brightness", "contrast"],
        help="These effects will be applied to the entire video"
    )

# Background Music
with st.sidebar.expander("🎵 Background Music"):
    background_music_file = st.file_uploader(
        "Upload Background Music",
        type=['mp3', 'wav', 'aac', 'flac'],
        help="Audio will be looped to match video duration"
    )

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📁 Media Upload & Management")
    
    uploaded_files = st.file_uploader(
        "Upload Media Files",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'heic', 'heif',
              'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm',
              'mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a'],
        accept_multiple_files=True,
        help="Supported: Images (including HEIC), Videos, Audio files"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
        
        st.subheader("📋 Uploaded Media")
        
        media_items = []
        
        for i, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name}", expanded=True):
                # Main media settings columns
                col_a, col_b, col_c = st.columns([2, 1, 1])
                
                with col_a:
                    temp_path = os.path.join(tempfile.gettempdir(), file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(file.getbuffer())
                    
                    media_type = st.session_state.video_generator.media_processor.detect_media_type(temp_path)
                    
                    if media_type == 'image':
                        try:
                            if file.name.lower().endswith(('.heic', '.heif')) and HEIC_SUPPORT:
                                image_array = st.session_state.video_generator.media_processor.process_heic_image(temp_path)
                                if image_array is not None:
                                    st.image(image_array, width=200)
                                    st.caption("✅ HEIC/HEIF supported")
                            else:
                                st.image(file, width=200)
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
                    
                    elif media_type == 'video':
                        st.video(file)
                        video_info = st.session_state.video_generator.media_processor.extract_video_info(temp_path)
                        st.caption(f"Duration: {video_info['duration']:.1f}s, {video_info['width']}x{video_info['height']}")
                    
                    
                    elif media_type == 'audio':
                        st.audio(file)
                        st.caption("🎵 Audio file")
                
                with col_b:
                    if media_type in ['image', 'video']:
                        duration = st.number_input(
                            f"Duration (s)",
                            min_value=0.1,
                            max_value=60.0,
                            value=5.0 if media_type == 'image' else 10.0,
                            step=0.1,
                            key=f"duration_{i}"
                        )
                        
                        speed = st.slider(
                            "Speed",
                            0.25, 4.0, 1.0, 0.25,
                            key=f"speed_{i}",
                            help="1.0 = normal, >1.0 = faster, <1.0 = slower"
                        )
                    else:
                        duration = 5.0
                        speed = 1.0
                    
                    if media_type in ['video', 'audio']:
                        volume = st.slider(
                            "Volume",
                            0.0, 2.0, 1.0, 0.1,
                            key=f"volume_{i}"
                        )
                    else:
                        volume = 1.0
                
                with col_c:
                    if media_type in ['image', 'video']:
                        effects = st.multiselect(
                            "Effects",
                            ["blur", "sharpen", "vintage", "glitch", "neon", "black_white"],
                            key=f"effects_{i}"
                        )
                        
                        if i > 0:
                            transition_options = [
                                "cut", "fade", "crossfade",
                                "slide_left", "slide_right", "slide_up", "slide_down",
                                "zoom_in", "zoom_out"
                            ]
                            
                            transition = st.selectbox(
                                "Transition",
                                transition_options,
                                index=1,
                                key=f"transition_{i}"
                            )
                        else:
                            transition = "cut"
                    else:
                        effects = []
                        transition = "cut"
            
            # Create media item
            if media_type in ['image', 'video']:
                media_item = MediaItem(
                    file_path=temp_path,
                    media_type=media_type,
                    duration=duration,
                    start_time=0,
                    effects=effects,
                    transition=transition,
                    volume=volume if media_type == 'video' else 1.0,
                    speed=speed,
                    text_overlay=CustomTextOverlay()  # Initialize with default text overlay
                )
                
                # Render text overlay UI and update the media item
                text_config = render_text_overlay_ui(media_item, i)
                media_item.text_overlay = text_config
                
                media_items.append(media_item)
        
        if media_items and st.button("🚀 Create Video Project", type="primary", use_container_width=True):
            background_music_path = None
            if background_music_file:
                background_music_path = os.path.join(tempfile.gettempdir(), background_music_file.name)
                with open(background_music_path, 'wb') as f:
                    f.write(background_music_file.getbuffer())
            
            project = VideoProject(
                title=project_title,
                total_duration=total_duration,
                quality=quality_options[selected_quality],
                fps=fps,
                resolution=resolution_options[selected_resolution],
                background_music=background_music_path,
                media_items=media_items,
                global_effects=global_effects
            )
            
            st.session_state.project = project
            st.success("✅ Project created successfully!")

with col2:
    st.header("🎬 Video Generation")
    
    if st.session_state.project:
        project = st.session_state.project
        
        st.markdown("""
        <div class="feature-card">
            <h4>📋 Project Information</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.metric("🎬 Title", project.title)
            st.metric("📐 Resolution", f"{project.resolution[0]}×{project.resolution[1]}")
            st.metric("🎞️ Frame Rate", f"{project.fps} FPS")
        
        with col_info2:
            st.metric("⏱️ Duration", f"{project.total_duration:.1f}s")
            st.metric("🎯 Quality", project.quality.value.title())
            st.metric("📁 Media Items", len(project.media_items))
        
        st.markdown("""
        <div class="feature-card">
            <h4>🎮 Generation Controls</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_prev1, col_prev2 = st.columns(2)
        
        with col_prev1:
            if st.button("👁️ Quick Preview", use_container_width=True):
                progress_bar = st.progress(0)
                status_container = st.empty()
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                
                def update_status(status):
                    status_container.markdown(f"""
                    <div class="status-processing">{status}</div>
                    """, unsafe_allow_html=True)
                
                try:
                    with st.spinner("🔄 Generating preview..."):
                        preview_path = st.session_state.video_generator.preview_generator.generate_quick_preview(
                            project,
                            progress_callback=update_progress,
                            status_callback=update_status
                        )
                    
                    status_container.markdown("""
                    <div class="status-success">✅ Preview generated!</div>
                    """, unsafe_allow_html=True)
                    
                    with open(preview_path, 'rb') as f:
                        preview_bytes = f.read()
                    
                    st.markdown("""
                    <div class="preview-container">
                        <h4>📺 Quick Preview</h4>
                        <p><small>Low quality preview • No ImageMagick required</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.video(preview_bytes)
                    
                    st.download_button(
                        label="📥 Download Preview",
                        data=preview_bytes,
                        file_name=f"preview_{project.title}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                    
                    progress_bar.empty()
                    
                except Exception as e:
                    status_container.markdown(f"""
                    <div class="status-error">❌ Preview error: {str(e)}</div>
                    """, unsafe_allow_html=True)
        
        with col_prev2:
            if st.button("📊 Performance Test", use_container_width=True):
                total_items = len(project.media_items)
                total_duration = sum(item.duration for item in project.media_items)
                estimated_time = total_items * 15 + total_duration * 2
                
                st.info(f"""
                🔄 **Performance Estimate:**
                - Total clips: {total_items}
                - Total duration: {total_duration:.1f}s
                - Estimated generation time: ~{estimated_time:.0f} seconds
                - Processing: PIL + OpenCV (No ImageMagick)
                """)
        
        if st.button("🎬 Generate Final Video", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_container = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
            
            def update_status(status):
                status_container.markdown(f"""
                <div class="status-processing">{status}</div>
                """, unsafe_allow_html=True)
            
            try:
                update_status("🚀 Initializing video generation...")
                
                output_path = st.session_state.video_generator.generate_video(
                    project,
                    progress_callback=update_progress,
                    status_callback=update_status
                )
                
                status_container.markdown("""
                <div class="status-success">✅ Video generated successfully!</div>
                """, unsafe_allow_html=True)
                
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                
                st.download_button(
                    label="📥 Download Final Video",
                    data=video_bytes,
                    file_name=f"{project.title}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
                
                st.video(video_bytes)
                progress_bar.empty()
                
            except Exception as e:
                status_container.markdown(f"""
                <div class="status-error">❌ Error: {str(e)}</div>
                """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Upload media files and configure your project to start generating videos!")

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666;'>
    <p><strong>🎬 AI Video Generator Pro</strong> • No ImageMagick Required</p>
    <p><small>Using: PIL + OpenCV + NumPy for all image processing</small></p>
    <p><small>✅ Pure Python implementation • 🚀 Fast & Reliable</small></p>
</div>
""", unsafe_allow_html=True)


