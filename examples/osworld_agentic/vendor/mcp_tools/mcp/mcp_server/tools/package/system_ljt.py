import uno
from functools import wraps
from decimal import Decimal, ROUND_HALF_UP
import subprocess
from PIL import Image
import sys
import os
import shutil

# import pandas as pd
from typing import List, Optional


class SystemTools:
    @staticmethod
    def remove_image_background(image_path, output_path):
        """
        Remove background from an image. 
        Automatically installs 'rembg' if not found.
        """
        try:
            # Try import, auto-install if missing
            # need install onnxruntime
            try:
                from rembg import remove
            except ImportError:
                print("Module 'rembg' not found. Installing now...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "rembg"])
                from rembg import remove  # Re-import after install

            try:
                import onnxruntime
            except ImportError:
                print("Module 'onnxruntime' not found. Installing now...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "onnxruntime"])
                import onnxruntime

            if not os.path.exists(image_path):
                return f"Error: image_path '{image_path}' does not exist."

            # Ensure output folder exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Perform background removal
            with open(image_path, 'rb') as inp:
                result = remove(inp.read())

            with open(output_path, 'wb') as out:
                out.write(result)

            return f"Background removed successfully → {output_path}"

        except Exception as e:
            return f"Error removing background: {e}"

    @staticmethod
    def convert_image_format(image_path, output_format, output_path):
        try:
            img = Image.open(image_path)
            if output_format.lower() == "jpg" and img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                background.save(output_path, output_format.upper())
            else:
                img.save(output_path, format=output_format.upper())
            return "Image converted"
        except Exception as e:
            return f"Error converting image: {e}"

    @staticmethod
    def git_operation(repo_path, operation, arguments=None):
        try:
            cmd = ["git", operation] + (arguments or [])
            subprocess.run(
                cmd, cwd=repo_path if repo_path else None, check=True)
            return f"Git {operation} executed"
        except Exception as e:
            return f"Error running git {operation}: {e}"

    @staticmethod
    def git_set_user_info(username, email, is_global=True, repo_path=None):
        """
        Set git username and email globally or for a specific repository.

        Args:
            username (str): Git username
            email (str): Git email
            is_global (bool): True for global config, False for repo-only
            repo_path (str, optional): Path to repository (only used if is_global=False)
        """
        try:
            # Check if git is installed
            if shutil.which("git") is None:
                return "Error: Git is not installed or not found in PATH."

            if is_global:
                subprocess.run(["git", "config", "--global",
                               "user.name", username], check=True)
                subprocess.run(["git", "config", "--global",
                               "user.email", email], check=True)
                return f"Git global user set: {username} <{email}>"
            else:
                target_path = repo_path if repo_path else "."
                subprocess.run(["git", "config", "user.name",
                               username], cwd=target_path, check=True)
                subprocess.run(["git", "config", "user.email",
                               email], cwd=target_path, check=True)
                return f"Git local user set for repo at {target_path}: {username} <{email}>"
        except subprocess.CalledProcessError as e:
            return f"Error setting git user info: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"

    @staticmethod
    def ffmpeg_video_to_gif(video_path, start_time, duration, output_path):
        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", video_path,
                "-vf", "fps=10,scale=480:-1:flags=lanczos",
                output_path
            ]
            subprocess.run(cmd, check=True)
            return "GIF created"
        except Exception as e:
            return f"Error creating GIF: {e}"
