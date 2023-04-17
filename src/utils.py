from yt_dlp import YoutubeDL
from typing import Tuple


def get_youtube_video_info(youtube_video_id: str = "dQw4w9WgXcQ") -> Tuple[str, str]:
    video_link = "https://www.youtube.com/watch?v=" + youtube_video_id

    ydl = YoutubeDL({'outtmpl': '%(id)s.%(ext)s'})
    with ydl:
        info_dict = ydl.extract_info(video_link, download=False)
        video_title = info_dict['title']
        video_desc = info_dict['description']

    return video_title, video_desc
