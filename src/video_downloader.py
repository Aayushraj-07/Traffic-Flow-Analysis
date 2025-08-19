import os
import yt_dlp

def download_video(url: str, output_path="data", filename="traffic_video.mp4"):
    os.makedirs(output_path, exist_ok=True)
    ydl_opts = {
        "outtmpl": os.path.join(output_path, filename),
        "format": "mp4/bestvideo+bestaudio",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Video downloaded: {os.path.join(output_path, filename)}")
    return os.path.join(output_path, filename)

if __name__ == "__main__":
    download_video("https://www.youtube.com/watch?v=MNn9qKG2UFI")
