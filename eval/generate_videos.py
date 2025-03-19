import os
from moviepy.editor import VideoFileClip, AudioFileClip


path = "/ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/outputs_v2c/"
path_out = path + "videos/"

if not os.path.exists(path_out):
    os.makedirs(path_out)


with open("/ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/data/v2c_test.lst", "r") as fr:
    lines = fr.readlines()


for idx, line in enumerate(lines):
    wav_p, video_p, txt_p, wav, video, txt = line.strip().split("\t")
    video_clip = VideoFileClip(video)
    audio_clip = AudioFileClip(wav)
    audio_gen_clip = AudioFileClip(path + "gen/" + str(idx).zfill(8) + ".wav")
    print("video audio durations", video_clip.duration, audio_clip.duration, audio_gen_clip.duration)
    os.system("cp " + video + " " + path_out + str(idx).zfill(8) + ".mp4")
    video_clip_gt = video_clip.set_audio(audio_clip)
    video_clip_gen = video_clip.set_audio(audio_gen_clip)
    video_clip_gt.write_videofile(path_out + str(idx).zfill(8) + ".gt.mp4", codec="libx264", audio_codec="aac")
    video_clip_gen.write_videofile(path_out + str(idx).zfill(8) + ".gen.mp4", codec="libx264", audio_codec="aac")



