from glob import glob
import os
from skvideo import io, measure, utils
from PIL import Image
import numpy as np
import heapq


class TopFrames:
    def __init__(self):
        self.video_paths = glob(os.path.join('data', '*.mp4'))

    def get_scores(self, video, cuts):
        scores = []
        start, end = 0, 1
        while end <= len(cuts):
            scene = video[cuts[start]: cuts[end], :, :, :]
            gray_scene = utils.rgb2gray(scene)
            start += 1
            end += 1
            niqe = measure.videobliinds_features(gray_scene)
            min_niqe = min(niqe)
            top_frame = np.argmin(niqe) + cuts[start]
            heapq.heappush(scores, (min_niqe, top_frame))
        return scores

    def top_frames(self, video, topk, scores):
        frames = []
        for _ in range(topk):
            _, index = heapq.heappop(scores)
            frame = video[index, :, :, :]
            frames.append(frame)
        return frames

    def save_frames(self, frames, name):
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(f'data/{name}N{i}.jpg')

    def scene_times(self, video, method, min_scene_length):
        cuts = measure.scenedet(
            video, method=method, min_scene_length=min_scene_length
        ).tolist()
        cuts.append(video.shape[0])
        return cuts

    def run(self, topk=5, method='histogram', min_scene_length=200):
        for path in self.video_paths:
            name = os.path.split(path)[-1].split('.')[0]
            video = io.vread(path)
            cuts = self.scene_times(video, method, min_scene_length)
            scores = self.get_scores(video, cuts)
            frames = self.top_frames(video, topk, scores)
            self.save_frames(frames, name)


if __name__ == '__main__':
    TopFrames().run()
