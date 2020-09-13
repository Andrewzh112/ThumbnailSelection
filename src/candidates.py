from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from glob import glob
import heapq
import os
from skvideo import io
from PIL import Image
from tqdm import tqdm
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import (InceptionResNetV2,
                                                    preprocess_input)
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TopFrames:
    def __init__(self):
        self.video_paths = glob(os.path.join('data', '*.mp4'))

    def _find_scenes(self, video_path, threshold=30.0):
        """https://pyscenedetect.readthedocs.io/en/latest/

        Args:
            video_path ([type]): [description]
            threshold (float, optional): [description]. Defaults to 30.0.

        Returns:
            [type]: [description]
        """
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold))
        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        time_frames = scene_manager.get_scene_list(base_timecode)
        return [(frame[0].get_frames(), frame[1].get_frames())
                for frame in time_frames]

    def predict_scores(self, frames, model, start):
        # frames = tf.convert_to_tensor(frames, dtype=tf.float32)
        frames = preprocess_input(frames.astype(np.float32))
        scores = model.predict(frames, batch_size=1, verbose=1)
        means = self.mean_score(scores)
        return (np.max(means), np.argmax(means) + start)

    def mean_score(self, scores):
        si = np.arange(1, 11, 1)
        score = scores * si
        return score.mean(axis=1)

    def get_scores(self, video, frame_cuts):
        """[summary]

        Args:
            video ([type]): [description]
            frame_cuts ([type]): [description]

        Returns:
            [type]: [description]
        """
        base_model = InceptionResNetV2(input_shape=(
            None, None, 3), include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)
        model.load_weights('weights/inception_resnet_weights.h5')
        score_list = []
        for start, end in tqdm(frame_cuts, total=len(frame_cuts)):
            if end - start > 1000:
                frames = video[start: end // 2, :, :, :].copy()
                if frames.shape[0] == 0:
                    continue
                best_section_score = self.predict_scores(
                    frames, model, start)
                start = end // 2
            frames = video[start: end, :, :, :].copy()
            if frames.shape[0] == 0:
                continue
            best_score = self.predict_scores(
                frames, model, start)
            if end - start > 1000 and best_section_score[0] > best_score[0]:
                best_score = best_section_score
            score_list.append(best_score)
        heapq.heapify(score_list)
        return score_list

    def top_frames(self, video, topk, scores):
        """[summary]

        Args:
            video ([type]): [description]
            topk ([type]): [description]
            scores ([type]): [description]

        Returns:
            [type]: [description]
        """
        frames = []
        for _ in range(topk):
            try:
                _, index = heapq.heappop(scores)
                frame = video[index, :, :, :]
            except IndexError:
                for _ in range(topk - len(frames)):
                    frames.append(frame)
                break
            frames.append(frame)
        return frames

    def save_frames(self, frames, name):
        """[summary]

        Args:
            frames ([type]): [description]
            name ([type]): [description]
        """
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(f'data/{name}N{i}.jpg')

    def topk(self, topk=5):
        """[summary]

        Args:
            topk (int, optional): [description]. Defaults to 5.
        """
        for path in self.video_paths:
            name = os.path.split(path)[-1].split('.')[0]
            frame_cuts = self._find_scenes(path)
            video = io.vread(path)
            scores = self.get_scores(video, frame_cuts)
            frames = self.top_frames(video, topk, scores)
            self.save_frames(frames, name)


if __name__ == '__main__':
    TopFrames().topk()
