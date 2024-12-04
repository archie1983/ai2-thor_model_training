import PIL.Image
import supervision as sv

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms


class YOLOWorldExtractor:
    def __init__(self):
        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

        self.class_names = ("person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, "
                       "traffic light, fire hydrant, stop sign, parking meter, bench, bird, "
                       "cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, "
                       "backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, "
                       "sports ball, kite, baseball bat, baseball glove, skateboard, "
                       "surfboard, tennis racket, bottle, wine glass, cup, fork, knife, "
                       "spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, "
                       "hot dog, pizza, donut, cake, chair, couch, potted plant, bed, "
                       "dining table, toilet, tv, laptop, mouse, remote, keyboard, "
                       "cell phone, microwave, oven, toaster, sink, refrigerator, book, "
                       "clock, vase, scissors, teddy bear, hair drier, toothbrush, plate, cloud, portrait, picture with clouds, tv table, glass table")

        # load config
        self.cfg = Config.fromfile(
            "./yolo_world_configs_and_weights/yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        )
        self.cfg.work_dir = "."
        self.cfg.load_from = "./yolo_world_configs_and_weights/yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth"
        self.runner = Runner.from_cfg(self.cfg)
        self.runner.call_hook("before_run")
        self.runner.load_or_resume()
        self.pipeline = self.cfg.test_dataloader.dataset.pipeline
        self.runner.pipeline = Compose(self.pipeline)
        self.runner.model.eval()

    def run_image(self,
            runner,
            input_image,
            max_num_boxes=100,
            score_thr=0.05,
            nms_thr=0.5,
            output_image="output.png",
    ):
        texts = [[t.strip()] for t in self.class_names.split(",")] + [[" "]]
        data_info = runner.pipeline(dict(img_id=0, img_path=input_image,
                                         texts=texts))

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = runner.model.test_step(data_batch)[0]
            runner.model.class_names = texts
            pred_instances = output.pred_instances

        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]
        output.pred_instances = pred_instances

        pred_instances = pred_instances.cpu().numpy()
        detections = sv.Detections(
            xyxy=pred_instances['bboxes'],
            class_id=pred_instances['labels'],
            confidence=pred_instances['scores']
        )

        txt_labels = [(texts[class_id][0], confidence) for class_id, confidence in zip(detections.class_id, detections.confidence)]

        labels = [
            f"{class_id} {confidence:0.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        image = PIL.Image.open(input_image)
        svimage = np.array(image)
        svimage = self.bounding_box_annotator.annotate(svimage, detections)
        svimage = self.label_annotator.annotate(svimage, detections, labels)
        return (svimage[:, :, ::-1], labels, detections, txt_labels)

    def process_image(self, img_url):
        print("Analysing: ", img_url)
        (img, labels, detections, txt_labels) = self.run_image(self.runner,img_url)
        #print("labels: ", labels)
        #print("detections: ", detections)
        #print("txt_labels", txt_labels)
        just_labels = [label_text for (label_text, confidence) in txt_labels]
        return just_labels
        #with sv.ImageSink(target_dir_path='/mount/test') as sink:
        #    sink.save_image(img, "proc_"+img_url)

#img = run_image(runner,"1.png")
#file = open('dog_processed.jpeg', 'w')
#file.write(img)
#file.close()
#sv.plot_image(img)

#with sv.ImageSink(target_dir_path='/mount/test') as sink:
#    sink.save_image(img)

if __name__ == "__main__":
    ywe = YOLOWorldExtractor()
    labels = ywe.process_image("scene_pics/train_28/115.png")
    print(labels)
