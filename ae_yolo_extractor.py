from enum import Enum
from ultralytics import YOLO
from ae_yolo_world_extractor import YOLOWorldExtractor

class YOLOType(Enum):
    NANO = 1 # nano
    MEDIUM = 2 # medium
    LARGE = 3 # large
    XLARGE = 4 # extra large
    WORLD = 5

    #@classmethod
    def yolo_flavour(self):
        if self == YOLOType.NANO:
            return "yolo11n"
        if self == YOLOType.MEDIUM:
            return "yolo11m"
        if self == YOLOType.LARGE:
            return "yolo11l"
        if self == YOLOType.XLARGE:
            return "yolo11x"

class YOLOExtractor:
    def __init__(self, yolo_type):
        # Load a model
        if yolo_type == YOLOType.WORLD:
            self.model = YOLOWorldExtractor()
        else:
            self.model = YOLO(yolo_type.yolo_flavour()) # pretrained YOLO11 model

        self.yolo_type = yolo_type

    ##
    # Analyze an image with YOLO and return a list of items that have been detected
    # in the image.
    ##
    def what_is_in_the_picture(self, image_url):
        if self.yolo_type == YOLOType.WORLD:
            return self.what_is_in_the_picture_yolo_world(image_url)
        else:
            return self.what_is_in_the_picture_yolo11(image_url)

    ##
    # Uses YOLO-World to tell what's in the picture
    ##
    def what_is_in_the_picture_yolo_world(self, image_url):
        image_scan_results = self.model.process_image(image_url)
        ret_set = set(image_scan_results)

        return ret_set

    ##
    # Uses YOLO11 to tell what's in the picture
    ##
    def what_is_in_the_picture_yolo11(self, image_url):
        image_scan_results = self.model(image_url)
        ret_set = set()

        for result in image_scan_results:
            for box in result.boxes:
                #print(int(box.cls.item()), box.cls, box.id)
                category_index = int(box.cls.item())
                res_text = result.names[category_index]
                #print(res_text)
                ret_set.add(res_text)

        return ret_set

if __name__ == "__main__":
    ye = YOLOExtractor(YOLOType.WORLD)
    res = ye.what_is_in_the_picture("scene_pics/train_1/1.png")
    print(res)
