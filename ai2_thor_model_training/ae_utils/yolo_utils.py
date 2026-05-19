class YoloUtils:
	def __init__(self):
		self.classes_for_yolo_finetune = ["door", "DoorWay", "window", "Sofa", "Desk", "DiningTable", "Fridge",
										  "Warderobe", "TVStand", "ShelvingUnit", "Bed", "Television",
										  "GarbageCan", "Painting"]
		self.classes_for_yolo_finetune = [lbl.upper() for lbl in self.classes_for_yolo_finetune]

	def extract_detections(self, event) -> list:
		"""
		Pull visible bounding boxes from an AI2-Thor event.

		Returns a list of dicts: {objectType, objectId, bbox, visible}
		"""
		detections = []
		img_h, img_w = event.frame.shape[:2]

		for obj in event.metadata["objects"]:
			if not obj["visible"]:
				continue
			obj_type = obj["objectType"]
			if obj_type.upper() not in self.classes_for_yolo_finetune:
			    continue

			# AI2-Thor provides 2D bounding boxes via instance segmentation map
			# We derive the tight bounding box from the instance segmentation mask
			instance_id = obj.get("objectId")
			if instance_id and event.instance_detections2D:
				bbox_raw = event.instance_detections2D.get(instance_id)
				if bbox_raw is not None:
					# bbox_raw = [x1, y1, x2, y2]  (pixel coords)
					x1, y1, x2, y2 = bbox_raw
					bbox = {
						"x": float(x1),
						"y": float(y1),
						"width": float(x2 - x1),
						"height": float(y2 - y1),
					}

					detections.append({
						"objectType": obj_type,
						"objectId": instance_id,
						"bbox": bbox,
						"visible": True,
					})

		annotations = self.prepare_yolo_annotation(detections, img_w=img_w, img_h=img_h)

		return annotations

	def get_class_id(self, class_name: str) -> int:
		"""Return YOLO class index for a given object type."""
		return self.classes_for_yolo_finetune.index(class_name.upper()) if class_name.upper() in self.classes_for_yolo_finetune else -1

	def bbox_to_yolo(self, bbox: dict, img_w: int, img_h: int):
		"""
		Convert AI2-Thor bounding box dict to YOLO format.

		AI2-Thor bbox keys: x, y, width, height  (pixel coords, top-left origin)
		YOLO format:        cx, cy, w, h          (normalised 0-1, centre-based)

		Returns (cx, cy, w, h) or None if the box is degenerate.
		"""
		x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
		if w <= 0 or h <= 0:
			return None

		cx = (x + w / 2) / img_w
		cy = (y + h / 2) / img_h
		nw = w / img_w
		nh = h / img_h

		# Clamp to [0, 1]
		cx = max(0.0, min(1.0, cx))
		cy = max(0.0, min(1.0, cy))
		nw = max(0.0, min(1.0, nw))
		nh = max(0.0, min(1.0, nh))

		return cx, cy, nw, nh

	def prepare_yolo_annotation(self, detections, img_w: int, img_h: int):
		lines = []
		for det in detections:
			cls_id = self.get_class_id(det["objectType"])

			if cls_id < 0:
				continue
			yolo_box = self.bbox_to_yolo(det["bbox"], img_w, img_h)
			#print("AE: ", cls_id, det["objectType"], det["bbox"], yolo_box)
			if yolo_box is None:
				continue
			cx, cy, w, h = yolo_box
			lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

		return lines