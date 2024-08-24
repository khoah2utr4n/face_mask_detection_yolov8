from ultralytics import YOLO

def load_model(pretrained_weights='yolov8n.pt'):
    return YOLO(pretrained_weights) 


def train_model(model, num_epochs, resume=False):
    results = model.train(data="data.yaml", epochs=num_epochs, imgsz=640, resume=resume)
    return results


def get_prediction(model, image_path):
    """Return a prediction of parsing model for parsing image.
        Args:
            model (torch.nn.Module): Model used for prediction.
            image_path (Path): Path to the image to be predicted. 
        Return:
            prediction (list): Prediction for parsing image.
    """
    results = model.predict(source=image_path, conf=0.7, save=False, verbose=False)
    prediction = []
    for i in range(len(results[0].boxes.xywhn)):
        pred = list(results[0].boxes.xywhn[i].cpu().numpy())
        class_index = results[0].boxes.cls[i].cpu().item()
        pred.append(class_index)
        conf = round(results[0].boxes.conf[i].cpu().item(), 2)
        pred.append(conf)
        prediction.append(pred)
    return prediction