import argparse
from model import train_model, load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model yolov8 for Face Mask Detection.')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs for training')
    parser.add_argument('--resume', default=False, type=bool,
                        help='continue training from the last epoch or not')
    parser.add_argument('--weights_path', default='yolov8n.pt', type=str,
                        help="Path to the model's weights (default: 'yolov8n.pt')")

    args = parser.parse_args()
    model = load_model(args.weights_path)
    training_result = train_model(model, args.epochs, resume=args.resume)