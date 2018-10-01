import train
import json
import argparse


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='Where to save outputs.')
    args, _ = parser.parse_known_args()

    for model in ['RNN', 'LSTM']:
        # for t in range(4, 50, 5):
        for t in [4, 9, 14, 19, 24, 29, 34, 39 ,44, 49, 74, 99]:
            print('\n\n Training {} on palindromes of length {}...'.format(model, t+1))
            cfg = Namespace(
                model_type=model,
                input_length=t,
                input_dim=1,
                num_classes=10,
                num_hidden=128,
                batch_size=128,
                learning_rate=0.001,
                train_steps=5000,
                max_norm=10.0,
                model_path=args.model_path
            )
            train.train(cfg)
