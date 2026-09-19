"""Reproducible subject-specific BCI IV-2a experiments (Apache-2.0)."""
import argparse
import hashlib
import json
import os
from pathlib import Path

MODEL_NAMES = ('atcnet', 'mamba', 'mamba-channel')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def fingerprint(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def split_training(x, y, fraction=0.2, seed=42):
    """Fit per-channel/timepoint statistics on training trials only."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=fraction,
                                         random_state=seed, stratify=y)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    scale = x[train_idx].std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return train_idx, val_idx, mean, scale


def build_model(name):
    from models import ATCNet_
    attention = {'atcnet': 'mha', 'mamba': 'mamba', 'mamba-channel': 'mamba_channel'}[name]
    return ATCNet_(n_classes=4, in_chans=22, in_samples=1125, attention=attention)


def read_session(data_dir, subject, training):
    import numpy as np
    from preprocess import load_BCI2a_data
    x, y = load_BCI2a_data(str(Path(data_dir).resolve()) + os.sep, subject, training)
    return x[:, None, :, :].astype(np.float32), y


def select_run(runs):
    """Only validation loss participates in selection; ties retain the first run."""
    return min(runs, key=lambda run: run['val_loss'])


def train(args):
    import numpy as np
    import tensorflow as tf
    data_dir = Path(args.data_dir)
    required = [data_dir / f'A{s:02d}{session}.mat' for s in args.subjects for session in ('T', 'E')]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError('Missing BCI IV-2a MAT files: ' + ', '.join(missing))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = {key: value for key, value in vars(args).items()}
    config['tensorflow_version'] = tf.__version__
    config['keras_version'] = tf.keras.__version__
    config['protocol'] = 'subject-specific/train-only-standardization/validation-selection-v1'
    config['data_sha256'] = {p.name: fingerprint(p) for p in required}
    write_json(output / 'config.json', config)
    manifests = {}
    for subject in args.subjects:
        x, y = read_session(data_dir, subject, True)
        ti, vi, mean, scale = split_training(x, y, args.val_fraction, args.split_seed)
        folder = output / f'subject-{subject:02d}'
        folder.mkdir()
        np.savez(folder / 'preprocessing.npz', train_indices=ti, val_indices=vi, mean=mean, scale=scale)
        x_train, x_val = (x[ti] - mean) / scale, (x[vi] - mean) / scale
        runs = []
        for seed in args.seeds:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = build_model(args.model)
            model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            weights = folder / f'seed-{seed}.weights.h5'
            history = model.fit(x_train, y[ti], validation_data=(x_val, y[vi]),
                epochs=args.epochs, batch_size=args.batch_size, verbose=2,
                callbacks=[tf.keras.callbacks.ModelCheckpoint(str(weights), monitor='val_loss',
                            save_best_only=True, save_weights_only=True),
                           tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                            patience=20, min_lr=0.0001), tf.keras.callbacks.TerminateOnNaN()])
            losses = np.asarray(history.history['val_loss'])
            if not np.isfinite(losses).all() or not weights.is_file():
                raise ValueError('Non-finite validation loss or missing checkpoint; experiment stopped.')
            runs.append({'seed': seed, 'val_loss': float(losses.min()),
                         'weights': weights.relative_to(output).as_posix()})
            write_json(folder / f'seed-{seed}.history.json', history.history)
        manifests[str(subject)] = {'runs': runs, 'selected': select_run(runs)}
        write_json(output / 'selection.json', manifests)
    evaluate(output, data_dir)


def evaluate(output, data_dir):
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    output, data_dir = Path(output), Path(data_dir)
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    selection = json.loads((output / 'selection.json').read_text(encoding='utf-8'))
    if set(selection) != {str(s) for s in config['subjects']}:
        raise ValueError('Incomplete experiment: not all requested subjects have selected checkpoints.')
    report = {}
    for subject in config['subjects']:
        filename = f'A{subject:02d}E.mat'
        if fingerprint(data_dir / filename) != config['data_sha256'][filename]:
            raise ValueError(f'Dataset changed: {filename}')
        x, y = read_session(data_dir, subject, False)
        with np.load(output / f'subject-{subject:02d}' / 'preprocessing.npz') as stats:
            x = (x - stats['mean']) / stats['scale']
        tf.keras.backend.clear_session()
        model = build_model(config['model'])
        selected = selection[str(subject)]['selected']
        model.load_weights(output / selected['weights'])
        prediction = model.predict(x, batch_size=config['batch_size'], verbose=0).argmax(axis=-1)
        report[str(subject)] = {'selected_seed': selected['seed'],
            'accuracy': float(accuracy_score(y, prediction)),
            'kappa': float(cohen_kappa_score(y, prediction)),
            'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1, 2, 3]).tolist()}
    summary = {'subjects': report,
               'mean_accuracy': float(np.mean([r['accuracy'] for r in report.values()])),
               'mean_kappa': float(np.mean([r['kappa'] for r in report.values()]))}
    write_json(output / 'metrics.json', summary)
    print(json.dumps(summary, indent=2))
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--model', choices=MODEL_NAMES, default='atcnet')
    training.add_argument('--data-dir', required=True)
    training.add_argument('--output', required=True, help='New directory; existing paths are never overwritten')
    training.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 10)))
    training.add_argument('--seeds', nargs='+', type=int, default=[1])
    training.add_argument('--split-seed', type=int, default=42)
    training.add_argument('--val-fraction', type=float, default=0.2)
    training.add_argument('--epochs', type=int, default=500)
    training.add_argument('--patience', type=int, default=100)
    training.add_argument('--batch-size', type=int, default=64)
    training.add_argument('--lr', type=float, default=0.001)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--data-dir', required=True)
    evaluation.add_argument('--output', required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == 'train':
        if any(s not in range(1, 10) for s in args.subjects) or len(set(args.subjects)) != len(args.subjects):
            p.error('subjects must be unique integers from 1 to 9')
        if any(s < 0 or s >= 2**32 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
            p.error('seeds must be unique integers in [0, 2**32)')
        if not 0 < args.val_fraction < 1 or min(args.epochs, args.batch_size) < 1 or args.patience < 0 or not args.lr > 0:
            p.error('invalid validation fraction, epochs, batch size, patience, or learning rate')
    try:
        if args.command == 'train':
            train(args)
        else:
            evaluate(args.output, args.data_dir)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
