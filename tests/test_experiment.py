import numpy as np
import pytest
import tensorflow as tf
from experiment import build_model, select_run, split_training, main
from mamba import MambaBlock


def test_scaler_excludes_validation():
    x = np.arange(80, dtype=np.float32).reshape(20, 1, 2, 2)
    y = np.tile(np.arange(4), 5)
    ti, vi, mean, scale = split_training(x, y)
    changed = x.copy()
    changed[vi] = 1000000
    ti2, vi2, mean2, scale2 = split_training(changed, y)
    np.testing.assert_array_equal(ti, ti2)
    assert not set(ti) & set(vi)
    assert set(ti) | set(vi) == set(range(20))
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(scale, scale2)
    np.testing.assert_allclose(((x[ti] - mean) / scale).mean(axis=0), 0, atol=1e-6)


def test_selection_ignores_test_results():
    assert select_run([{'val_loss': .2, 'test_accuracy': .1},
                       {'val_loss': .3, 'test_accuracy': .9}])['test_accuracy'] == .1


@pytest.mark.parametrize('name', ['atcnet', 'mamba', 'mamba-channel'])
def test_model_gradient_and_weights(name, tmp_path):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(7)
    model = build_model(name)
    x = tf.random.normal((2, 1, 22, 1125))
    with tf.GradientTape() as tape:
        probabilities = model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.constant([0, 1]), probabilities))
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients)
    expected = model(x, training=False).numpy()
    np.testing.assert_allclose(expected.sum(axis=-1), 1, atol=1e-6)
    path = tmp_path / 'model.weights.h5'
    model.save_weights(path)
    restored = build_model(name)
    restored.load_weights(path)
    np.testing.assert_allclose(expected, restored(x, training=False), atol=1e-6)
    blocks = [layer for layer in model.layers if isinstance(layer, MambaBlock)]
    if name == 'mamba':
        assert len(blocks) == 5 and all(layer.d_model == 32 for layer in blocks)
    elif name == 'mamba-channel':
        assert len(blocks) == 5 and all(layer.d_model == 16 for layer in blocks)


def test_mamba_serialization(tmp_path):
    inputs = tf.keras.Input((6, 4))
    model = tf.keras.Model(inputs, MambaBlock(4)(inputs))
    x = tf.random.normal((2, 6, 4))
    expected = model(x, training=False).numpy()
    path = tmp_path / 'mamba.keras'
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(expected, restored(x, training=False), atol=1e-6)


def test_default_subjects_and_invalid_cli():
    from experiment import parser
    args = parser().parse_args(['train', '--data-dir', 'data', '--output', 'runs/test'])
    assert args.subjects == list(range(1, 10))
    with pytest.raises(SystemExit):
        main(['train', '--data-dir', 'data', '--output', 'runs/test', '--subjects', '0'])


def test_training_evaluation_roundtrip(tmp_path, monkeypatch):
    import experiment
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    for session in ('T', 'E'):
        (data_dir / f'A01{session}.mat').write_bytes(b'synthetic-test-only')
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 1, 22, 1125)).astype(np.float32)
    y = np.tile(np.arange(4), 5)
    output = tmp_path / 'experiment'
    def load(directory, subject, training):
        if not training:
            assert (output / 'selection.json').is_file()
        return x, y
    monkeypatch.setattr(experiment, 'read_session', load)
    main(['train', '--data-dir', str(data_dir), '--output', str(output),
          '--subjects', '1', '--epochs', '1', '--batch-size', '4'])
    import json
    first = json.loads((output / 'metrics.json').read_text())
    second = experiment.evaluate(output, data_dir)
    assert first == second
    with pytest.raises(SystemExit):
        main(['train', '--data-dir', str(data_dir), '--output', str(output), '--subjects', '1'])
    (data_dir / 'A01E.mat').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Dataset changed'):
        experiment.evaluate(output, data_dir)
