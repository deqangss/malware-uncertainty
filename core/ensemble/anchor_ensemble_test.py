from absl.testing import absltest
from absl.testing import parameterized
import tempfile

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer

from core.ensemble.anchor_ensemble import AnchorEnsemble
from core.ensemble.dataset_lib import build_dataset_from_numerical_data

architectures = ['dnn', 'text_cnn', 'multimodalitynn', 'r2d2', 'droidectc']

class MyTestCaseAnchorEnsemble(parameterized.TestCase):
    def setUp(self):
        self.x_dict, self.y_dict = dict(), dict()
        self.x_np, self.y_np = load_breast_cancer(return_X_y=True)
        self.x_dict['dnn'] = self.x_np
        self.y_dict['dnn'] = self.y_np
        x = np.random.randint(0, 256, (10, 10))
        y = np.random.choice(2, 10)
        self.x_dict['text_cnn'] = x
        self.y_dict['text_cnn'] = y
        x = [self.x_np] * 5
        self.x_dict['multimodalitynn'] = x
        self.y_dict['multimodalitynn'] = self.y_np
        x = np.random.uniform(0., 1., size=(10, 299, 299, 3))
        y = np.random.choice(2, 10)
        self.x_dict['r2d2'] = x
        self.y_dict['r2d2'] = y
        x = np.random.randint(0, 10000, size=(10, 1000))
        y = np.random.choice(2, 10)
        self.x_dict['droidectc'] = x
        self.y_dict['droidectc'] = y

    @parameterized.named_parameters([(arc_type, arc_type) for arc_type in architectures])
    def test_anchor_ensemble(self, arc_type):
        with tempfile.TemporaryDirectory() as output_dir:
            x = self.x_dict[arc_type]
            y = self.y_dict[arc_type]
            if arc_type is not 'multimodalitynn':
                train_dataset = build_dataset_from_numerical_data((x, y))
                val_dataset = build_dataset_from_numerical_data((x, y))
                n_samples = x.shape[0]
                input_dim = x.shape[1:]
            else:
                train_data = build_dataset_from_numerical_data(tuple(x))
                train_y = build_dataset_from_numerical_data(self.y_np)
                train_dataset = tf.data.Dataset.zip((train_data, train_y))
                val_data = build_dataset_from_numerical_data(tuple(x))
                val_y = build_dataset_from_numerical_data(self.y_np)
                val_dataset = tf.data.Dataset.zip((val_data, val_y))
                n_samples = x[0].shape[0]
                input_dim = [x[i].shape[1] for i in range(len(x))]

            anchor_ensemble = AnchorEnsemble(architecture_type=arc_type,
                                             model_directory=output_dir)
            anchor_ensemble.fit(train_dataset, val_dataset, input_dim=input_dim)

            res = anchor_ensemble.predict(x)
            self.assertEqual(anchor_ensemble.get_n_members(), anchor_ensemble.n_members)
            self.assertTrue(res.shape == (n_samples, anchor_ensemble.n_members, 1))

            anchor_ensemble.evaluate(x, y)


if __name__ == '__main__':
    absltest.main()
