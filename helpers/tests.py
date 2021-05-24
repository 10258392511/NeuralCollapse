import unittest
import torch
from oblique import *


class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        M, N = 5, 10
        features = torch.randn((N, M))
        labels = torch.randint(0, M, (N,))
        criterion = torch.nn.CrossEntropyLoss()
        # print(f"{features.shape}, {labels.shape}")
        loss_gt = criterion(features, labels)
        loss = softmax_loss(features.numpy().T, labels.numpy())
        # print(f"gt: {loss_gt}, loss: {loss}")
        # self.assertAlmostEqual(loss, loss_gt.item(), msg=f"gt: {loss_gt}, loss: {loss}")
        self.assertTrue(numpy.allclose(loss_gt, loss), msg=f"gt: {loss_gt}, loss: {loss}")


if __name__ == '__main__':
    unittest.main()
