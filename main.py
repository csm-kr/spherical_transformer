import torch
import torch.nn as nn
import time
from thop.profile import profile

from datasets.mnist_cube_dataset import Mnist_Cube_Dataset
from torch.utils.data import DataLoader
from models.sphtr import SPHTransformer


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def main_wokrer():

    # R / R  - rotate True True
    train_set = Mnist_Cube_Dataset(root='D:\data\MNIST', split='train', rotate=True, num_edge=15)
    test_set = Mnist_Cube_Dataset(root='D:\data\MNIST', split='test', rotate=True, num_edge=15)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=BATCH_SIZE,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             shuffle=False,
                             pin_memory=True)

    model = SPHTransformer(model_dim=32, num_patches=6, num_head=8, num_layers=3, dropout=0.0, num_classes=10, input_dim=225)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        # 8. train
        tic = time.time()
        model.train()
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # get lr
            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            # time
            toc = time.time()

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}], Loss: {4:.4f}, LR: {5:.5f}, Time: {6:.2f}'.format(epoch + 1,
                                                                                                        NUM_EPOCHS, i,
                                                                                                        len(train_set) // BATCH_SIZE,
                                                                                                        loss.item(),
                                                                                                        lr,
                                                                                                        toc - tic),
                  end="")

        print("")

        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            model.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).long().sum().item()

        print('Test Accuracy: {0}'.format(100 * correct / total))


if __name__ == '__main__':
    main_wokrer()
