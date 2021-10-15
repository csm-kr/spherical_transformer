import torch
import torch.nn as nn
import time
import visdom

from thop.profile import profile

from datasets.mnist_cube_dataset import Mnist_Cube_Dataset
from datasets.mnist_erp_dataset import Mnist_ERP_Dataset
from torch.utils.data import DataLoader
from models.sphtr import SPHTransformer
from models.sphtr_erp import SPHTransformer_ERP


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def main_wokrer():

    vis = visdom.Visdom(port='8097')

    # R / R  - rotate True True

    # erp
    train_set = Mnist_ERP_Dataset(root='D:\data\MNIST', split='train', rotate=True, bandwidth=25)
    test_set = Mnist_ERP_Dataset(root='D:\data\MNIST', split='test', rotate=True, bandwidth=25)

    # cube map
    # train_set = Mnist_Cube_Dataset(root='D:\data\MNIST', split='train', rotate=True, num_edge=15)
    # test_set = Mnist_Cube_Dataset(root='D:\data\MNIST', split='test', rotate=True, num_edge=15)

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

    # model = SPHTransformer(model_dim=32, num_patches=6, num_head=8, num_layers=3, dropout=0.0, num_classes=10, input_dim=225)
    model = SPHTransformer_ERP(model_dim=32, num_patches=25, num_head=8, num_layers=3, dropout=0.0, num_classes=10, input_dim=225)
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
            if i % 10 == 0:
                vis.line(X=torch.ones((1, 1)) * i + epoch * len(train_loader),
                         Y=torch.Tensor([loss]).unsqueeze(0),
                         update='append',
                         win='loss',
                         opts=dict(x_label='step',
                                   y_label='loss',
                                   title='loss',
                                   legend=['total_loss']))

        print("")

        val_loss = 0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            model.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).long().sum().item()

        accuracy = correct / total
        val_avg_loss = val_loss / len(test_loader)

        print('Test Accuracy: {:.3f}'.format(100 * accuracy))

        if vis is not None:
            vis.line(X=torch.ones((1, 2)) * epoch,
                     Y=torch.Tensor([accuracy, val_avg_loss]).unsqueeze(0),
                     update='append',
                     win='test_loss_acc',
                     opts=dict(x_label='epoch',
                               y_label='test_loss and acc',
                               title='test_loss and accuracy',
                               legend=['accuracy', 'avg_loss']))


if __name__ == '__main__':
    main_wokrer()
