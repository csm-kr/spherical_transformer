import torch
import torch.nn as nn
import time
import visdom

from thop.profile import profile

# MNIST
from datasets.mnist_cube_dataset import Mnist_Cube_Dataset
from datasets.mnist_erp_dataset import Mnist_ERP_Dataset
from datasets.mnist_icosa_dataset import Mnist_Icosa_Dataset

# CIFAR
from datasets.cifar_erp_dataset import Cifar_ERP_Dataset
from datasets.cifar_cube_dataset import Cifar_Cube_Dataset
from datasets.cifar_icosa_dataset import Cifar_Icosa_Dataset

# ImageNet
from datasets.imagenet_erp_dataset import ImageNet_ERP_Dataset
from datasets.imagenet_cube_dataset import ImageNet_Cube_Dataset
from datasets.imagenet_cube_dataset import ImageNet_Cube_Dataset

# Panoramic
from datasets.panoramic_erp_dataset import Panoramic_ERP_Dataset
from datasets.panoramic_cube_dataset import Panoramic_Cube_Dataset


from torch.utils.data import DataLoader
from models.sphtr import SPHTransformer
from models.sphtr_erp import SPHTransformer_ERP
from models.cnn import ConvNet


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def main_wokrer():

    vis = visdom.Visdom(port='8097')

    # R / R  - rotate True True

    ############################## MNIST ##############################
    # ---------- erp ----------
    # train_set = Mnist_ERP_Dataset(root='D:\data\MNIST', split='train', rotate=True, bandwidth=25)
    # test_set = Mnist_ERP_Dataset(root='D:\data\MNIST', split='test', rotate=True, bandwidth=25)

    # ---------- cube ----------
    # train_set = Mnist_Cube_Dataset(root='D:\data\MNIST', split='train', rotate=True,  bandwidth=25, num_edge=15)
    # test_set = Mnist_Cube_Dataset(root='D:\data\MNIST', split='test', rotate=True,  bandwidth=25, num_edge=15)

    # ---------- icosahedron ----------
    # train_set = Mnist_Icosa_Dataset(root='D:\data\MNIST', split='train', rotate=True,  bandwidth=25, division_level=3)
    # test_set = Mnist_Icosa_Dataset(root='D:\data\MNIST', split='test', rotate=True,  bandwidth=25, division_level=3)

    ############################## CIFAR ##############################
    # ---------- erp ----------
    # train_set = Cifar_ERP_Dataset(root='D:\data\CIFAR10', split='train', rotate=True, bandwidth=50)
    # test_set = Cifar_ERP_Dataset(root='D:\data\CIFAR10', split='test', rotate=True, bandwidth=50)

    # ---------- cube ----------
    # train_set = Cifar_Cube_Dataset(root='D:\data\CIFAR10', split='train', rotate=True, bandwidth=50, num_edge=29)
    # test_set = Cifar_Cube_Dataset(root='D:\data\CIFAR10', split='test', rotate=True, bandwidth=50, num_edge=29)

    # ---------- icosa ----------
    train_set = Cifar_Icosa_Dataset(root='D:\data\CIFAR10', split='train', bandwidth=50, division_level=4)
    test_set = Cifar_Icosa_Dataset(root='D:\data\CIFAR10', split='test', bandwidth=50, division_level=4)

    ############################## ImageNet ##############################
    # ---------- erp ----------
    # train_set = ImageNet_ERP_Dataset(root='D:\data\ILSVRC_classification', split='train', is_minival=False,
    #                                  rotate=True, bandwidth=50)
    # test_set = ImageNet_ERP_Dataset(root='D:\data\ILSVRC_classification', split='train', is_minival=False,
    #                                 rotate=True, bandwidth=50)

    # ---------- cube ----------
    # train_set = ImageNet_Cube_Dataset(root='D:\data\ILSVRC_classification', split='train', is_minival=False,
    #                                   rotate=True, bandwidth=100, num_edge=58)
    # test_set = ImageNet_Cube_Dataset(root='D:\data\ILSVRC_classification', split='train', is_minival=True,
    #                                  rotate=True, bandwidth=100, num_edge=58)

    ############################### Panoramic dataset ##############################
    # ---------- erp ----------
    # train_set = Panoramic_ERP_Dataset(root='D:\data\panorama_360', split='train', rotate=True, bandwidth=100)
    # test_set = Panoramic_ERP_Dataset(root='D:\data\panorama_360', split='test', rotate=True, bandwidth=100)

    # ---------- cube ----------
    # train_set = Panoramic_Cube_Dataset(root='D:\data\panorama_360', split='train', rotate=True, bandwidth=100, num_edge=58)
    # test_set = Panoramic_Cube_Dataset(root='D:\data\panorama_360', split='test', rotate=True, bandwidth=100, num_edge=58)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=BATCH_SIZE,
                              num_workers=8,
                              shuffle=True,
                              pin_memory=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             shuffle=False,
                             pin_memory=True)

    # ---------- convolution for erp ----------
    # model = ConvNet()

    # ---------- transformer for erp ----------
    # model = SPHTransformer_ERP(model_dim=24, num_patches=25, num_head=8,
    #                            num_layers=6, dropout=0.0, num_classes=10, input_dim=50)
    # model = SPHTransformer_ERP(model_dim=64, num_patches=100, num_head=8,
    #                            num_layers=6, dropout=0.0, num_classes=10, input_dim=50 * 3)
    # model = SPHTransformer_ERP(model_dim=128, num_patches=100, num_head=8,
    #                            num_layers=6, dropout=0.0, num_classes=10, input_dim=50 * 3)

    # ---------- transformer for cube ----------

    # model = SPHTransformer(model_dim=24, num_patches=6, num_head=8,
    #                        num_layers=6, dropout=0.0, num_classes=10, input_dim=225)
    # model = SPHTransformer(model_dim=64, num_patches=6, num_head=8,
    #                        num_layers=6, dropout=0.0, num_classes=10, input_dim=29 * 29 * 3)
    # model = SPHTransformer(model_dim=128, num_patches=6, num_head=8,
    #                        num_layers=6, dropout=0.0, num_classes=1000, input_dim=58 * 58 * 3)
    # model = SPHTransformer(model_dim=128, num_patches=6, num_head=8,
    #                        num_layers=6, dropout=0.0, num_classes=10, input_dim=58 * 58 * 3)

    # ---------- transformer for icosahedron ----------
    # model = SPHTransformer(model_dim=24, num_patches=20, num_head=8,
    #                        num_layers=6, dropout=0.0, num_classes=10, input_dim=64)

    # model = SPHTransformer(model_dim=64, num_patches=20, num_head=8,
    #                        num_layers=6, dropout=0.0, num_classes=10, input_dim=256 * 3)

    model = SPHTransformer(model_dim=64, num_patches=320, num_head=8,
                           num_layers=6, dropout=0.1, num_classes=10, input_dim=16 * 3)

    # model = SPHTransformer(model_dim=128, num_patches=20, num_head=8,
    #                        num_layers=6, dropout=0.0, num_classes=10, input_dim=1024 * 3)

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
            print('\rEpoch [{0}/{1}], Iter [{2}/{3}], Loss: {4:.4f}, LR: {5:.5f}, Time: {6:.2f}'.format(epoch,
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
        print(accuracy)

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
