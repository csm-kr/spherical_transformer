import torch
import torch.nn as nn


def test(test_loader, criterion, device, model, opts):
    # state_dict = torch.load('./saves/best_acc.pth.tar')
    state_dict = torch.load('./saves/best_acc_0.8539.pth.tar')
    model.load_state_dict(state_dict)

    val_loss = 0
    correct = 0
    total = 0
    best_acc = 0

    for i, (images, labels) in enumerate(test_loader):
        model.eval()

        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).long().sum().item()

    accuracy = correct / total
    val_avg_loss = val_loss / len(test_loader)

    # print('Test Accuracy: {:.3f}'.format(100 * accuracy))
    print(accuracy)


if __name__ == '__main__':
    from models.sphtr import SPHTransformer
    from datasets.mnist_icosa_dataset import Mnist_Icosa_Dataset
    from torch.utils.data import DataLoader
    device = torch.device('cuda:0')
    model = SPHTransformer(model_dim=24, num_patches=20, num_head=8,
                           num_layers=7, dropout=0.0, num_classes=10, input_dim=64).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    test_set = Mnist_Icosa_Dataset(root='D:\data\MNIST', split='test', rotate=True,  bandwidth=25, division_level=3)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=128,
                             num_workers=4,
                             shuffle=False,
                             pin_memory=True)

    test(test_loader, criterion, device, model, None)


