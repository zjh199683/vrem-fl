from models import SegmentationCNN
from datasets import ApolloscapeDataset
import torch
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_apollo_records(idx):

    road = 'road02down'
    records = [2, 3, 4, 5, 6, 18, 19, 20, 21]
    record = records[idx]
    return [{'road' : road, 'record' : record}]


def train(model, loss_fn, metric, train_set, optimizer):

    running_loss = 0.
    iou = 0.

    model.train(True)

    for i, data in tqdm(enumerate(train_set)):
        inputs, labels, _ = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)['out']
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iou += metric(outputs, labels)

    avg_loss = running_loss / (i + 1)
    avg_iou = iou / (i + 1)
    print('Train Loss {} Train mIoU {}'.format(avg_loss, avg_iou))


def validate(model, loss_fn, metric, validation_set):

    running_vloss = 0.
    viou = 0.
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_set):
            vinputs, vlabels, _ = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)['out']
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            viou += metric(voutputs, vlabels)

    avg_vloss = running_vloss / (i + 1)
    avg_viou = viou / (i + 1)
    print('Validation Loss {} Validation mIoU {}'.format(avg_vloss, avg_viou))


device = 'cuda'
lr = 2.5e-4
epochs = 20

model = SegmentationCNN(pretrained=True, num_classes=11).to(device)
metric = JaccardIndex(task='multiclass', ignore_index=255, num_classes=11).to(device)
loss = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean').to(device)
optimizer = torch.optim.SGD(([{'params': model.mobilenetv3_deeplabv3.backbone.parameters(), 'lr': lr},
                              {'params': model.mobilenetv3_deeplabv3.classifier.parameters(), 'lr': 10 * lr}]))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0,
                                                       last_epoch=-1, verbose=False)

path = '/home/giovanni/Desktop/fed-vehicles/fl_simulator_nn/data/apolloscape/'
apollo_train = ApolloscapeDataset(road_record_list=get_apollo_records(slice(0, 7)), base_dir=path)
apollo_test = ApolloscapeDataset(road_record_list=get_apollo_records(slice(7, 8)), base_dir=path)

train_set = DataLoader(dataset=apollo_train, batch_size=8, shuffle=True, drop_last=True)
validation_set = DataLoader(dataset=apollo_test, batch_size=8, shuffle=True, drop_last=False)

for i in range(epochs):
    print('---------------------')
    print('Starting epoch', i + 1)
    train(model, loss, metric, train_set, optimizer)
    validate(model, loss, metric, validation_set)
    scheduler.step()

