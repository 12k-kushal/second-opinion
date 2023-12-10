import os
import copy
import math
import time
import utils
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for _, _, inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


class resnet_pretrained(nn.Module):  # For copying grayscale channels to use in resnet50
    def __init__(self, model):
        super(resnet_pretrained, self).__init__()
        self.transform = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
        self.model = model

    def forward(self, x):
        x = self.transform(x)
        return self.model(x)


def main():
    use_cuda = True
    device = torch.device("cuda:5" if use_cuda and torch.cuda.is_available() else "cpu")
    batch_size = 36
    num_classes = 2
    num_epochs = 50
    modality = "MG"

    resnet18 = torchvision.models.resnet18(pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    resnet18.fc = nn.Linear(512, num_classes)
    model = resnet_pretrained(resnet18)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)

    mean = [0.5, 0.5, 0.5]
    std_dev = [0.5, 0.5, 0.5]
    num_workers = 2

    file_name = (
        "resnet_pretrained" + modality
    )  # name of the output files that will be created
    # if not os.path.exists('/projects/dso_mammovit/project_kushal/models'):
    #     os.mkdir('/projects/dso_mammovit/project_kushal/models')

    # if not os.path.exists('/projects/dso_mammovit/project_kushal/results'):
    #     os.mkdir('/projects/dso_mammovit/project_kushal/results')

    path_to_model = (
        "/../../models/" + file_name + ".tar"
    )  # name of the folder where to save the models
    path_to_results = (
        "/../../models/" + file_name + ".xlsx"
    )
    path_to_results_text = (
        "/../../models/" + file_name + ".txt"
    )

    # input file names
    csv_file_modality = "./../../data/labels.csv"  # name of the file which contains path to the images and other information of the images.
    df_modality = pd.read_csv(csv_file_modality)
    print("the original df modality shape:", df_modality.shape)
    df_train = df_modality[df_modality.subject_id.str.contains("Training")]
    df_test = df_modality[df_modality.subject_id.str.contains("Test")]
    df_train, df_val = train_test_split(
        df_train, test_size=0.1, shuffle=True, stratify=df_train["class"]
    )

    preprocess_train = utils.data_augmentation_train(mean, std_dev)

    preprocess_val = transforms.Compose(
        [
            transforms.Resize((1600, 1600)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std_dev),
        ]
    )

    dataset_gen_train = utils.BreastCancerDataset_generator(
        df_train, modality, preprocess_train
    )
    dataloader_train = DataLoader(
        dataset_gen_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=utils.MyCollate,
    )

    dataset_gen_val = utils.BreastCancerDataset_generator(
        df_val, modality, preprocess_val
    )
    dataloader_val = DataLoader(
        dataset_gen_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=utils.MyCollate,
    )

    dataset_gen_test = utils.BreastCancerDataset_generator(
        df_test, modality, preprocess_val
    )
    dataloader_test = DataLoader(
        dataset_gen_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=utils.MyCollate,
    )

    total_instances = df_modality.shape[0]
    train_instances = df_train.shape[0]
    val_instances = df_val.shape[0]
    test_instances = df_test.shape[0]
    print("training instances:", train_instances)
    print("Validation instances:", val_instances)
    print("Test instances:", test_instances)
    batches_train = int(math.ceil(train_instances / batch_size))
    batches_val = int(math.ceil(val_instances / batch_size))
    batches_test = int(math.ceil(test_instances / batch_size))

    dataloaders = {"train": dataloader_train, "val": dataloader_val}

    model, hist = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        is_inception=False,
    )


if __name__ == "__main__":
    main()
