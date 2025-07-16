import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import random


class Distance_MLP(torch.nn.Module):
    def __init__(self, d_input, d_hid, d_out, dropout):
        super(Distance_MLP, self).__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(d_input * 2, d_hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_hid, d_out)
        )
        self.linear = torch.nn.Linear(d_out, 1)

    def forward(self, x1, x2):
        # x1 = self.nn(x1)
        # x2 = self.nn(x2)
        # dot product
        # x = torch.bmm(x1.unsqueeze(1), x2.unsqueeze(2))
        # x = x.view(-1, 1)
        x = torch.cat([x1, x2], dim=1)
        x = self.nn(x)
        x = self.linear(x)
        return x


class DistanceDataset(Dataset):
    def __init__(self, polys_coded, distance_dataset,device=torch.device('cuda')):
        self.polys_coded = polys_coded
        self.distance_dataset = list(distance_dataset.items())
        self.device = device

    def __len__(self):
        return len(self.distance_dataset)

    def __getitem__(self, idx):
        (id1, id2), actual_distance = self.distance_dataset[idx]
        poly1_embedding = self.polys_coded[id1].detach().clone().to(self.device).view(-1)
        poly2_embedding = self.polys_coded[id2].detach().clone().to(self.device).view(-1)
        actual_distance = torch.tensor(actual_distance, dtype=torch.float).to(self.device).view(-1)
        return poly1_embedding, poly2_embedding, actual_distance


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        # self.fc22 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        # x = torch.relu(self.fc22(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


class MLP_relationship(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_relationship, self).__init__()
        self.fc1 = torch.nn.Linear(input_size * 2, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        # self.fc22 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        # x = torch.relu(self.fc22(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_regression_model(model, train_data, test_data, validation_data, epochs=100, batch_size=32,
                           learning_rate=0.001, device='cuda', print_result=False, mse=True):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    if mse:
        criterion = torch.nn.MSELoss()
        training_criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.L1Loss()
        training_criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float().to(device))
            loss = training_criterion(outputs, labels.float().view(-1, 1).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        mse_test = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs.float().to(device))
                mse_test += criterion(outputs, labels.float().view(-1, 1).to(device)).item()
        mse_test /= len(test_loader)

        mse_val = 0.0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = model(inputs.float().to(device))
                mse_val += criterion(outputs, labels.float().view(-1, 1).to(device)).item()
        mse_val /= len(validation_loader)

        if epoch == 0:
            initiation_mse = running_loss / len(train_loader)
            initiation_mse_test = mse_test
            initiation_mse_val = mse_val
        if print_result and epoch % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, MSE: {mse_test:.4f}  {mse_val:.4f}')
    return initiation_mse, initiation_mse_test, initiation_mse_val, running_loss / len(train_loader), mse_test, mse_val


def train_classification_model(model, train_data, test_data, validation_data, epochs=100, batch_size=32,
                               learning_rate=0.001, device='cuda', print_result=False):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(epochs):
        false_predictions = []
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x1, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(x1.float().to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
        accuracy_train = 100 * correct / total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x1, labels in test_loader:
                outputs = model(x1.float().to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
                false_predictions.append((x1, labels, predicted))
        accuracy_test = 100 * correct / total

        correct = 0
        total = 0
        with torch.no_grad():
            for x1, labels in validation_loader:
                outputs = model(x1.float().to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
        accuracy_val = 100 * correct / total

        if epoch == 0:
            initiation_accuracy_train = accuracy_train
            initiation_accuracy_test = accuracy_test
            initiation_accuracy_val = accuracy_val

        if print_result and epoch % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy:{accuracy_train:.2f}%   {accuracy_test:.2f}%  {accuracy_val:.2f}%')

    return initiation_accuracy_train, initiation_accuracy_test, initiation_accuracy_val, accuracy_train, accuracy_test, accuracy_val, false_predictions


def test_representation_embed(model_embedding, z_size, labels, device, poly_ids_train, poly_ids_test, ply_ids_val,
                              batch_size=32, epochs=10, print_result=False, classification=True, mse=False):
    if classification:
        train_data = []
        test_data = []
        validation_data = []
        labels_onehot = {}
        labels_set = set([label for id, label in labels.items()])
        print(labels_set)

        for i, label in enumerate(labels_set):
            labels_onehot[label] = i

        for i in poly_ids_train:
            x = model_embedding[i]
            y = labels[i]
            y = labels_onehot[y[0]]
            train_data.append((x, y))
        for i in poly_ids_test:
            x = model_embedding[i]
            y = labels[i]
            y = labels_onehot[y[0]]
            test_data.append((x, y))
        for i in ply_ids_val:
            x = model_embedding[i]
            y = labels[i]
            y = labels_onehot[y[0]]
            validation_data.append((x, y))

        mlp = MLP(input_size=z_size, hidden_size=256, output_size=len(labels_onehot)).to(device)
        result_class = train_classification_model(mlp, train_data, test_data, validation_data=validation_data,
                                                  epochs=epochs, batch_size=batch_size, learning_rate=0.001,
                                                  device=device, print_result=print_result)
        initiation_accuracy_train, initiation_accuracy_test, initiation_accuracy_val, accuracy_train, accuracy_test, accuracy_val, false_predictions = result_class
        print(
            f'initiation_accuracy_train: {initiation_accuracy_train:.2f}%, accuracy_test: {initiation_accuracy_test:.2f}%, accuracy_val: {initiation_accuracy_val:.2f}%')
        print(
            f'afterwards_accuracy_train: {accuracy_train:.2f}%, accuracy_test: {accuracy_test:.2f}%, accuracy_val: {accuracy_val:.2f}%')
        return result_class
    else:
        train_data = []
        test_data = []
        validation_data = []

        for i in poly_ids_train:
            x = model_embedding[i]
            y = labels[i]
            train_data.append((x, y))
        for i in poly_ids_test:
            x = model_embedding[i]
            y = labels[i]
            test_data.append((x, y))
        for i in ply_ids_val:
            x = model_embedding[i]
            y = labels[i]
            validation_data.append((x, y))

        mlp = MLP(input_size=z_size, hidden_size=256, output_size=1).to(device)
        result_reg = train_regression_model(mlp, train_data, test_data, validation_data=validation_data, epochs=epochs,
                                            batch_size=batch_size, learning_rate=0.001, device=device,
                                            print_result=print_result, mse=mse)
        initiation_mse, initiation_mse_test, initiation_mse_val, running_loss, mse_test, mse_val = result_reg

        print(
            f'initiation_running_mse: {initiation_mse:.4f}, mse_test: {initiation_mse_test:.4f}, mse_val: {initiation_mse_val:.4f}')
        print(f'afterwards_running_mse: {running_loss:.4f}, mse_test: {mse_test:.4f}, mse_val: {mse_val:.4f}')

        return result_reg

def test_distance(polys_dict,polys_location_embedding,num_training = 5,num_epochs = 30,num_pairs = 50000,type1 = 'Polygon',type2 = 'Polygon',device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    polys_location_embedding = torch.tensor(polys_location_embedding)
    distance_dataset = {}
    #type1 2  in 'Polygon' # 'Point' # 'LineString'
    type1_polys_dict = dict([(key,value) for key,value in polys_dict.items() if value.geom_type == type1])
    type2_polys_dict = dict([(key,value) for key,value in polys_dict.items() if value.geom_type == type2])
    if len(type1_polys_dict)==0 or len(type2_polys_dict)==0:
        print('No corresponding datatype in the current dataset, try others: [\'MultiPolygon\',\'Polygon\',\'Point\' , \'LineString\']')
        return
    i=0
    while len(distance_dataset) < num_pairs:
        id1 = random.choice(list(type1_polys_dict.keys()))
        id2 = random.choice(list(type2_polys_dict.keys()))
        poly1 = type1_polys_dict[id1]
        poly2 = type2_polys_dict[id2]
        if id1 == id2 or poly1.geom_type != type1 or poly2.geom_type != type2:
            continue
        actual_distance = poly1.distance(poly2)
        distance_dataset[(id1, id2)] = actual_distance
        i+=1

    pairs_list = list(distance_dataset.keys())
    random.shuffle(pairs_list)
    training_ids = pairs_list[:round(0.6 * len(pairs_list))]
    testing_ids = pairs_list[round(0.6 * len(pairs_list)):round(0.8 * len(pairs_list))]
    validation_ids = pairs_list[round(0.8 * len(pairs_list)):]
    trainging_dataset = DistanceDataset(polys_location_embedding,
                                        {pair: distance_dataset[pair] for pair in training_ids})
    validation_dataset = DistanceDataset(polys_location_embedding,
                                         {pair: distance_dataset[pair] for pair in validation_ids})
    testing_dataset = DistanceDataset(polys_location_embedding, {pair: distance_dataset[pair] for pair in testing_ids})
    train_loader = DataLoader(trainging_dataset, batch_size=512, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=512, shuffle=False)
    testing_loader = DataLoader(testing_dataset, batch_size=512, shuffle=False)

    poly_feature_dim = polys_location_embedding.shape[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_list = []
    for i in range(num_training):
        model = Distance_MLP(poly_feature_dim, 256, 512, 0.0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.L1Loss()  # nn.MSELoss()  # Using MSE loss for distance prediction
        eval_criterion = torch.nn.L1Loss()  # Using L1 loss for evaluation

        loss_variation = []
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            outputs = []
            for poly1_embedding, poly2_embedding, actual_distance in train_loader:
                # print(poly1_embedding.shape, poly2_embedding.shape, actual_distance.shape)
                optimizer.zero_grad()
                poly1_embedding = poly1_embedding.view(-1, poly_feature_dim).to(device)
                poly2_embedding = poly2_embedding.view(-1, poly_feature_dim).to(device)
                actual_distance = actual_distance.view(-1, 1).to(device)
                output = model(poly1_embedding, poly2_embedding)
                loss = criterion(output, actual_distance)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for poly1_embedding, poly2_embedding, actual_distance in validation_loader:
                    poly1_embedding = poly1_embedding.view(-1, poly_feature_dim).to(device)
                    poly2_embedding = poly2_embedding.view(-1, poly_feature_dim).to(device)
                    actual_distance = actual_distance.view(-1, 1).to(device)
                    output = model(poly1_embedding, poly2_embedding)
                    loss = eval_criterion(output, actual_distance)
                    val_loss += loss.item()

            val_loss /= len(validation_loader)

            ground_truths = []
            outputs = []
            testing_loss = 0.0
            with torch.no_grad():
                for poly1_embedding, poly2_embedding, actual_distance in testing_loader:
                    poly1_embedding = poly1_embedding.view(-1, poly_feature_dim).to(device)
                    poly2_embedding = poly2_embedding.view(-1, poly_feature_dim).to(device)
                    actual_distance = actual_distance.view(-1, 1).to(device)
                    output = model(poly1_embedding, poly2_embedding)
                    loss = eval_criterion(output, actual_distance)
                    testing_loss += loss.item()
                    ground_truths += actual_distance.view(-1).detach().cpu().numpy().tolist()
                    outputs += output.view(-1).detach().cpu().numpy().tolist()

            testing_loss /= len(testing_loader)

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                # plt.scatter(ground_truths, outputs,s=1, alpha=0.5)
                # plt.plot([min(ground_truths), max(ground_truths)], [min(ground_truths), max(ground_truths)], color='red', linestyle='--')
                # plt.xlabel('Ground Truth Distances')
                # plt.ylabel('Predicted Distances')
                # plt.title('Ground Truth vs Predicted Distances')
                # plt.show()
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}, Testing Loss: {testing_loss:.10f}")
            loss_variation.append(testing_loss)
        acc_list.append(np.mean(testing_loss))
    print(f"Average Testing Loss over {num_training} trainings: {np.mean(acc_list):.10f}, Std: {np.std(acc_list):.10f}")