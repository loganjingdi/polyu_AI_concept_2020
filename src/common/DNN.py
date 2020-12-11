import argparse
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import MultiLayerPerceptron
from dataset import AdultDataset

bs = 64 #batch size --> bs * iterations of N = approx. no.samples
lr = 1 #learning rate 
numepoch = 7
seed = 0
N = 10

data = pd.read_csv('census.csv')
overrep = data.loc[data["income"] == "<=50K"]
overrep = pd.DataFrame(overrep) #(
overrepnew = overrep.sample(n = 11208, axis = 0 , random_state = seed)
underrep = data.loc[data["income"] == ">50K"]
underrep = pd.DataFrame(underrep)
data = pd.concat([overrepnew,underrep])
data = pd.DataFrame(data)
income = pd.Index([">50K", "<=50K"])

categorical_feats = ['workclass', 'race', 'education_level', 'marital-status', 'occupation',
                    'relationship', 'sex', 'native-country', 'income']
label_encoder = LabelEncoder() #Convert string values into integers

cts_feats = ['age','education-num','capital-gain','capital-loss','hours-per-week']

#Extract and encode categorical data with label encoder
cat_data = data.drop(columns = cts_feats) # (22416, 9) --> includes income
cat_data_encoded = cat_data.apply(label_encoder.fit_transform) #(22416,9) --> encode into numbers from strings

# Extract 'income' into separate variable
y = cat_data_encoded['income'] # (22416,1) DataFrame --> contains numbers in 'income'

# Drop 'income' from original data
data = data.drop(columns=['income']) #remove from data (concatenated <=50K and >50K originals)

# Drop 'income' from categorical features: We don't want it in one-hot encoding
categorical_feats.remove('income') #remove from list

#Convert y into numpy array
y = y.values  # (22416) <-- convert DataFrame to numpy array

# Encode categorical data with OneHotEncoder (to get rid of inherent bias)
oneh_encoder = OneHotEncoder()

cat_data_encoded_del = cat_data_encoded.drop(columns=['income']) #remove 'income' from encoded cat_data_encoded DataFrame
cat_onehot = oneh_encoder.fit(cat_data_encoded_del.values)
cat_onehot = oneh_encoder.transform(cat_data_encoded_del.values).toarray() # (22416, 97) --> numpy array

cts_data = data.drop(columns = categorical_feats) # (22416,6) --> data (no 'income')

#Compute mean and std. dev. of each column
cts_mean = cts_data.mean(axis = 0) #column-wise avg
cts_std = cts_data.std(axis = 0)

#Normalize continuous features
cts_data = cts_data.sub(cts_mean,axis=1) #subtract all elements by the mean
cts_data = cts_data.div(cts_std,axis=1) #divide all elements by the std dev.
X = np.concatenate([cts_data, cat_onehot], axis = 1) #[][]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)




X_Train = torch.from_numpy(X_train) #2d [[],[],...]
y_Train = torch.from_numpy(y_train) #1d [1,1,0,0,....]

X_Test = torch.from_numpy(X_test)
y_Test = torch.from_numpy(y_test)


def load_data(batch_size):
    # INSTANTIATE 2 AdultDataset !!!CLASSES!!!
    train_ds = AdultDataset(X_Train, y_Train)  # can get the ith sample with the __get_item__ in this class
    valid_ds = AdultDataset(X_Test, y_Test)

    # INSTANTIATE 2 instances of the DataLoader !!!CLASS!!! (passing in train, valid datasets)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(valid_ds, batch_size=bs)

    return train_loader, val_loader


def load_model(learning_rate):
    # INSTANTIATE 1 MLP model !!!CLASS!!!
    model = MultiLayerPerceptron(102)  # 102 is input_size

    # DEFINE loss and optimzer functions
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    return model, loss_fnc, optimizer


def accuracy(predictions, label):
    totalCor = 0
    ind = 0

    for c in predictions.flatten():
        if (c.item() > 0.5):
            r = 1.0
        else:
            r = 0.0

        if (r == label[ind].item()):
            totalCor += 1
        ind += 1

    return (totalCor / len(label))


def evaluate(model, val_loader):
    total_corr = 0

    for i, data in enumerate(val_loader, 0):  # go through each batch
        batch, label = data
        predictbatch = model(batch.float())  # tensor size (bs) --> grad_fnc = sigmoidbackward

        ind = 0
        for c in predictbatch.flatten():
            if (c.item() > 0.5):
                r = 1.0
            else:
                r = 0.0

            if (r == label[ind].item()):
                total_corr += 1
            ind += 1

    return float(total_corr) / len(val_loader.dataset)


def main():
    # Seed for random initial weights
    torch.manual_seed(seed)

    # LOAD DATA
    train_loader, val_loader = load_data(bs)

    model, loss_fnc, optimizer = load_model(lr)  # lr defined here

    batchlossv = 0.0  # values (accumulated per N batches)
    batchaccv = 0.0  # loss/sample (accumulated per N batches)


    acc_batch= []
    loss_batch = []

    start_time = time.time()
    for epoch in range(
            numepoch):  # use trainloader #1 epoch, does prediction on all 17932 training samples (matrix multiply)
        nbatch = []  # batch accum
        ntime = []

        trainloss_list = []  # trainloss accum for loss (accum AVG LOSS every Nth batch)
        trainacc_list = []  # trainacc accum for trainAcc

        testacc_list = []  # testacc accum for validAcc

        batchloss_accum = 0.0  # accumed (loss/batch for N batches) v before
        batchacc_accum = 0.0  # accumed (loss/batch for N batches) v before

        for i, data in enumerate(train_loader, 0):  # ith batch, each sample
            # batchacc = 0.0 # loss/sample (accumulated per N batches)
            model.train()


            batch, label = data  # batch is size bs, each (103 col), lable is size bs, each (1 col)

            batch = Variable(batch).float()
            label = Variable(label).float()

            optimizer.zero_grad()
            predict = model(batch)

            print(predict.squeeze())

            batchloss = loss_fnc(input=predict.squeeze(),
                                 target=label)  # 1 no.-> single batch's loss --> squeeze (bs,1) to (bs)
            batchloss.backward()  # compute gradients
            optimizer.step()  # modify weights and bias

            batchloss_accum = batchloss_accum + batchloss.item()  # sums loss/batch (PyTorch calculates loss over all samples in a batch to a single number)
            print('batchloss', batchloss)

            batchacc = accuracy(predict.squeeze(), label)  # 1 no. --> (bs,)
            batchacc_accum = batchacc_accum + batchacc
            print('batchacc', batchacc)

            if i % N == N - 1:  # print every N batches
                model.eval()  ####ADDED
                vacc = evaluate(model, val_loader)  ####

                print(epoch)
                print("avgloss/batch:", f'{batchloss_accum / N:.4f}', " avgacc/batch: ", f'{batchacc_accum / N:.4f}',
                      " validAcc: ", f'{vacc:.4f}')
                acc_batch.append(batchacc_accum / N)
                loss_batch.append(batchloss_accum / N)

                # prints [epoch, samples gone through (aka. #batches)] avgloss/batch: # avgacc/batch: #

                # accumulate avgloss/N batches and avgacc/N batches (for plotting)
                trainloss_list.append(batchloss_accum / N)
                trainacc_list.append(batchacc_accum / N)
                testacc_list.append(vacc)

                batchloss_accum = 0.0  # reset for next avg (over N batches)
                batchacc_accum = 0.0  # reset

                nbatch.append(i + 1)  # latest epoch

                Nbatchtime = time.time() - start_time
                ntime.append(Nbatchtime)
    ######

    # print('ntime', ntime)
    #
    # print('nbatch', nbatch)
    # print('trainloss_list', trainloss_list)
    #
    # print('trainacc_list', trainacc_list)
    # print('testacc_list', testacc_list)
    print('acc_batch:', acc_batch)
    print('loss_batch:', loss_batch)





if __name__ == "__main__":
    main()





