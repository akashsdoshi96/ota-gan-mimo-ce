from fedambgan_model.data import loader
from fedambgan_model.server import server
from fedambgan_model.client import client

def federated_learning():
    # hyper parameter
    n_client = 4  #Number of UEs
    n_epoch = 600 #Number of rounds

    # dataset
    print('Initialize Dataset...')
    data_loader = loader()

    # initialize server
    print('Initialize Server...')
    s = server(size=n_client, dataset=data_loader)

    # initialize client
    print('Initialize Client...')
    clients = []
    for i in range(n_client):
        clients.append(client(rank=i))

    # federated learning
    for e in range(0,n_epoch):
        print('\n================== Round {:>3} =================='.format(e + 1))
        X_train = s.return_datasets()
        for i in range(25): #Number of training iterations l
            nc = 0
            for c in clients:
                c.run(X_train[nc],data_loader,i,nc) #Local(UE) training
                nc += 1
            s.aggregate(e,i) #Global(BS) training


if __name__ == '__main__':
    federated_learning()
