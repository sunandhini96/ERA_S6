# ERA_S6

# Part 1

##Backpropogation in network:

input: 2 neurons

one hidden layer : 2 neurons

output : 2 neurons

<img width="718" alt="image" src="https://github.com/sunandhini96/ERA_Session6/assets/63030539/0a213da1-1994-4ee3-abfd-615503ba70ff">

## Equations :


<img width="425" alt="image" src="https://github.com/sunandhini96/ERA_Session6/assets/63030539/1d5d9180-e8ee-46fd-b300-db432edacbd2">





In Forward, first stage network pass through all initial weights. For hidden layer it took the input from input layer and by using initial weights, hidden layer neurons get temporary hidden value, by applying activation function hidden layer outputs would be updated. Then pass to the output layer. After the output layer each output related to corresponding error(difference between target and network predicted output). By backpropogating loss value should be minimized, weights will be updated.



![image](https://github.com/sunandhini96/ERA_Session6/assets/63030539/d5d8b473-489f-4f13-8bb6-060d704c2950)
![image](https://github.com/sunandhini96/ERA_Session6/assets/63030539/1242a67d-a544-4c34-862e-4b139836aac0)


By using different learning rates, we observed by increasing the learning rate -> convergence fast but for complex data when learning rate small then loss converge but slow process.


# Part 2 :

## **Network** :

import torch.nn.functional as F

dropout_value = 0.1

class Net(nn.Module):

    def __init__(self):
    
        super(Net, self).__init__()
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05)
        ) # output_size = 26 | RF 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        ) # output_size = 24 | RF 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24 | RF 5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12 | RF 7

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(0.05)
        ) # output_size = 10 | RF 11
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(0.05)
        ) # output_size = 8 | RF 15
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(0.05)
        ) # output_size = 6 | RF 19
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(14),
            #nn.Dropout(0.05)
        ) # output_size = 6 | RF 23
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1 | RF 43

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) #RF 43


        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
     


## **Summary of the model ** :

<img width="438" alt="image" src="https://github.com/sunandhini96/ERA_S6/assets/63030539/0e1b727e-baac-4782-b1b6-d834217a90dc">


## Output Logs over 15 epochs :

EPOCH: 0
loss=0.07790514081716537 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 21.28it/s]

Test set: Average loss: 0.0763, Accuracy: 9796/10000 (97.96%)

EPOCH: 1
loss=0.09397108107805252 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.60it/s]

Test set: Average loss: 0.0633, Accuracy: 9808/10000 (98.08%)

EPOCH: 2
loss=0.018836800009012222 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 24.92it/s]

Test set: Average loss: 0.0315, Accuracy: 9905/10000 (99.05%)

EPOCH: 3
loss=0.08602729439735413 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.84it/s]

Test set: Average loss: 0.0285, Accuracy: 9909/10000 (99.09%)

EPOCH: 4
loss=0.006202130112797022 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.84it/s]

Test set: Average loss: 0.0243, Accuracy: 9931/10000 (99.31%)

EPOCH: 5
loss=0.02641492523252964 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.73it/s]

Test set: Average loss: 0.0270, Accuracy: 9913/10000 (99.13%)

EPOCH: 6
loss=0.02400313876569271 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.53it/s]

Test set: Average loss: 0.0218, Accuracy: 9937/10000 (99.37%)

EPOCH: 7
loss=0.02343975193798542 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.52it/s]

Test set: Average loss: 0.0206, Accuracy: 9943/10000 (99.43%)

EPOCH: 8
loss=0.029719745740294456 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.26it/s]

Test set: Average loss: 0.0204, Accuracy: 9940/10000 (99.40%)

EPOCH: 9
loss=0.03981277346611023 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.46it/s]

Test set: Average loss: 0.0207, Accuracy: 9944/10000 (99.44%)

EPOCH: 10
loss=0.004856004845350981 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.22it/s]

Test set: Average loss: 0.0203, Accuracy: 9945/10000 (99.45%)

EPOCH: 11
loss=0.032489705830812454 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.65it/s]

Test set: Average loss: 0.0200, Accuracy: 9943/10000 (99.43%)

EPOCH: 12
loss=0.05288465693593025 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.10it/s]

Test set: Average loss: 0.0195, Accuracy: 9941/10000 (99.41%)

EPOCH: 13
loss=0.0554722435772419 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.94it/s]

Test set: Average loss: 0.0201, Accuracy: 9938/10000 (99.38%)

EPOCH: 14
loss=0.0320533886551857 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.49it/s]

Test set: Average loss: 0.0189, Accuracy: 9942/10000 (99.42%)

We created the network like squeeze and expand (transition layer in between) to reduce the parameters and increase the performance. We trained the our model with a 15 epochs, we observed 7 th epoch itself achieved 99.4 % test accuracy. 
