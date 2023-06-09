# ERA_Session6

# Part 1

##Backpropogation in network:

input: 2 neurons

one hidden layer : 2 neurons

output : 2 neurons

<img width="718" alt="image" src="https://github.com/sunandhini96/ERA_Session6/assets/63030539/0a213da1-1994-4ee3-abfd-615503ba70ff">


<img width="425" alt="image" src="https://github.com/sunandhini96/ERA_Session6/assets/63030539/1d5d9180-e8ee-46fd-b300-db432edacbd2">



In Forward, first stage network pass through all initial weights. For hidden layer it took the input from input layer and by using initial weights, hidden layer neurons get temporary hidden value, by applying activation function hidden layer outputs would be updated. Then pass to the output layer. After the output layer each output related to corresponding error(difference between target and network predicted output). By backpropogating loss value should be minimized, weights will be updated.

By using different learning rates we observe for simple input learning rate increases convergence fast but for complex data when learning rate small then loss converge but slow process.

![image](https://github.com/sunandhini96/ERA_Session6/assets/63030539/d5d8b473-489f-4f13-8bb6-060d704c2950)
![image](https://github.com/sunandhini96/ERA_Session6/assets/63030539/1242a67d-a544-4c34-862e-4b139836aac0)
