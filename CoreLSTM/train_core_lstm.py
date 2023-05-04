"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import torch 
import matplotlib.pyplot as plt
import random

import sys
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)      
# Before run: replace ... with current directory path

from CoreLSTM.core_lstm import CORE_NET


class LSTM_Trainer():

    """
        Class to train core LSTM model for optical illusions.
        Training on multiple versions of the data (i.e. mirrored or not, 
        one or both rotation directions). 
        
    """

    ## General parameters
    random.seed(1111)

    def __init__(self, 
            loss_function, 
            learning_rate, 
            betas, 
            weight_decay, 
            batch_size, 
            hidden_num, 
            layer_norm, 
            num_dim, 
            num_feat
        ):

        self.device = torch.device('cpu') # 'cuda' if torch.cuda.is_available() else 
        print(f'DEVICE TrainM: {self.device}')

        self._model = CORE_NET(
            input_size=num_dim*num_feat, 
            hidden_layer_size=hidden_num, 
            layer_norm=layer_norm
        )
        self.batch_size = batch_size
        self._loss_function = loss_function
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), 
            lr=learning_rate, 
            betas=betas, 
            weight_decay=weight_decay
        )

        print('Initialized model!')
        print(self._model)
        print(self._loss_function)
        print(self._optimizer)


    def train(self, 
            epochs, 
            train_sequences,         # tensor of training sequences SxN
            save_path, 
            preprocessor, 
            noise
        ):

        losses = []

        [seq_ins, 
            seq_tars, 
            batch_size, 
            num_batches, 
            num_input] = self.restructure_training_sequence(train_sequences) 

        num_batches_new = int((num_batches*(2/3)))

        for ep in range(epochs):

            single_losses = torch.tensor([0.0], device=self.device) 

            self._model.zero_grad()
            self._optimizer.zero_grad()

            # the following can be paralized!!!! But takes up memory!!!

            for i in range(len(train_sequences)):
                idx = torch.randperm(num_batches)[:round((num_batches_new))]

                state = self._model.init_hidden(num_batches_new)

                ins_t = seq_ins[i][idx]          # shape: num batches x batch size x num input
                tars_t = seq_tars[i][idx]        # shape: num batches x batch size x num input

                # add noise
                if noise is not None: 
                    ins_t = preprocessor.add_noise(ins_t, noise)

                outs = []
                for j in range(batch_size):
                    _input = ins_t[:,j,:].view(num_batches_new, num_input).to(self.device)
                    out, state = self._model.forward(_input, state)
                    outs.append(out)

                outs = torch.stack(outs).to(self.device)        # shape: batch size x num batches x num input
                single_loss = self._loss_function(outs, tars_t.permute(1,0,2))                  

                single_loss.backward()

                with torch.no_grad():
                    single_losses += single_loss

            self._optimizer.step()
            
            with torch.no_grad():
                ep_loss = single_losses.clone().detach()

                # save loss of epoch
                losses.append(ep_loss.item())
                if ep%25 == 1:
                    print(f'epoch: {ep:3} loss: {single_losses.item():10.8f}')
        
        print(f'epoch: {ep:3} loss: {single_losses.item():10.10f}')

        self.save_model(save_path)

        return losses


    def restructure_training_sequence(self, ts):
        seq_ins = []
        seq_tars = []

        for train_sequence in ts:
            inputs = []
            targets = []

            for seq, label in train_sequence:
                inputs.append(seq)
                target = torch.cat((seq[1:,:], label), dim=0)
                targets.append(target)
            
            ins = []
            tars = []
            for i in range(len(train_sequence)):
                ins.append(inputs[i])
                tars.append(targets[i])
            ins = torch.stack(ins).to(self.device)
            tars = torch.stack(tars).to(self.device)

            seq_ins += [ins]
            seq_tars += [tars]

        batch_size = seq.size()[0]
        num_batches = ins.size()[0]
        num_input = ins.size()[2]

        return [seq_ins, 
                seq_tars, 
                batch_size, 
                num_batches, 
                num_input] 
    

    def plot_losses(self, losses, plot_path):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(losses, 'r')
        axes.grid(True)
        axes.set_xlabel('epochs')
        axes.set_ylabel('loss')
        axes.set_title('History of MSELoss during training')

        
        plt.savefig(f'{plot_path}_losses.png')
        plt.savefig(f'{plot_path}_losses.pdf')
        # plt.show()


    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
        print('Model was saved in: ' + path)


def main():
    pass

if __name__ == '__main__':
    main()