import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard
import os

import voxcelebdataset
import voxcelebnetwork
import torchsummary


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VoxCelebNetworkDecoder(nn.Module):
    def __init__(self, encoder_feature_sizes, channel_sizes):
        super().__init__()

        self._encoder_feature_sizes = encoder_feature_sizes
        layers = []
        layers += [nn.Unflatten(dim=1, unflattened_size=encoder_feature_sizes[-2])]
        for i in range(len(channel_sizes) - 1):
            out_ch = channel_sizes[-(i+1)]
            in_ch = channel_sizes[-(i+2)]
            layers += [
                nn.MaxUnpool2d(2),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
                nn.ConvTranspose2d(out_ch, in_ch, 3, padding=1),
            ]
        self._layers = nn.Sequential(*layers)
        self._max_pool_indicies = []
    
    def set_max_pool_indicies(self, max_pool_indicies):
        self._max_pool_indicies = max_pool_indicies

    def forward(self, images):
        x = images
        pool_idx = 0
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, self._max_pool_indicies[-(pool_idx+1)], output_size=self._encoder_feature_sizes[-(i+2)][-2:])
                pool_idx += 1
            else:
                x = layer(x)
        return x


class VoxCelebNetworkMAE:
    def __init__(self, learning_rate: float, epochs, mask: str, mask_ratio: float, log_dir, num_test_images, test_interval, print_interval) -> None:
        self._mask = mask
        self._mask_ratio = mask_ratio

        self._encoder = voxcelebnetwork.VoxCelebNetwork(max_pool_return_indicies=True).to(DEVICE)
        self._decoder = VoxCelebNetworkDecoder(self._encoder._feature_sizes, self._encoder.channel_sizes).to(DEVICE)
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            self._encoder = nn.DataParallel(self.encoder)
            self._decoder = nn.DataParallel(self.decoder)

        x = torch.rand((16, 1, 256, 301)).to(DEVICE)
        y = self._step(x)

        self._optimizer = torch.optim.Adam(
            [{self._encoder.parameters()}, {self._decoder.parameters()}],
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_epoch = 0
        self._num_train_epochs = epochs
        self._num_test_images = num_test_images
        self._test_interval = test_interval
        self._print_interval = print_interval

    def load(self, epoch):
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{epoch}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._encoder.load_state_dict(state['encoder_network_state_dict'])
            self._decoder.load_state_dict(state['decoder_network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_epoch = epoch + 1
            print(f'Loaded checkpoint iteration {epoch=}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {epoch} found.'
            )

    def save(self, epoch):
        torch.save(
            {
                'encoder_network_state_dict': self._encoder.state_dict(),
                'decoder_network_state_dict': self._decoder.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
            },
            f'{os.path.join(self._log_dir, "state")}{epoch}.pt'
        )
        print(f'Saved checkpoint {epoch=}')
    
    def _step(self, image_batch):
        x = image_batch.to(DEVICE)
        batch_size = x.shape[0]

        mask = torch.ones_like(image_batch, requires_grad=True)
        if self._mask == 'boxes':
            mask = torch.ones_like(image_batch)
            box_size = (16, 30)
            grid_dims = (16, 10)
            row_idxs = np.arange(grid_dims[0]) * box_size[0]
            row_idxs = np.append(row_idxs, 256)
            col_idxs = np.arange(grid_dims[1]) * box_size[1]
            col_idxs = np.append(col_idxs, 301)
            num_boxes = np.prod(grid_dims)
            num_boxes_to_mask = np.round(num_boxes*self._mask_ratio).astype(int)
            for i in range(batch_size):
                boxes_to_mask = np.random.choice(num_boxes, replace=False, size=num_boxes_to_mask)
                boxes_to_mask = [(box // grid_dims[1], box % grid_dims[1]) for box in boxes_to_mask]
                for row, col in boxes_to_mask:
                    row_start, row_end = row_idxs[row], row_idxs[row+1]
                    col_start, col_end = col_idxs[col], col_idxs[col+1]
                    mask[i, 0, row_start:row_end, col_start:col_end] = 0.0

        x = self._encoder(x * mask)
        self._decoder.set_max_pool_indicies(self._encoder._max_pool_indicies)
        x = self._decoder(x)

        # Only compute loss on masked portions, set all other values to 0
        rev_mask = -mask + 1
        return F.mse_loss(x * rev_mask, image_batch * rev_mask)

    def train(self, dataloader_train, dataloader_val, writer):
        print(f'Starting training at iteration {self._start_train_step}.')
        self._encoder.train()
        self._decoder.train()
        for epoch in range(self._num_train_epochs, start=self._start_train_epoch):
            for i_step, image_batch, _ in enumerate(dataloader_train):
                self._optimizer.zero_grad()
                loss = self._step(image_batch)
                loss.backward()
                self._optimizer.step()

                if i_step % self._print_interval == 0:
                    print(
                        f'Iteration {i_step}: '
                        f'loss: {loss.item():.3f}, '
                    )
                    writer.add_scalar('loss/train', loss.item(), i_step)

            with torch.no_grad():
                losses = []
                for val_task_batch in dataloader_val:
                    loss = self._step(val_task_batch)
                    losses.append(loss.item())
                loss = np.mean(losses)
            print(
                f'Validation: '
                f'loss: {loss:.3f}, '
            )
            writer.add_scalar('loss/val', loss, i_step)
            self.save(epoch)

    def test(self):
        pass


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/voxceleb/mae.mask:{args.mask}.mask_ratio:{args.mask_ratio}.epochs:{args.num_epochs}.lr:{args.learning_rate}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    print(f'Device: {DEVICE}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    mae = VoxCelebNetworkMAE(args.learning_rate, args.num_epochs, args.mask, args.mask_ratio, args.log_dir, args.test_count, args.test_interval, args.print_interval)
    torchsummary.summary(mae._encoder, (1, 256, 301), batch_size=args.batch_size, device=DEVICE.type)
    torchsummary.summary(mae._decoder, (1024), batch_size=args.batch_size, device=DEVICE.type)
    if args.checkpoint_epoch > -1:
        mae.load(args.checkpoint_epoch)
    else:
        print('Checkpoint loading skipped.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform pre-training through a MAE on the voxceleb dataset')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--mask', type=str, default='none', choices=['none', 'strips', 'boxes'],
                        help='Mask to use for the MAE (default="none")')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Mask ratio to use (default=0.75)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs to train for')
    parser.add_argument('--checkpoint_epoch', type=int, default=-1,
                        help=('checkpoint epoch to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--train_url_directory', type=str, required=True, 
                        help='Voxceleb "URLs and timestamps" directory for the train data')
    parser.add_argument('--test_url_directory', type=str, required=True, 
                        help='Voxceleb "URLs and timestamps" directory for the test data')
    parser.add_argument('--spectrogram_directory', type=str, required=True, 
                        help='Directory where the spectrograms are stored of each celebrities utterances')
    parser.add_argument('--test_interval', type=int, required=False, default=50,
                        help='Specify how often to test the model during training (default 50)')
    parser.add_argument('--print_interval', type=int, required=False, default=10,
                        help='Specify how often to test the model during training (default 10)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--test_count', type=int, required=False, default=5000,
                        help='Number of test images to run when testing (default 5000)')

    args = parser.parse_args()
    main(args)