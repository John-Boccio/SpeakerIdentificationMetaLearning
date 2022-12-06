import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard
import os

import voxcelebdataset
import voxcelebnetwork


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
    def __init__(self, learning_rate: float, num_iterations: int, mask: str, mask_ratio: float, log_dir, test_interval, print_interval) -> None:
        self._mask = mask
        self._mask_ratio = mask_ratio

        self._encoder = voxcelebnetwork.VoxCelebNetwork(max_pool_return_indicies=True).to(DEVICE)
        self._decoder = VoxCelebNetworkDecoder(self._encoder._feature_sizes, self._encoder.channel_sizes).to(DEVICE)
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            self._encoder = nn.DataParallel(self.encoder)
            self._decoder = nn.DataParallel(self.decoder)

        self._optimizer = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_iteration = 0
        self._num_train_iterations = num_iterations
        self._test_interval = test_interval
        self._print_interval = print_interval

    def load(self, iteration):
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{iteration}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._encoder.load_state_dict(state['encoder_network_state_dict'])
            self._decoder.load_state_dict(state['decoder_network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_iteration = iteration
            print(f'Loaded checkpoint iteration {iteration=}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {iteration} found.'
            )

    def save(self, iteration):
        torch.save(
            {
                'encoder_network_state_dict': self._encoder.state_dict(),
                'decoder_network_state_dict': self._decoder.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
            },
            f'{os.path.join(self._log_dir, "state")}{iteration}.pt'
        )
        print(f'Saved checkpoint {iteration=}')
    
    def _step(self, image_batch, labels):
        image_batch = image_batch.to(DEVICE)
        batch_size = image_batch.shape[0]

        mask = torch.ones_like(image_batch)
        if self._mask == 'boxes':
            mask = torch.ones_like(image_batch)
            box_size = (16, 30)
            grid_dims = (16, 10)
            row_idxs = np.arange(grid_dims[0]) * box_size[0]
            row_idxs = np.append(row_idxs, 256)
            col_idxs = np.arange(grid_dims[1]) * box_size[1]
            col_idxs = np.append(col_idxs, 301)
            num_boxes = np.prod(grid_dims)
            num_boxes_to_mask = np.round(num_boxes * self._mask_ratio).astype(int)
            for i in range(batch_size):
                boxes_to_mask = np.random.choice(num_boxes, replace=False, size=num_boxes_to_mask)
                boxes_to_mask = [(box // grid_dims[1], box % grid_dims[1]) for box in boxes_to_mask]
                for row, col in boxes_to_mask:
                    row_start, row_end = row_idxs[row], row_idxs[row+1]
                    col_start, col_end = col_idxs[col], col_idxs[col+1]
                    mask[i, :, row_start:row_end, col_start:col_end] = 0.0
        elif self._mask =='strips':
            strip_size = 10
            num_strips = 301 // strip_size
            strip_idxs = np.arange(num_strips) * strip_size
            strip_idxs = np.append(strip_idxs, 301)
            num_strips_to_mask = np.round(num_strips*self._mask_ratio).astype(int)
            for i in range(batch_size):
                strips_to_mask = np.random.choice(num_strips, replace=False, size=num_strips_to_mask)
                for col in strips_to_mask:
                    col_start, col_end = strip_idxs[col], strip_idxs[col+1]
                    mask[i, :, :, col_start:col_end] = 0.0
        mask.requires_grad = True
        mask = mask.to(DEVICE)

        x = self._encoder(image_batch * mask)
        self._decoder.set_max_pool_indicies(self._encoder._max_pool_indicies)
        x = self._decoder(x)

        # Only compute loss on masked portions, set all other values to 0
        opposite_mask = (-mask + 1)
        return F.mse_loss(x * opposite_mask, image_batch * opposite_mask)

    def train(self, dataloader_train, dataloader_val, writer):
        print(f'Starting training at iteration {self._start_train_iteration}.')
        self._encoder.train()
        self._decoder.train()
        for i_step, batch in enumerate(dataloader_train, start=self._start_train_iteration):
            image_batch = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)

            self._optimizer.zero_grad()
            loss = self._step(image_batch, labels)
            loss.backward()
            self._optimizer.step()

            if i_step % self._print_interval == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                )
                writer.add_scalar('loss/train', loss.item(), i_step)

            if i_step == 0 or (i_step+1) % self._test_interval == 0:
                with torch.no_grad():
                    losses = []
                    for batch in dataloader_val:
                        image_batch = batch[0].to(DEVICE)
                        labels = batch[1].to(DEVICE)
                        loss = self._step(image_batch, labels)
                        losses.append(loss.item())
                    loss = np.mean(losses)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step+1)
                self.save(i_step+1)

    def test(self, dataloader_test, test_iterations):
        losses = []
        for batch in dataloader_test:
            image_batch = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            losses.append(self._step(image_batch, labels).item())
        mean = np.mean(losses)
        std = np.std(losses)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(test_iterations)
        print(
            f'Loss over {test_iterations} test iterations: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/voxceleb/mae.mask:{args.mask}.mask_ratio:{args.mask_ratio}.lr:{args.learning_rate}.batch_size:{args.batch_size}'
    print(f'log_dir: {log_dir}')
    print(f'Device: {DEVICE}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    mae = VoxCelebNetworkMAE(args.learning_rate, args.num_iterations, args.mask, args.mask_ratio, log_dir, args.test_interval, args.print_interval)
    if args.checkpoint_iteration > -1:
        mae.load(args.checkpoint_iteration)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        checkpoint = args.checkpoint_iteration if args.checkpoint_iteration != -1 else 0
        train_iterations = args.batch_size * (args.num_iterations - checkpoint)
        print(f'Training MAE with {args.num_iterations} iterations')
        dataloader_train = voxcelebdataset.get_voxceleb_dataloader(
            args.batch_size,
            train_iterations,
            args.train_url_directory,
            args.spectrogram_directory,
        )
        dataloader_val = voxcelebdataset.get_voxceleb_dataloader(
            args.batch_size,
            args.batch_size * 15,
            args.test_url_directory,
            args.spectrogram_directory,
        )
        mae.train(
            dataloader_train,
            dataloader_val,
            writer
        )
    else:
        print(f'Testing MAE with {args.test_iterations} iterations')
        dataloader_test = voxcelebdataset.get_voxceleb_dataloader(
            args.batch_size,
            args.batch_size * args.test_iterations,
            args.test_url_directory,
            args.spectrogram_directory,
        )
        mae.test(dataloader_test, args.test_iterations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform pre-training through a MAE on the voxceleb dataset')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Initial learning rate for the network')
    parser.add_argument('--mask', type=str, default='none', choices=['none', 'strips', 'boxes'],
                        help='Mask to use for the MAE (default="none")')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Mask ratio to use (default=0.75)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_iterations', type=int, default=5000,
                        help='number of iterations to train for')
    parser.add_argument('--checkpoint_iteration', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--train_url_directory', type=str, required=True, 
                        help='Voxceleb "URLs and timestamps" directory for the train data')
    parser.add_argument('--test_url_directory', type=str, required=True, 
                        help='Voxceleb "URLs and timestamps" directory for the test data')
    parser.add_argument('--spectrogram_directory', type=str, required=True, 
                        help='Directory where the spectrograms are stored of each celebrities utterances')
    parser.add_argument('--print_interval', type=int, required=False, default=10,
                        help='Specify how often to print during training (default 10)')
    parser.add_argument('--test_interval', type=int, required=False, default=500,
                        help='Specify how often to test the model during training (default 500)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--test_iterations', type=int, required=False, default=100,
                        help='Number of iteration to run when testing (default 100)')

    args = parser.parse_args()
    main(args)