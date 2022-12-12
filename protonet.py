import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard
import torchsummary
from pathlib import Path

import voxcelebdataset
import voxcelebnetwork
import util

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ProtoNet:

    def __init__(self, learning_rate, log_dir, num_test_tasks, test_interval, print_interval, pretrained_weights):
        self._network = voxcelebnetwork.VoxCelebNetwork()
        if pretrained_weights is not None:
            self._network.load_state_dict(pretrained_weights)

        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            self._network = nn.DataParallel(self._network)
        self._network.to(DEVICE)

        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0
        self._num_test_tasks = num_test_tasks
        self._test_interval = test_interval
        self._print_interval = print_interval

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)

            num_supports = len(labels_support)
            num_queries = len(labels_query)
            num_classes = labels_support[-1] + 1
            K = len(labels_support) / num_classes

            f = self._network(torch.cat((images_support, images_query)))
            f_supports = f[:num_supports]
            f_queries = f[num_supports:]

            prototypes = torch.zeros((num_classes, f_supports.shape[-1])).to(DEVICE)
            prototypes = prototypes.index_add(0, labels_support, f_supports, alpha=1/K)

            distances_support = torch.cat([torch.norm(prototypes - f_supports[i], dim=1).unsqueeze(0) for i in range(num_supports)])
            distances_query = torch.cat([torch.norm(prototypes - f_queries[i], dim=1).unsqueeze(0) for i in range(num_queries)])
            loss_batch.append(F.cross_entropy(-distances_query, labels_query, reduction='mean'))

            accuracy_support_batch.append(util.score(-distances_support, labels_support))
            accuracy_query_batch.append(util.score(-distances_query, labels_query))

        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch)
        )

    def train(self, dataloader_train, dataloader_val, writer):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            loss, accuracy_support, accuracy_query = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % self._print_interval == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'support accuracy: {accuracy_support.item():.3f}, '
                    f'query accuracy: {accuracy_query.item():.3f}'
                )
                writer.add_scalar('loss/train', loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/support',
                    accuracy_support.item(),
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/query',
                    accuracy_query.item(),
                    i_step
                )

            if i_step == 0 or (i_step+1) % self._test_interval == 0:
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for val_task_batch in dataloader_val:
                        loss, accuracy_support, accuracy_query = (
                            self._step(val_task_batch)
                        )
                        losses.append(loss.item())
                        accuracies_support.append(accuracy_support)
                        accuracies_query.append(accuracy_query)
                    loss = np.mean(losses)
                    accuracy_support = np.mean(accuracies_support)
                    accuracy_query = np.mean(accuracies_query)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'support accuracy: {accuracy_support:.3f}, '
                    f'query accuracy: {accuracy_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step+1)
                writer.add_scalar(
                    'val_accuracy/support',
                    accuracy_support,
                    i_step+1
                )
                writer.add_scalar(
                    'val_accuracy/query',
                    accuracy_query,
                    i_step+1
                )

                self._save(i_step+1)

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(self._num_test_tasks)
        print(
            f'Accuracy over {self._num_test_tasks} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    log_dir = args.log_dir
    pretrained_weights = args.pretrained_weights
    if log_dir is None:
        log_dir = f'./logs/voxceleb/protonet.way:{args.num_way}.support:{args.num_support}.query:{args.num_query}.lr:{args.learning_rate}.batch_size:{args.batch_size}'
        if pretrained_weights is not None:
            assert args.mask is not None
            assert args.mask_ratio is not None
            log_dir += f'.mask:{args.mask}.mask_ratio:{args.mask_ratio}'
    print(f'log_dir: {log_dir}')
    print(f'Device: {DEVICE}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
 
    if pretrained_weights is not None:
        print(f'Using pretrained weights from {pretrained_weights}')
        state = torch.load(pretrained_weights)
        pretrained_weights = state['encoder_network_state_dict']

    protonet = ProtoNet(args.learning_rate, log_dir, args.test_tasks, args.test_interval, args.print_interval, pretrained_weights)
    torchsummary.summary(protonet._network, (1, 256, 301), batch_size=args.num_way * (args.num_support + args.num_query), device=DEVICE.type)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations - args.checkpoint_step)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = voxcelebdataset.get_voxceleb_task_dataloader(
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks,
            args.train_url_directory,
            args.spectrogram_directory,
        )
        dataloader_val = voxcelebdataset.get_voxceleb_task_dataloader(
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4,
            args.test_url_directory,
            args.spectrogram_directory,
        )
        protonet.train(
            dataloader_train,
            dataloader_val,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = voxcelebdataset.get_voxceleb_task_dataloader(
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.test_tasks,
            args.test_url_directory,
            args.spectrogram_directory,
        )
        protonet.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform few shot speaker identification using meta learning on the voxceleb dataset')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Path to pretrained weights from MAE')
    parser.add_argument('--mask', type=str, default=None, choices=['none', 'strips', 'boxes'],
                        help='Mask to use for the MAE')
    parser.add_argument('--mask_ratio', type=float, default=None,
                        help='Mask ratio to use')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=5000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
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
    parser.add_argument('--test_tasks', type=int, required=False, default=600,
                        help='Number of test tasks to run when testing (default 600)')

    main_args = parser.parse_args()
    main(main_args)
