from keras.optimizers import Adam
from lib.model_acgan_cifar10 import *
from lib.utils import *
from param import *

# Set CUDA visible device to GPU:0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

# Adam parameters suggested in https://arxiv.org/abs/1511.06434
adam_lr = 0.0002
adam_beta_1 = 0.5


def train(prog=True):
    """Main function to train GAN"""

    # Load MNIST
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train, x_test = x_train.squeeze(), x_test.squeeze()
    y_train, y_test = y_train.squeeze(), y_test.squeeze()

    # Build model
    d = build_discriminator()
    g = build_generator()

    # Set up optimizers
    adam = Adam(lr=adam_lr, beta_1=adam_beta_1)

    # Set loss function and compile models
    g.compile(optimizer=adam, loss='binary_crossentropy')
    d.compile(
        optimizer=adam,
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    combined = combine_g_d(g, d)
    combined.compile(
        optimizer=adam,
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    n_train = x_train.shape[0]
    n_batch = int(n_train / BATCH_SIZE)
    for epoch in range(N_EPOCH):
        print('Epoch {} of {}'.format(epoch + 1, N_EPOCH))
        progress_bar = Progbar(target=n_batch)

        epoch_g_loss = []
        epoch_d_loss = []

        for index in range(n_batch):
            progress_bar.update(index, force=True)
            d_loss = np.zeros(3)

            # Train the discriminator for N_DIS iterations before training
            # the generator once
            for _ in range(N_DIS):
                # ----------------- Train discriminator ---------------------- #
                # Train with real samples first
                smp_ind = np.random.choice(n_train, BATCH_SIZE)
                x_real = x_train[smp_ind]
                y_real = y_train[smp_ind]
                if index % 30 != 0:
                    y_d = np.random.uniform(0.7, 1.2, size=(BATCH_SIZE, ))
                else:
                    y_d = np.random.uniform(0.0, 0.3, size=(BATCH_SIZE, ))
                d_loss += d.train_on_batch(x_real, [y_d, y_real])

                # Train with generated samples, generate samples from g
                z = np.random.normal(0, 0.5, (BATCH_SIZE, LATENT_SIZE))
                # Sample some labels from p_c
                y_sampled = np.random.randint(0, 10, BATCH_SIZE)
                x_g = g.predict([z, y_sampled.reshape((-1, 1))], verbose=0)
                if index % 30 != 0:
                    y_d = np.random.uniform(0.0, 0.3, size=(BATCH_SIZE, ))
                else:
                    y_d = np.random.uniform(0.7, 1.2, size=(BATCH_SIZE, ))
                d_loss += d.train_on_batch(x_g, [y_d, y_sampled])

            # Log average discriminator loss over N_DIS
            epoch_d_loss.append(d_loss / N_DIS)

            # ---------------- Train generator ------------------------------- #
            # Generate 2 * BATCH_SIZE samples to match d's batch size
            z = np.random.uniform(0, 0.5, (2 * BATCH_SIZE, LATENT_SIZE))
            y_sampled = np.random.randint(0, 10, 2 * BATCH_SIZE)
            y_g = np.random.uniform(0.7, 1.2, size=(2 * BATCH_SIZE, ))

            epoch_g_loss.append(combined.train_on_batch(
                [z, y_sampled.reshape((-1, 1))], [y_g, y_sampled]))

        print('\nTesting for epoch {}:'.format(epoch + 1))
        n_test = x_test.shape[0]

        # ---------------- Test discriminator -------------------------------- #
        z = np.random.uniform(0, 0.5, (n_test, LATENT_SIZE))
        y_sampled = np.random.randint(0, 10, n_test)
        x_g = g.predict([z, y_sampled.reshape((-1, 1))], verbose=0)

        x_d = np.concatenate((x_test, x_g))
        y_d = np.array([1] * n_test + [0] * n_test)
        y_aux = np.concatenate((y_test, y_sampled), axis=0)

        d_test_loss = d.evaluate(x_d, [y_d, y_aux], verbose=0)
        d_train_loss = np.mean(np.array(epoch_d_loss), axis=0)

        # ---------------- Test generator ------------------------------------ #
        z = np.random.uniform(0, 0.5, (2 * n_test, LATENT_SIZE))
        y_sampled = np.random.randint(0, 10, 2 * n_test)
        y_g = np.ones(2 * n_test)

        g_test_loss = combined.evaluate(
            [z, y_sampled.reshape((-1, 1))], [y_g, y_sampled], verbose=0)
        g_train_loss = np.mean(np.array(epoch_g_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(g_train_loss)
        train_history['discriminator'].append(d_train_loss)
        test_history['generator'].append(g_test_loss)
        test_history['discriminator'].append(d_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *d.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # Aave weights every epoch
        g.save_weights("{}weight_g_epoch_{:03d}.hdf5".format(
            WEIGHT_DIR, epoch), True)
        d.save_weights("{}weight_d_epoch_{:03d}.hdf5".format(
            WEIGHT_DIR, epoch), True)

        # generate some digits to display
        noise = np.random.uniform(0, 0.5, (100, LATENT_SIZE))

        sampled_labels = np.array([
            [i] * 10 for i in range(10)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = g.predict(
            [noise, sampled_labels], verbose=0)

        def vis_square(data, padsize=1, padval=-1):

            # force the number of filters to be square
            n = int(np.ceil(np.sqrt(data.shape[0])))
            padding = ((0, n ** 2 - data.shape[0]), (0, padsize),
                       (0, padsize)) + ((0, 0),) * (data.ndim - 3)
            data = np.pad(data, padding, mode='constant',
                          constant_values=(padval, padval))

            # tile the filters into an image
            data = data.reshape((n, n) + data.shape[1:]).transpose(
                (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
            data = data.reshape(
                (n * data.shape[1], n * data.shape[3]) + data.shape[4:])

            return (data * SCALE + SCALE).astype(np.uint8)

        img = vis_square(generated_images)

        Image.fromarray(img).save(
            '{}plot_epoch_{:03d}_generated.png'.format(VIS_DIR, epoch))

    pickle.dump({'train': train_history, 'test': test_history},
                open('wgan-history.pkl', 'wb'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-prog", dest="prog", action="store_false")
    parser.set_defaults(prog=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    train(prog=args.prog)
