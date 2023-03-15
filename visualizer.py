import csv
import glob
import os
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from matplotlib.colors import LinearSegmentedColormap

PATH_RESULTS = 'experiment/results/'
OUTPUT_PATH = 'results.csv'
OUTPUT_IMG = 'graphs/'


# resultFolders = glob.glob(PATH_RESULTS + '*')
# with open(OUTPUT_PATH, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['name', 'duration', 'epochs', 'epoch_duration', 'scaled_epoch_duration', 'scaled_duration',])
#     writable_rows = []
#
#     for resultFolder in resultFolders:
#         testName = os.path.basename(resultFolder)
#
#         # skip old testcases
#         if not testName.endswith('-1'):
#             continue
#
#         datasize = int(testName.split('-')[3])
#         fileName = resultFolder + '/log_fit.txt'
#
#         with open(fileName) as f:
#             epochs = len(f.readlines()) - 1
#
#         fileName = resultFolder + '/time.txt'
#
#         with open(fileName) as f:
#             lines = f.readlines()
#             strDuration = lines[2]
#             d = 0
#             if ',' in strDuration:
#                 splits = strDuration.split(', ')
#                 d = int(splits[0].split(' ')[0])
#                 strDuration = splits[1]
#             dt = datetime.strptime(strDuration, "%H:%M:%S.%f")
#
#             duration = timedelta(days=d, hours=dt.hour, minutes=dt.minute, seconds=dt.second)
#             duration = duration.total_seconds()
#         duration_scaled = duration / datasize
#         duration_per_epoch = duration / epochs
#         duration_per_epoch_scaled = duration_per_epoch / datasize
#
#         writable_rows.append([
#             testName,
#             duration,
#             epochs,
#             duration_per_epoch,
#             duration_per_epoch_scaled,
#             duration_scaled,
#         ])
#
#     writer.writerows(natsorted(writable_rows, key=lambda writable_row: writable_row[0]))


def collect_rows_match_regex(reg):
    found_rows = []
    with open(OUTPUT_PATH, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if re.match(reg, row[0]):
                found_rows.append(row)
    return found_rows


def collect_values(reg, column_idx):
    found_rows = collect_rows_match_regex(reg)
    return list(map(lambda found_row: float(found_row[column_idx]), found_rows))

# g2 - pci speed
def plot_pci_speed(title, model_name, data_size, gpu_speed, with_pcie_3=False):
    plt.title(title)
    group_names = ['4 GB/s', '8 GB/s']
    if with_pcie_3:
        group_names.append('16 GB/s')
    group_count = len(group_names)
    x_axis_positions = np.arange(group_count)
    batch_32 = collect_values(model_name + r'-\d-' + gpu_speed + '-' + data_size + '-32-1', 3)
    batch_64 = collect_values(model_name + r'-\d-' + gpu_speed + '-' + data_size + '-64-1', 3)
    batch_128 = collect_values(model_name + r'-\d-' + gpu_speed + '-' + data_size + '-128-1', 3)

    width = 0.8 / 3
    plt.bar(x_axis_positions - width, batch_32, width, label='Batch 32')
    plt.bar(x_axis_positions, batch_64, width, label='Batch 64')
    # if with_pcie_3:
    plt.bar(x_axis_positions + width, batch_128, width, label='Batch 128')

    plt.xticks(x_axis_positions, group_names)

    plt.ylabel('Dauer Epoche (s)')
    plt.xlabel('PCIe Geschwindigkeit')
    plt.legend()


def save_pci_speed(save_name, title, model_name, data_size, gpu_speed, with_pcie_3=False):
    plt.clf()
    plot_pci_speed(title, model_name, data_size, gpu_speed, with_pcie_3)
    plt.savefig(OUTPUT_IMG + save_name + '.png')


def build_pci_speed():
    save_pci_speed('inception-g2-33-2', "Inception G3 - GPU 33% Daten 2%", 'inception', '2', '33', True)
    save_pci_speed('inception-g2-66-2', "Inception G3 - GPU 66% Daten 2%", 'inception', '2', '66', True)
    save_pci_speed('inception-g2-100-2', "Inception G3 - GPU 100% Daten 2%", 'inception', '2', '100', True)

    save_pci_speed('inception-g2-33-5', "Inception G3 - GPU 33% Daten 5%", 'inception', '5', '33')
    save_pci_speed('inception-g2-66-5', "Inception G3 - GPU 66% Daten 5%", 'inception', '5', '66')
    save_pci_speed('inception-g2-100-5', "Inception G3 - GPU 100% Daten 5%", 'inception', '5', '100')

    save_pci_speed('inception-g2-33-10', "Inception G3 - GPU 33% Daten 10%", 'inception', '10', '33')
    save_pci_speed('inception-g2-66-10', "Inception G3 - GPU 66% Daten 10%", 'inception', '10', '66')
    save_pci_speed('inception-g2-100-10', "Inception G3 - GPU 100% Daten 10%", 'inception', '10', '100')

    save_pci_speed('transformer-g2-33-2', "Transformer G3 - GPU 33% Daten 2%", 'transformer', '2', '33', True)
    save_pci_speed('transformer-g2-66-2', "Transformer G3 - GPU 66% Daten 2%", 'transformer', '2', '66', True)
    save_pci_speed('transformer-g2-100-2', "Transformer G3 - GPU 100% Daten 2%", 'transformer', '2', '100', True)

    save_pci_speed('transformer-g2-33-5', "Transformer G3 - GPU 33% Daten 5%", 'transformer', '5', '33')
    save_pci_speed('transformer-g2-66-5', "Transformer G3 - GPU 66% Daten 5%", 'transformer', '5', '66')
    save_pci_speed('transformer-g2-100-5', "Transformer G3 - GPU 100% Daten 5%", 'transformer', '5', '100')


# g3 - gpu speed
def plot_gpu_speed(title, model_name, data_size, batch_size, with_pcie_3=False):
    group_names = ['33%', '66%', '100%']

    pcie_1 = collect_values(model_name + r'-1-\d+-' + data_size + '-' + batch_size + '-1', 3)
    pcie_2 = collect_values(model_name + r'-2-\d+-' + data_size + '-' + batch_size + '-1', 3)
    plt.title(title)
    plt.plot(group_names, pcie_1, label='PCIe 1')
    plt.plot(group_names, pcie_2, label='PCIe 2')
    if with_pcie_3:
        pcie_3 = collect_values(model_name + r'-3-\d+-' + data_size + '-' + batch_size + '-1', 3)
        plt.plot(group_names, pcie_3, label='PCIe 3')

    plt.ylabel('Dauer Epoche (s)')
    plt.xlabel('GPU')
    plt.legend()
    plt.grid(True)


def save_gpu_speed(save_name, title, model_name, data_size, batch_size, with_pcie_3=False):
    plt.clf()
    plot_gpu_speed(title, model_name, data_size, batch_size, with_pcie_3)
    plt.savefig(OUTPUT_IMG + save_name + '.png')


def build_gpu_speed():
    save_gpu_speed('inception-g3-32-2', "Inception G2 - Batch 32 Daten 2%", 'inception', '2', '32', True)
    save_gpu_speed('inception-g3-64-2', "Inception G2 - Batch 64 Daten 2%", 'inception', '2', '64', True)
    save_gpu_speed('inception-g3-128-2', "Inception G2 - Batch 128 Daten 2%", 'inception', '2', '128', True)

    save_gpu_speed('inception-g3-32-5', "Inception G2 - Batch 32 Daten 5%", 'inception', '5', '32')
    save_gpu_speed('inception-g3-64-5', "Inception G2 - Batch 64 Daten 5%", 'inception', '5', '64')
    save_gpu_speed('inception-g3-128-5', "Inception G2 - Batch 128 Daten 5%", 'inception', '5', '128')

    save_gpu_speed('inception-g3-32-10', "Inception G2 - Batch 32 Daten 10%", 'inception', '10', '32')
    save_gpu_speed('inception-g3-64-10', "Inception G2 - Batch 64 Daten 10%", 'inception', '10', '64')
    save_gpu_speed('inception-g3-128-10', "Inception G2 - Batch 128 Daten 10%", 'inception', '10', '128')

    save_gpu_speed('transformer-g3-32-2', "Transformer G2 - Batch 32 Daten 2%", 'transformer', '2', '32', True)
    save_gpu_speed('transformer-g3-64-2', "Transformer G2 - Batch 64 Daten 2%", 'transformer', '2', '64', True)
    save_gpu_speed('transformer-g3-128-2', "Transformer G2 - Batch 128 Daten 2%", 'transformer', '2', '128', True)

    save_gpu_speed('transformer-g3-32-5', "Transformer G2 - Batch 32 Daten 5%", 'transformer', '5', '32')
    save_gpu_speed('transformer-g3-64-5', "Transformer G2 - Batch 64 Daten 5%", 'transformer', '5', '64')
    save_gpu_speed('transformer-g3-128-5', "Transformer G2 - Batch 128 Daten 5%", 'transformer', '5', '128')


# g1 - dataset speed
# def plot_datasize_speed(title, model_name, pci_speed, gpu_speed):
#     group_names = ['2%', '5%', '10%'
#                    ]
#
#     gpu_33 = collect_values(model_name + r'-' + pci_speed + '-' + gpu_speed + '-\d+-32-1', 5)
#     gpu_66 = collect_values(model_name + r'-' + pci_speed + '-' + gpu_speed + '-\d+-64-1', 5)
#     gpu_100 = collect_values(model_name + r'-' + pci_speed + '-' + gpu_speed + '-\d+-128-1', 5)
#
#     plt.title(title)
#     plt.plot(group_names, gpu_33, label='Batch 32')
#     plt.plot(group_names, gpu_66, label='Batch 64')
#     plt.plot(group_names, gpu_100, label='Batch 128')
#
#     plt.ylabel('Skalierte Epoche (s)')
#     plt.xlabel('Datengrösse')
#     plt.legend()
#     plt.show()


# plot_datasize_speed('', 'inception', '2', '33')
# plot_datasize_speed('', 'inception', '2', '66')
# plot_datasize_speed('', 'inception', '2', '100')

def save_dataset_speed(save_name, title, model_name, pci_speed, gpu_speed, data_sizes):
    # plt.clf()
    batch_sizes = ['32', '64', '128']

    speeds = []
    for ds in data_sizes:
        columns = []
        for bs in batch_sizes:
            vals = collect_values(model_name + r'-' + pci_speed + '-' + gpu_speed + '-' + ds + '-' + bs + '-1', 5)
            columns.append(vals[0])
        speeds.append(columns)

    fig, ax = plt.subplots()
    values = [item for sub_list in speeds for item in sub_list]
    max_value = max(values)
    im = ax.imshow(speeds, cmap=LinearSegmentedColormap.from_list('gr', ["g", "r"], N=max_value))

    # Loop over data dimensions and create text annotations.
    for i in range(len(data_sizes)):
        for j in range(len(batch_sizes)):
            text = ax.text(j, i, str(int(speeds[i][j])) + " (" + str(int(100 - speeds[i][j] * 100 / max_value)) + "%)",
                           ha="center", va="center", color="w")

    ax.set_xticks(np.arange(len(batch_sizes)), labels=batch_sizes)
    ax.set_yticks(np.arange(len(data_sizes)), labels=data_sizes)
    ax.set_title(title)
    plt.ylabel('Datensatz-Grösse')
    plt.xlabel('Batch-Grösse')
    fig.tight_layout()
    # plt.show()
    plt.savefig(OUTPUT_IMG + save_name + '.'
                                         'png')

def build_dataset_speed():
    save_dataset_speed('inception-g1-1-33', 'Inception G1 - GPU 33% PCIe 1', 'inception', '1', '33', ['10', '5', '2'])
    save_dataset_speed('inception-g1-1-66', 'Inception G1 - GPU 66% PCIe 1', 'inception', '1', '66', ['10', '5', '2'])
    save_dataset_speed('inception-g1-1-100', 'Inception G1 - GPU 100% PCIe 1', 'inception', '1', '100', ['10', '5', '2'])

    save_dataset_speed('inception-g1-2-33', 'Inception G1 - GPU 33% PCIe 2', 'inception', '2', '33', ['10', '5', '2'])
    save_dataset_speed('inception-g1-2-66', 'Inception G1 - GPU 66% PCIe 2', 'inception', '2', '66', ['10', '5', '2'])
    save_dataset_speed('inception-g1-2-100', 'Inception G1 - GPU 100% PCIe 2', 'inception', '2', '100', ['10', '5', '2'])

    save_dataset_speed('transformer-g1-1-33', 'Transformer G1 - GPU 33% PCIe 1', 'transformer', '1', '33', ['5', '2'])
    save_dataset_speed('transformer-g1-1-66', 'Transformer G1 - GPU 66% PCIe 1', 'transformer', '1', '66', ['5', '2'])
    save_dataset_speed('transformer-g1-1-100', 'Transformer G1 - GPU 100% PCIe 1', 'transformer', '1', '100', ['5', '2'])

    save_dataset_speed('transformer-g1-2-33', 'Transformer G1 - GPU 33% PCIe 2', 'transformer', '2', '33', ['5', '2'])
    save_dataset_speed('transformer-g1-2-66', 'Transformer G1 - GPU 66% PCIe 2', 'transformer', '2', '66', ['5', '2'])
    save_dataset_speed('transformer-g1-2-100', 'Transformer G1 - GPU 100% PCIe 2', 'transformer', '2', '100', ['5', '2'])


build_pci_speed()
build_gpu_speed()
build_dataset_speed()
