from genericpath import isdir
import pandas
import matplotlib.pyplot as plt

from os import listdir, getcwd
from os.path import isfile, join
import os

dir = os.path.dirname(os.path.abspath(__file__))

strong = [f for f in listdir(dir) if isfile(f) if f.endswith('strong_20.csv')]
weak = [f for f in listdir(dir) if isfile(f) if f.endswith('weak_20.csv')]
weak[0], weak[1] = weak[1], weak[0]

dict = {
  'strong': strong,
  'weak': weak
}

for scaling in dict:

  figScaling, axScaling = plt.subplots()
  figSpeedup, axSpeedup = plt.subplots()

  for csv in dict[scaling]:
    version = csv.split('_')[0][len('nbody'):]
    df = pandas.read_csv(csv, sep=';')

    ## Execution time
    axScaling.plot(df['NPROC'], df['TIME(s)-BY-NODE'], label=f"{version}-node")
    axScaling.plot(df['NPROC'], df['TIME(s)-BY-SLOT'], label=f"{version}-slot")
    axScaling.scatter(df['NPROC'], df['TIME(s)-BY-NODE'])
    axScaling.scatter(df['NPROC'], df['TIME(s)-BY-SLOT'])

    axScaling.legend()
    axScaling.spines['right'].set_visible(False)
    axScaling.spines['top'].set_visible(False)
    axScaling.grid(axis = 'y', color='gray', linestyle='-', zorder=0)
    axScaling.grid(axis = 'x', color='gray', linestyle='--', zorder=0)
    axScaling.set_xlabel("Number of processes")
    axScaling.set_ylabel("Execution time (s)")
    axScaling.set_xticks(df['NPROC'])
    axScaling.set_title(f'{scaling.capitalize()} scaling', pad=10)

    ### Speedup
    axSpeedup.plot(df['NPROC'], df['TIME(s)-BY-NODE'].iloc[0]/df['TIME(s)-BY-NODE'], label=f"{version}-node")
    axSpeedup.plot(df['NPROC'], df['TIME(s)-BY-NODE'].iloc[0]/df['TIME(s)-BY-SLOT'], label=f"{version}-slot")
    axSpeedup.scatter(df['NPROC'], df['TIME(s)-BY-NODE'].iloc[0]/df['TIME(s)-BY-NODE'])
    axSpeedup.scatter(df['NPROC'], df['TIME(s)-BY-NODE'].iloc[0]/df['TIME(s)-BY-SLOT'])
    axSpeedup.legend()

    axSpeedup.spines['right'].set_visible(False)
    axSpeedup.spines['top'].set_visible(False)
    axSpeedup.grid(axis = 'y', color='gray', linestyle='-', zorder=0)
    axSpeedup.grid(axis = 'x', color='gray', linestyle='--', zorder=0)
    axSpeedup.set_xlabel("Number of processes")
    axSpeedup.set_ylabel("Speedup")
    axSpeedup.set_xticks(df['NPROC'])
    axSpeedup.set_title(f'Speedup for {scaling} scaling', pad=10)

  figScaling.savefig(f'{scaling} scaling')
  figSpeedup.savefig(f'{scaling} scaling speedup')