import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymcd.mcd import Calculate_MCD
from tqdm import tqdm


def calculate_mcd_for_wav(wav, target):
    if not os.path.exists(target):
        print("not exist", target)
        return 0

    try:
        _mcd = mcd_toolbox.calculate_mcd(target, wav)
    except Exception as e:
        print(f"Error in {target, wav}, {e}")
        return 0

    # if _mcd > 12:
    #     print(wav, target)

    return _mcd


import sys

test_lst = sys.argv[1]
output_path = sys.argv[2]
mode = sys.argv[3]


#mode = "dtw"  # dtw_sl
mcd_toolbox = Calculate_MCD(MCD_mode=mode)

with open(test_lst, "r") as fr:
    lines = fr.readlines()

path = output_path

gen_wavs = [path + "gen/" + str(idx).zfill(8) + ".wav" for idx, line in enumerate(lines)]
targets = [path + "tgt/" + str(idx).zfill(8) + ".wav" for idx, line in enumerate(lines)]

mcd = 0
nums = 0
mcd_values = []
with ProcessPoolExecutor(max_workers=64) as executor:
    results = list(tqdm(executor.map(calculate_mcd_for_wav, gen_wavs, targets), total=len(gen_wavs)))

mcd_values = [it for it in results if it != 0]
mcd_avg = np.mean(mcd_values)

if mode == "dtw":
    print(f"Average MCD: {mcd_avg:.3f}")
if mode == "dtw_sl":
    print(f"Average MCD_SL: {mcd_avg:.3f}")

