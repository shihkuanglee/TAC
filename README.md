## Introduction

#### This repository provides the software implementation in calculating the temporal autocorrelation of speech as the speech feature for replayed speech detection system, it is used in the study:

Shih-Kuang Lee, Yu Tsao, and Hsin-Min Wang, “[Detecting Replay Attacks Using Single-Channel Audio: The Temporal Autocorrelation of Speech](https://homepage.iis.sinica.edu.tw/papers/whm/25385-F.pdf),” in 2022 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) (APSIPA ASC 2022), Chiang Mai, Thailand, Nov. 2022.

## Dependencies
```
pip install -r requirements.txt
```

## Usage

Add submodule to your repository:
```
git submodule add git@github.com:shihkuanglee/TAC.git
```

Then the library is now able to be import:
```
from TAC.calc_g import stft, tac_v8
```

stft(configuration used in the study):
```
Obs = stft(data, size=1024, shift=256)[np.newaxis,:,:]
```

tac_v8(configuration used in the study):
```
G_block = tac_v8(
          Obs.transpose(2, 0, 1),
          taps=16,
          delay=2,
          iterations=3,
          psd_context=0)
```

## Citation Information

Shih-Kuang Lee, Yu Tsao, and Hsin-Min Wang, “Detecting Replay Attacks Using Single-Channel Audio: The Temporal Autocorrelation of Speech,” in 2022 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) (APSIPA ASC 2022), Chiang Mai, Thailand, Nov. 2022.
```bibtex
@INPROCEEDINGS{Lee2211:Detecting,
  AUTHOR={Shih-Kuang Lee and Yu Tsao and Hsin-Min Wang},
  TITLE={{Detecting Replay Attacks Using Single-Channel Audio: The Temporal Autocorrelation of Speech}},
  BOOKTITLE={2022 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) (APSIPA ASC 2022)},
  ADDRESS={Chiang Mai, Thailand},
  MONTH={nov},
  YEAR={2022}}
```

## Licensing

This repository is licensed under the [MIT License](https://github.com/shihkuanglee/TAC/blob/main/LICENSE).

This repository includes modified codes from [nara_wpe](https://github.com/fgnt/nara_wpe), also MIT licensed.
