# FakE-DiT 

> FakE-DiT is a DDPM with a Transformer backbone, trained on MNIST. 😎 

> This model wan't simple to make tbh unlike U-nets
.FakE-DiT took about 2 and 1/2 hours to train. 🕐

> The model results were poor . so i had to retrain it every time i get bad ones with better configs 🤷‍♂️

<img width="550" height="200" alt="denoised1" src="https://github.com/user-attachments/assets/1df83dd8-5b92-4589-be79-29808058b53a" />
<hr>
<img width="550" height="200" alt="denoised" src="https://github.com/user-attachments/assets/d6c809a1-73fe-4a55-8114-55e36caf93d7" />
<hr>
<img width="550" height="200" alt="denoised3" src="https://github.com/user-attachments/assets/0ada475d-355c-4ef0-9b57-212ab11bafb6" />
<hr>
<img width="550" height="200" alt="denoised2" src="https://github.com/user-attachments/assets/13758a54-66a1-4854-bdc6-ac1d7e8e10b0" />


## Requirements

```python
pip install -r requirements.txt
```
## Train
```python
python train.py
```

## Generate

```python
python test.py
```

The model starts from random noise and progressively denoises it to generate a new MNIST digit.

## Project Structure

```text
FakeE-DiT/
├── model/
│   └── fake_dit.pth
├── model.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```
