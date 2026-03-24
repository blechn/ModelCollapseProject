from argparse import ArgumentParser
import os
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader

import lightning as L

from typing import Tuple, Type

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data.mnist import tds_to_dl, get_mnist_loaders
from src.metrics.prediction_accuracy import compute_metrics as compute_acc_cm
from src.metrics.entropy_metrics import compute_entropy_metrics
from src.models.classification.cnn import get_trained as get_c
from src.models.generative.realnvp import RealNVP
from src.models.generative.flow_matching import FlowMatching


def full_experiment(model_cls: Type[L.LightningModule], **kwargs):
    # create baseline
    # get original mnist data
    trl, tel = get_mnist_loaders()

    # compute metrics for the original data
    eval_trainer = L.Trainer()
    cnn = get_c(**kwargs)
    predictions = eval_trainer.predict(cnn, dataloaders=tel)

    pred_probs, pred_labels, pred_features = zip(*predictions)
    pred_probs = torch.cat(pred_probs)
    pred_labels = torch.cat(pred_labels)

    acc, cm = compute_acc_cm(tel.dataset.tensors[1].argmax(dim=1).cpu(), pred_probs.cpu())
    conf_entropy, div_entropy = compute_entropy_metrics(tel.dataset.tensors[1].argmax(dim=1).cpu(), pred_probs.cpu())

    current_data = (trl, tel)

    print(f"Baseline accuracy: {acc}")
    print(f"Baseline entropies:\t{conf_entropy:.3f}\t{div_entropy:.3f}")

    # collapse epoch loop
    results_dict = {
        'accuracies': [],
        'cms': [],
        'conf_entropies': [],
        'div_entropies': [],
    }


    for i in range(kwargs.get('collapse_epochs', 10)):
        trainer = L.Trainer(max_epochs=kwargs.get('max_epochs', 10))
        model = model_cls()
        acc, cm, conf_entropy, div_entropy, current_data = experiment_step(exp_type=kwargs.get('experiment_type', 'full'), trainer=trainer, model=model, current_data=current_data, epoch_idx=i, **kwargs)
        results_dict['accuracies'].append(acc)
        results_dict["cms"].append(cm)
        results_dict["conf_entropies"].append(conf_entropy)
        results_dict["div_entropies"].append(div_entropy)

        print(f"Accuracy\t{i}: {acc}")
        print(f"Confidence entropy: {conf_entropy:.3f}\tDiversity entropy: {div_entropy:.3f}")

    # save results
    save_experiment_results(results_dict=results_dict, model_name=model_cls.__name__)

def save_experiment_results(results_dict, model_name, output_dir="results"):
    out_path = Path(output_dir) / model_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    epochs = list(range(len(results_dict['accuracies'])))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Collapse Epoch')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(epochs, results_dict['accuracies'], color='tab:blue', marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx() # on the right axis entropy
    ax2.set_ylabel('Entropy', color='tab:red')
    ax2.plot(epochs, results_dict['conf_entropies'], color='tab:red', linestyle='--', marker='s', label='Conf. Entropy')
    ax2.plot(epochs, results_dict['div_entropies'], color='tab:green', linestyle=':', marker='^', label='Div. Entropy')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f"Experiment Metrics Evolution: {model_name}")
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    plt.savefig(out_path / "metrics_evolution.png")
    print(f"Saved metrics plot to {out_path}/metrics_evolution.png")

    # 3. Save Individual Confusion Matrices
    for i, cm in enumerate(results_dict['cms']):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - Epoch {i} ({model_name})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(out_path / f"cm_epoch_{i}.png")
        plt.close()

    # 4. Save Raw Data to CSV
    df = pd.DataFrame(results_dict)
    df.to_csv(out_path / "results_summary.csv", index_label="epoch")
    print(f"Saved numerical results to {out_path}/results_summary.csv")
    return out_path

def save_sample_grid(samples, labels, output_path, title="Generated Samples"):
    """
    Saves a 10x5 grid of samples (5 per class).
    Args:
        samples (Tensor): [N, 1, 28, 28]
        labels (Tensor): [N] or [N, 10] (integers or one-hot)
        output_path (Path): Where to save the image.
    """
    # Convert one-hot to integer if necessary
    if labels.dim() > 1:
        labels = labels.argmax(dim=1)
        
    fig, axes = plt.subplots(10, 5, figsize=(10, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for digit in range(10):
        # Find indices where label matches the current digit
        indices = (labels == digit).nonzero(as_tuple=True)[0]
        
        # Take the first 5 found
        selected_indices = indices[:5]
        
        for i in range(5):
            ax = axes[digit, i]
            if i < len(selected_indices):
                img = samples[selected_indices[i]].squeeze().cpu().numpy()
                ax.imshow(img, cmap='gray')
            
            ax.axis('off')
            if i == 0:
                ax.set_title(f"Digit {digit}", loc='left', fontsize=10)

    plt.suptitle(title, fontsize=16)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def experiment_step(exp_type, trainer: L.Trainer, model: L.LightningModule, current_data: Tuple[DataLoader, DataLoader], **kwargs):
    # train new model on the current data
    trainer.fit(model=model, train_dataloaders=current_data[0], val_dataloaders=current_data[1])

    # generate new data from trained model
    # for next training and eval here
    if exp_type == 'full':
        generated_data_x, generated_data_y = model.sample(6000, **kwargs) # for full experiment we need to replace the full 60000 images
        gds = TensorDataset(generated_data_x, generated_data_y)
        gl = tds_to_dl(gds)
    elif exp_type == 'replace':
        gen_x, gen_y = model.sample(int(6000 * kwargs.get('replace-percentage', 0.2)))
        current_data_x = current_data[0].dataset.tensors[0]
        current_data_y = current_data[0].dataset.tensors[1]
        indices = torch.randperm(current_data_x.shape[0], device=current_data_x.device)
        current_data_x[indices] = gen_x.to(device=current_data_x.device)
        current_data_y[indices] = gen_y.to(device=current_data_y.device)
        gds = TensorDataset(current_data_x, current_data_y)
        gl = tds_to_dl(gds)

    elif exp_type == 'add':
        gen_x, gen_y = model.sample(int(6000 * kwargs.get('add-percentage', 0.2)))
        gds = TensorDataset(gen_x, gen_y)
        cds = ConcatDataset([current_data[0].dataset, gds])
        gl = DataLoader(cds, batch_size=64, shuffle=True)

        
    generated_test_x, generated_test_y = model.sample(1000, **kwargs) # test set also same size as original data
    #gds = TensorDataset(generated_data_x, generated_data_y)
    #gl = tds_to_dl(gds)
    gts = TensorDataset(generated_test_x, generated_test_y)
    gtl = tds_to_dl(gts)

    grid_path = Path("results") / Path(model.__class__.__name__) / f"samples_epoch_{kwargs.get('epoch_idx')}.png"
    save_sample_grid(
        samples=generated_data_x,
        labels=generated_data_y,
        output_path=grid_path,
        title=f"Generated Data - Model {model.__class__.__name__} Collapse Epoch {kwargs.get('epoch_idx')}"
    )

    # metrics on generated data
    eval_trainer = L.Trainer(logger=False)
    cnn = get_c(**kwargs)
    predictions = eval_trainer.predict(cnn, dataloaders=gtl)
    pred_probs, pred_labels, _ = zip(*predictions)
    pred_probs = torch.cat(pred_probs)
    pred_labels = torch.cat(pred_labels)

    acc, cm = compute_acc_cm(generated_test_y.argmax(dim=1).cpu(), pred_probs.cpu())
    conf_entropy, div_entropy = compute_entropy_metrics(generated_test_y.argmax(dim=1).cpu(), pred_probs.cpu())
    current_data = (gl, gtl)
    return acc, cm, conf_entropy, div_entropy, current_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--modelcls", type=str, default="realnvp", choices=["realnvp", "flowmatching"], help="Which generative model to use for experiments.")
    parser.add_argument("--max_epochs", action="store", type=int, default=1, help="How many epochs will each model train.")
    parser.add_argument("--collapse_epochs", action="store", type=int, default=2, help="How many collapse epochs the experiment will run for.")
    parser.add_argument("--ode_steps", action="store", type=int, default=5, help="How many ODE steps to use for Flow matching sampling.")
    parser.add_argument("--fashion", action="store_true", default=False, help="Use the FashionMNIST dataset instead of the original MNIST.")
    parser.add_argument("--experiment", action="store", default='full', help="Which experiment to run.", choices=['full', 'replace', 'add'])
    args = parser.parse_args()
    args_dict = vars(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    if args.modelcls == "realnvp":
        model_cls = RealNVP
    else:
        model_cls = FlowMatching

    full_experiment(model_cls=model_cls, device=device, **args_dict)