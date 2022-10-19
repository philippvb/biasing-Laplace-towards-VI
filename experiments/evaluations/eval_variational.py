import sys
sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/paper")
sys.path.append("/home/hennig/pvbachmann87/VariationalLaplace")


from src.datasets import get_dset
from src.models import ClassificationDiagVB
from src.models.models import SmallCNN, VariationalSmallCNN
from src.utils.argparser import checkpoint_parse
from src.callbacks.evaluation import evaluate_statistics
from src.metrics.classification import confidence

args = checkpoint_parse()
dataset = get_dset(**args)
dataset.setup_all()
network = VariationalSmallCNN(**args)
network_normal = SmallCNN(**args)
model = ClassificationDiagVB.load_from_checkpoint("run_checkpoints/07-10/Variba_new/" + "epoch=19-step=39.ckpt", model=network, model_normal=network_normal)

pred, target = model.predict_from_dataloader(dataset.val_dataloader(), samples=1, mean=True)
stats = evaluate_statistics(pred, target, n_classes=10)
print("Stats", stats)

ood_args = dict(args)
ood_args["dset"] = "fashionmnist"
fashion_mnist = get_dset(**ood_args)
fashion_mnist.setup_all()
pred_fmnist, _ = model.predict_from_dataloader(fashion_mnist.val_dataloader(), samples=1, mean=True)
fmnist_conf = confidence(pred_fmnist.softmax(dim=-1))
print("FMNIST conf", fmnist_conf)

noise_args = dict(args)
noise_args["dset"] = "mnistnoise"
noise_dataset = get_dset(**noise_args)
noise_dataset.setup_all()
pred_noise, _ = model.predict_from_dataloader(noise_dataset.val_dataloader(), samples=1, mean=True)
noise_conf = confidence(pred_noise.softmax(dim=-1))
print("Noise conf", noise_conf)

