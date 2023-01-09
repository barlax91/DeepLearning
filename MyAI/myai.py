# Importation des modules nécessaires
import gradio as gr
from fastai.vision.all import *
from duckduckgo_search import ddg_images
from fastcore.all import *
from fastdownload import download_url
import pathlib

plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

#importation des données d'entrainement
def search_images(term, max_images=200): return L(ddg_images(term, max_results=max_images)).itemgot('image')
urls = search_images('dangerous mushrooms', max_images=1)
urls[0]

Test = 'mush.jpg'
download_url(urls[0], Test, show_progress=False)
 
im = Image.open(Test)
im.to_thumb(256,256)

download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)

searches = 'forest','dangerous mushrooms'
path = Path('dangerousmush_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)

#Some photos might not download correctly which could cause our model training to fail, so we'll remove them:
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

#To train a model, we'll need DataLoaders, which is an object that contains a training set (the images used to create a model)
#and a validation set (the images used to check the accuracy of a model -- not used during training). In fastai we can create that easily using a DataBlock, and view sample images from it
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)

# Définition de la fonction de prédiction
def predict(image):

  # Création du modèle
  learn = cnn_learner(dls, resnet18, metrics=accuracy)

  # Chargement des poids pré-entraînés
  learn.load('stage-1')

  # Prédiction de la classe du champignon à partir de l'image
  prediction = learn.predict(image)[0]
  return prediction

# Création de l'interface utilisateur avec Gradio
gr.Interface(predict, inputs=Image, output_type='class').launch()

