# 🌍 Geo2Vec: Shape- and Distance-Aware Neural Representation of Geospatial Entities

> 🚀 *Building wheels for next-generation GeoAI* — a moonshot for spatial understanding.

**We've made a major update, the speed is **10 times** faster than before, when learning complex entities or large region.**

**Learning 100k entities in details need less than 1 hour now.**


---

![Geo2Vec Intuition](./pics/visio144.png)

---

## 📌 Overview

**Spatial Representation Learning** serves as the foundation for a wide range of GeoAI applications. We introduce **Geo2Vec**, a unified spatial representation learning framework for generating **general-purpose** embeddings of geospatial entities.


### 🧠 What is Geo2Vec?

Geo2Vec is a tool for generating informative representations of geospatial entities—including points, lines, multi-line, polygons, multipolygons, and polygons with holes.  
It can extract **global-level** location representations, **fine-grained** geometric representations, or **both**, and these embeddings can be seamlessly used for any downstream GeoAI task, and boost GeoAI model's performance. 

 **Signed Distance Field (SDF)–based representation** — Each entity is converted into a signed distance field. We sample points within this field as training data and train a neural network to model the SDF.


---

## ✨ Key Features

-  **Embeddings without intermediate space** — Learn spatial representation directly from the coordinate space, no feature engineering or fourier transform.  
-  **Adaptive Positional Encoding** — Capture fine detail and complex geometric patterns of geospatial entities.  
-  **Rotation-Invariant Positional Encoding** — Optional module for orientation-invariant shape representations.  
-  **Supports multiple geometry types** — Point, LineString, MultiLineString, Polygon, MultiPolygon, and Polygon with holes.


---

## 📊 Applications

- 🏢 Land-use & building classification  
- 🧭 Topology & spatial relation reasoning  
- 📦 Compact vector representations for large-scale geospatial datasets  
- 🧠 And many more GeoAI applications you can imagine!

---

## 🗺️ Datasets

Datasets used in our experiments can be found in the `data` folder.  
Additional large datasets are available on Google Drive: 
  [Dataset Link](https://drive.google.com/file/d/1lsd0pf2qwMxCL6a6tXFd7m_RxnWEs6bn/view?usp=drive_link).

## Tutorials

Tutorial is updated, you could directly use `tutorial.ipynb` to learn the representation in a more intuitive and instructional way. 

**Option 1 - use Geo2Vec in your own code.** If you want to incorporate Geo2Vec directly into your own pipeline, import the `list2vec` function from `runners/list2embedding.py` and pass it a plain Python list of your geospatial entities (any mix of points, lines, or polygons). It returns a NumPy array with one embedding row per entity:

```python
from runners.list2embedding import list2vec

geo_list = [poly1, poly2, poly3]   # your list of shapely geometries
embedding = list2vec(geo_list, Geo_dim=128)
```

**Option 2 - run `main.py` with a config file.** If you'd rather not write any code, point a config under `configs/` (e.g. `configs/main.yaml`) at your `.gpkg`/`.pkl` file and run:

```bash
python main.py --config configs/main.yaml
```

`main.py` samples, trains, and saves the learned location, shape, and combined embeddings as `.npy` files next to `save_file_name` in your config.

## 🛠️ Installation

> More setup details will be added soon!

```bash
git clone https://github.com/chuchen2017/GeoNeuralRepresentation.git
pip install -r requirements.txt
```

## Tips

1. **DO NOT** normalize the learned embedding, this will lead to loss of features. 
2. You can still add **new entities** after you already have a Geo2Vec model. Using the `save_model_path` of the list2vec function, this will keep the newly added entity embeddings in the same latent space of the previous ones. 


