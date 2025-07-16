# 🧠*Geo2Vec: Shape- and Distance-Aware Neural Representation of Geospatial Entities*

The original codes of the paper. More detail will be added soon!

---

### 📌 Overview

Spatial representation learning underpins a wide range of GeoAI applications, such as land-use classification and urban analytics, by encoding not just the shapes and locations of geospatial entities (points, polylines, polygons) but also their topological and distance relationships. Recent methods like Poly2Vec leverage Fourier transforms to generate unified embeddings for different spatial object types. However, these approaches rely on uniform, non-adaptive sampling in the Fourier domain, due to the lack of direct correspondence between real-world geometry and the Fourier feature space. As a result, they often produce overly smooth representations that fail to capture fine-grained structural features such as sharp edges and boundaries.

To overcome these limitations, we introduce Geo2Vec, a novel spatial representation learning method inspired by the signed distance field (SDF). Geo2Vec adaptively samples real-world points, encoding each by its signed distance to the target entity (positive outside, negative inside), thus grounding the embedding in geometric proximity and containment. We train a neural network to approximate the SDF of each spatial object, effectively learning a compact, geometry-aware representation. To enhance expressiveness, we also propose a rotation-invariant positional encoding scheme for the sampled points, allowing the model to capture high-frequency spatial variations and promoting more interpretable embeddings for downstream GeoAI tasks.

Empirical results show that Geo2Vec outperforms existing methods in spatial reasoning tasks, topological and distance relationship inference, and significantly improves performance on a range of GeoAI applications.

---

![Intuition](./pics/visio144.png)

