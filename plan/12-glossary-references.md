# Glossary and References

## 1. Project terms

**Baseline**
The simplest reasonable comparison. For classification, always predicting the most common class. For retrieval, a random rotation.

**CLIP**
A jointly trained image–text model. Strong zero-shot baseline, but it requires multimodal training.

**Cosine similarity**
The cosine of the angle between two vectors. Used for retrieval after L2 normalization.

**DINOv2**
A self-supervised vision encoder. Produces strong general-purpose image features without labels.

**Embedding**
A dense vector representation of an input. Similar inputs should have similar vectors.

**EuroSAT**
A dataset of 27,000 labeled 64×64 Sentinel-2 tiles in 10 land-use classes.

**Frozen**
The model’s weights are not updated. Used here to test whether alignment exists without training.

**Macro-F1**
The average F1 across classes, giving each class equal weight. Useful when class balance is uneven.

**MiniLM**
A small sentence-transformer text encoder. `all-MiniLM-L6-v2` produces 384-dimensional embeddings.

**Out-of-distribution (OOD)**
Data that differs from the training distribution. Upper Arlington tiles are the OOD test here.

**Platonic Representation Hypothesis (PRH)**
The hypothesis that independently trained models converge toward a shared representation of reality, so their embeddings become increasingly aligned.

**Procrustes**
A method for finding the orthogonal transformation that best aligns one set of points to another. Closed form via SVD.

**Retrieval**
Finding the nearest items in an embedding space. Here: text query → satellite tiles.

**Semi-orthogonal**
A rectangular matrix \(W\) with \(W^T W = I\). Used when embedding dimensions differ.

**Sentinel-2**
A European Space Agency satellite mission with 10–60 m resolution and frequent revisit.

**Top-k accuracy**
The share of queries where the correct answer is among the top k results.

**Zero-shot**
Using a model on classes or tasks it was not explicitly trained on.

## 2. Upper Arlington terms

**Upper Arlington**
A city in Franklin County, Ohio, northwest of Columbus. Area 9.83 sq mi; population about 36,800 (2020).

**City center**
40.02056° N, 83.06639° W.

**Boundaries**
West: Scioto River. North/east: Columbus. South: Marble Cliff and Grandview Heights.

**Tree canopy**
About 34.2% of land area, based on a 2011–2021 canopy assessment. About 33% of land is considered possible additional canopy.

**Tree City USA**
A designation from the Arbor Day Foundation; Upper Arlington has held it since 1990.

**Urban Forestry Plan**
A city planning effort underway in 2026 to guide tree canopy and natural asset decisions.

**Sustainability agenda**
Upper Arlington’s framework with goals including zero waste, zero pollution, and zero emissions by 2050.

**Copernicus Browser**
A free web tool for browsing and downloading Sentinel-2 imagery.

## 3. Core references

| Topic | Reference | Link |
|---|---|---|
| PRH paper | Huh et al., “The Platonic Representation Hypothesis” | https://arxiv.org/abs/2405.07987 |
| DINOv2 | Oquab et al. | https://arxiv.org/abs/2304.07193 |
| MiniLM | Sentence Transformers model card | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| EuroSAT | Helber et al. | https://arxiv.org/abs/1709.00029 |
| CLIP | Radford et al. | https://arxiv.org/abs/2103.00020 |
| Procrustes | SciPy docs | https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html |
| Gradio | Demo framework | https://www.gradio.app |
| Copernicus Browser | Sentinel-2 imagery | https://browser.dataspace.copernicus.eu |
| Sentinel-2 | ESA mission page | https://sentinel.esa.int/web/sentinel/missions/sentinel-2 |
| Upper Arlington | City overview | https://en.wikipedia.org/wiki/Upper_Arlington,_Ohio |
| Upper Arlington sustainability | City page | https://www.upperarlingtonoh.gov/506/Sustainability-in-UA |
| Upper Arlington parks/forestry | City page | https://www.upperarlingtonoh.gov/263/Parks-Forestry |
| Upper Arlington GIS | City page | https://www.upperarlingtonoh.gov/288/GIS-Maps-Data |
| Tree canopy summary | 2011–2021 report | https://columbusrecparks.com/wp-content/uploads/2024/12/Upper-Arlington-Ohio-2011-2021-Tree-Canopy-Summary.pdf |
| Google Colab | Compute | https://colab.research.google.com |
| Hugging Face | Models/datasets | https://huggingface.co |

## 4. Citation table for the write-up

| Used | What for | Link |
|---|---|---|
| EuroSAT | training tiles | |
| Sentinel-2 / Copernicus Browser | Upper Arlington tiles | |
| DINOv2 | image encoder | |
| all-MiniLM-L6-v2 | text encoder | |
| PyTorch | deep learning | |
| torchvision | EuroSAT loader, ResNet-18 | |
| Hugging Face Transformers | model loading | |
| Sentence Transformers | MiniLM | |
| scikit-learn | metrics | |
| SciPy | Procrustes | |
| Gradio | demo | |
| Google Colab | compute | |
| Upper Arlington canopy/GIS sources | local context | |
| AI coding assistant | [name] | |
| Tutorials used | [links] | |

## 5. Key formulas

### Orthogonal Procrustes

\[
W = U V^T \quad \text{where} \quad X^T Y = U S V^T
\]

### Transform

\[
z = (x - \mu_x) W
\]

### Cosine similarity

\[
\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}
\]

### Top-k retrieval

Return the k items with the highest similarity to the query.

## 6. One-sentence explanations for judges

- **PRH:** “Independently trained models may converge on a shared representation of reality.”
- **Procrustes:** “A closed-form rotation that aligns one embedding space to another.”
- **DINOv2:** “A frozen vision model that turns each satellite tile into a vector.”
- **MiniLM:** “A frozen text model that turns each description into a vector.”
- **EuroSAT:** “27,000 labeled Sentinel-2 tiles across 10 land-use classes.”
- **Top-1 retrieval:** “The correct class was the nearest result.”
- **Random rotation:** “A null model that tells us whether the alignment is meaningful.”
- **Upper Arlington test:** “50 local tiles, never used for fitting, to see whether the alignment transfers.”
