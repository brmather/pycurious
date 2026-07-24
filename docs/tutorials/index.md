# Tutorials

Two matching sets of Jupyter notebooks work through PyCurious end to end — one for
the **Bouligand** optimiser and one for the **Tanaka** centroid method. Each set
starts from a synthetic anomaly with a known answer, recovers it with
uncertainties, and finishes by mapping Curie depth over real data.

The notebooks also ship with the package: install the `examples` extra and grab
your own copies from the [repository](https://github.com/brmather/pycurious/tree/master/Examples/Notebooks).

```{note}
The **Ex5** notebooks map Curie depth over the EMAG2 v3 magnetic anomaly and
compare against the Li *et al.* (2017) global Curie-depth model. Each downloads
~600 MB of external data, so they are **rendered here as code without executing**.
Run them locally after installing the `examples`, `mapping`, and `download`
extras; {py:mod}`pycurious.download` fetches and caches the datasets on first run.
```

## Bouligand — fitting the fractal spectrum

```{toctree}
:maxdepth: 1

bouligand/Ex1-Plot-power-spectrum
bouligand/Ex2-Compute-Curie-depth
bouligand/Ex3-Posing-the-inverse-problem
bouligand/Ex4-Spatial-variation-of-Curie-depth
bouligand/Ex5-Mapping-Curie-depth-EMAG2
```

## Tanaka — the centroid method

```{toctree}
:maxdepth: 1

tanaka/Ex1-Plot-amplitude-spectrum
tanaka/Ex2-Compute-Curie-depth
tanaka/Ex3-Parameter-exploration
tanaka/Ex4-Spatial-variation-of-Curie-depth
tanaka/Ex5-Mapping-Curie-depth-EMAG2
```
