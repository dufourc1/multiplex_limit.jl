# Code for "Inference for decorated graphs and application to multiplex networks" (Dufour and Olhede, 2024)

The images in the paper were done with `Julia  1.10.1` and the manifest file is provided.

To run the data analysis experiments, you need to download the data from the following links, and put it in a `data` folder in the root directory of the project:
- Multiplex network of human diseases [https://github.com/manlius/MultiplexDiseasome/tree/master](https://github.com/manlius/MultiplexDiseasome/tree/master)
- Sociopattern high school contact dataset [https://networks.skewed.de/net/sp_high_school](https://networks.skewed.de/net/sp_high_school)


(Please note that a recent change in `NetworkHistogram.jl` was made in the computation of the initial bandwidth. To 
get exactly the results from the paper, use `> add NetworkHistogram#legacy_bandwidth` when installing the package; using the `Manifest.toml` file will install the correct version.)
