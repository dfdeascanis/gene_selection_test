# gene selection testing

## Testing variable gene selection methods for single-cell analysis

#### Here we are testing two methods for finding significantly variable genes in single-cell RNA-seq (scRNA-seq) data. 

#### One is a homebrew python implemention of finding significant genes based on proportion of dropouts compared against a NULL negative binomial model adapted from the M3drop [package](https://www.bioconductor.org/packages/release/bioc/html/M3Drop.html)

#### The other is the more popularized implementation of analytical pearson residuals adapted from [Lause et al. (2021)](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02451-7#availability-of-data-and-materials)


#### Both of these methods aim to deconvolute the underlying biology with scRNA-seq data by finding genes that significantly deviate from two separate models. Here we highlight the potential benefits and potential limitations of each.


# References

1. Tallulah S Andrews, Martin Hemberg, M3Drop: dropout-based feature selection for scRNASeq, Bioinformatics, Volume 35, Issue 16, August 2019, Pages 2865–2867, (https://doi.org/10.1093/bioinformatics/bty1044)

2. Lause, J., Berens, P. & Kobak, D. Analytic Pearson residuals for normalization of single-cell RNA-seq UMI data. Genome Biol 22, 258 (2021). (https://doi.org/10.1186/s13059-021-02451-7)

3. Hafemeister, C., Satija, R. Normalization and variance stabilization of single-cell RNA-seq data using regularized negative binomial regression. Genome Biol 20, 296 (2019). (https://doi.org/10.1186/s13059-019-1874-1)

4. Svensson, V. Droplet scRNA-seq is not zero-inflated. Nat Biotechnol 38, 147–150 (2020). (https://doi.org/10.1038/s41587-019-0379-5)