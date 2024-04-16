import time
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from KDEpy import FFTKDE
from scipy import interpolate
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from pyglmgampoi import fit_glmgp_offset
from pyglmgampoi import bw_SJr
from pyglmgampoi import ksmooth

from sklearn.utils.sparsefuncs import mean_variance_axis
from statsmodels.nonparametric.kernel_regression import KernelReg
from patsy import dmatrix
import scipy.sparse

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)

def make_cell_attr(count_data, cell_names):
    assert count_data.shape[0] == len(cell_names)
    total_cell = np.squeeze(np.asarray(count_data.sum(1)))
    log10_umi = np.log10(total_cell)
    genes_per_cell = np.squeeze(np.asarray((count_data > 0).sum(1)))
    log10_genes_per_cell = np.log10(genes_per_cell)
    cell_attr = pd.DataFrame({"umi": total_cell, "log10_umi": log10_umi})
    cell_attr.index = cell_names
    cell_attr["n_expressed_genes"] = genes_per_cell
    # this is referrred to as gene in SCTransform
    cell_attr["log10_gene"] = log10_genes_per_cell
    cell_attr["umi_per_gene"] = log10_umi / genes_per_cell
    cell_attr["log10_umi_per_gene"] = np.log10(cell_attr["umi_per_gene"])
    return cell_attr

def gmean_array(count_data, gmean_eps=1):
    gmean = np.exp(np.log(count_data + gmean_eps).mean(0)) - gmean_eps
    return gmean

def gmean_sparse(count_data, gmean_eps=1):

    gmean = np.array(gmean_array(count_data.todense(), gmean_eps))
    gmean = np.squeeze(gmean)
    return gmean

def dds(genes_log10_gmean_step1, grid_points=2**10):
    # density dependent downsampling
    # print(genes_log10_gmean_step1.shape)
    # if genes_log10_gmean_step1.ndim <2:
    #    genes_log10_gmean_step1 = genes_log10_gmean_step1[:, np.newaxis]
    x, y = (
        FFTKDE(kernel="gaussian", bw="silverman")
        .fit(np.asarray(genes_log10_gmean_step1))
        .evaluate(grid_points=grid_points)
    )
    density = interpolate.interp1d(x=x, y=y, assume_sorted=False)
    sampling_prob = 1 / (density(genes_log10_gmean_step1) + np.finfo(float).eps)

    # sampling_prob = 1 / (density + np.finfo(float).eps)
    return sampling_prob / sampling_prob.sum()

def get_model_params_pergene_glmgp_offset(gene_umi, coldata, log_umi, design="~ 1"):
    if not scipy.sparse.issparse(gene_umi):
        pass
    else:
        gene_umi = gene_umi.todense()
    params = fit_glmgp_offset(y=gene_umi, coldata=coldata, design=design, log_umi=log_umi)

    return params
def get_model_params_allgene_glmgp(count_data, coldata, use_offset=False):

    results = []
    log_umi = np.log(np.ravel(count_data.sum(1)))
    if use_offset:
        results = get_model_params_pergene_glmgp_offset(count_data.T, coldata, log_umi)
    else:
        results = get_model_params_pergene_glmgp(count_data.T, coldata)
    
    params_df = pd.DataFrame(results)

    return params_df

def sparse_var(X, axis=None):
    mean, var = mean_variance_axis(X, axis)
    return var

def dense_var(X, axis=None):
    var = np.var(X, axis)
    return var

def bwSJ(genes_log10_gmean_step1, bw_adjust=3):
    # See https://kdepy.readthedocs.io/en/latest/bandwidth.html
    fit = FFTKDE(kernel="gaussian", bw="ISJ").fit(np.asarray(genes_log10_gmean_step1))
    _ = fit.evaluate()
    bw = fit.bw * bw_adjust
    return np.array([bw], dtype=float)

def robust_scale_binned(y, x, breaks):
    def robust_scale(x):
        return (x - np.median(x)) / (stats.median_abs_deviation(x) + np.finfo(float).eps)

    bins = pd.cut(x=x, bins=breaks, ordered=True)

    # categories = bins.categories
    # bins = np.digitize(x=x, bins=breaks)
    df = pd.DataFrame({"x": y, "bins": bins})
    tmp = df.groupby("bins").apply(robust_scale)
    order = df["bins"].argsort()
    tmp = tmp.loc[order]  # sort_values(by=["bins"])
    score = tmp["x"]
    return score

def is_outlier(y, x, th=10):
    bin_width = (np.nanmax(x) - np.nanmin(x)) * bwSJ(x, bw_adjust=1 / 2)
    eps = np.finfo(float).eps * 10
    bin_width = bin_width[0]
    breaks1 = np.arange(start=np.nanmin(x) - eps, stop=np.nanmax(x) + bin_width, step=bin_width)

    breaks2 = np.arange(
        start=np.nanmin(x) - eps - bin_width / 2,
        stop=np.nanmax(x) + bin_width,
        step=bin_width)
    score1 = robust_scale_binned(y, x, breaks1)
    score2 = robust_scale_binned(y, x, breaks2)
    return np.vstack((np.abs(score1), np.abs(score2))).min(0) > th

    # categories = bins.categories
    # bins = npy.digitize(x=x, bins=breaks)
    df = pd.DataFrame({"x": y, "bins": bins})
    tmp = df.groupby("bins").apply(robust_scale)
    order = df["bins"].argsort()
    tmp = tmp.loc[order]  # sort_values(by=["bins"])
    score = tmp["x"]
    return score

def get_regularized_params(
    model_parameters,
    genes,
    genes_step1,
    genes_log10_gmean_step1,
    genes_log10_gmean,
    cell_attr,
    count_data,
    batch_var=None,
    bw_adjust=3,
    gmean_eps=1,
    theta_regularization="od_factor",
    exclude_poisson=False,
    poisson_genes=None,
    useR=False,
):
    model_parameters = model_parameters.copy()

    model_parameters_fit = pd.DataFrame(
        np.nan, index=genes, columns=model_parameters.columns)

    x_points_df = pd.DataFrame({"gene_log10_gmean": genes_log10_gmean})
    x_points_df["min_gene_log10_gmean_step1"] = genes_log10_gmean_step1.min()

    x_points_df["x_points"] = np.nanmax(x_points_df, axis=1)
    x_points_df["max_gene_log10_gmean_step1"] = np.nanmax(genes_log10_gmean_step1)
    x_points_df["x_points"] = x_points_df[["x_points", "max_gene_log10_gmean_step1"]].min(1)
    x_points = x_points_df["x_points"].values
    for column in model_parameters.columns:
        if column == "theta":
            continue
        endog = model_parameters.loc[genes_step1, column].values
        exog_fit = genes_log10_gmean_step1  # .values
        if useR:
            bw = bw_SJr(genes_log10_gmean_step1, bw_adjust=bw_adjust)  # .values)
            params = ksmooth(genes_log10_gmean, genes_log10_gmean_step1, endog, bw[0])
            index = model_parameters_fit.index.values[np.asarray(params["order"]) - 1]
            model_parameters_fit.loc[index, column] = params["smoothed"]
        else:
            bw = bwSJ(genes_log10_gmean_step1, bw_adjust=bw_adjust)  # .values)
            reg = KernelReg(
                endog=endog, exog=exog_fit, var_type="c", reg_type="ll", bw=bw)
            fit = reg.fit(x_points)
            model_parameters_fit[column] = np.squeeze(fit[0])
        # print(bw)
    theta = np.power(10, genes_log10_gmean) / (np.power(10, model_parameters_fit["od_factor"]) - 1)

    model_parameters_fit["theta"] = theta

    if exclude_poisson:
        # relace theta by inf
        if poisson_genes is not None:
            print("len poisson genes", len(poisson_genes))
            model_parameters_fit.loc[poisson_genes, "theta"] = np.inf
            model_parameters_fit.loc[poisson_genes, "od_factor"] = 0

            model_parameters_fit.loc[poisson_genes, "log10_umi"] = np.log(10)
            gene_mean = pd.Series(np.ravel(count_data.mean(0)), index=genes)
            mean_cell_sum = np.mean(np.ravel(count_data.sum(1)))
            model_parameters_fit.loc[poisson_genes, "Intercept"] = np.log(gene_mean[poisson_genes]) - np.log(mean_cell_sum)

    return model_parameters_fit

def get_residuals(count_data, model_matrix, model_parameters_fit):
    
    def pearson_residual(y, mu, theta, min_var=-np.inf):
        variance = mu + np.divide(mu**2, theta.reshape(-1, 1).T)
        variance[variance < min_var] = min_var
        pearson_residuals = np.divide(y - mu, np.sqrt(variance))
        return pearson_residuals
    
    subset = np.asarray(model_parameters_fit[model_matrix.design_info.column_names].values)
    theta = np.asarray(model_parameters_fit["theta"].values)
    mu = np.exp(np.dot(subset, model_matrix.T)).T

    residuals = pearson_residual(count_data, mu, theta)

    res_clip_range = np.sqrt(count_data.shape[0])

    residuals = np.clip(residuals, a_min=-res_clip_range, a_max=res_clip_range)

    return residuals

def correct(residuals, cell_attr, latent_var, model_parameters_fit, umi):
    # replace value of latent variables with its median
    cell_attr = cell_attr.copy()
    for column in latent_var:
        cell_attr.loc[:, column] = cell_attr.loc[:, column].median()
    model_matrix = dmatrix(" + ".join(latent_var), cell_attr)
    non_theta_columns = [x for x in model_matrix.design_info.column_names if x != "theta"]
    coefficients = model_parameters_fit[non_theta_columns]
    theta = model_parameters_fit["theta"].values

    mu = np.exp(np.dot(coefficients.values, model_matrix.T)).T
    variance = mu + np.divide(mu**2, np.tile(theta.reshape(-1, 1), mu.shape[0]).T)
    corrected_data = mu + residuals.T.values * np.sqrt(variance)
    corrected_data[corrected_data < 0] = 0
    corrected_counts = csr_matrix(corrected_data.astype(int))

    return corrected_counts

def vst(
    count_data,
    gene_names=None,
    cell_names=None,
    n_cells=5000,
    latent_var=["log10_umi"],
    gmean_eps=1,
    min_cells=5,
    n_genes=2000,
    theta_given=10,
    correct_counts=False,
    exclude_poisson=True,
    fix_slope=True,
    verbosity=0,
    useR=False
):
    """Perform variance stabilizing transformation.

    Residuals are currently stored for all genes (might be memory intensive for larger datasets).

    Parameters
    ----------
    count_data: matrix
                 Sparse or dense matrix with genes as rows and cells as columns
                 (same as Seurat)
    gene_names: list
                List of gene names for count_data
    cell_names: list
                List of cell names for count_data
    n_cells: int
             Number of cells to use for estimating parameters in Step1: default is 5000
    n_genes: int
             Number of genes to use for estimating parameters in Step1; default is 2000
    threads: int
             Number of threads to use (caveat: higher threads require higher memory)
    method: string
            "theta_ml" or "offset": method to use for estimating theta; default is "theta_ml"
    theta_given: int
                 Used only when method == "offset", for fixing the value of inverse overdispersion parameter
                 following Lause et al. (2021) offset model; default is 10
    theta_regularization: string
                         "od_factor" or "theta": parameter to run smoothing operation on for theta,
                         od_factor = 1 +mu/theta; default is od_factor

    residual_type: string
                  "pearson" or "deviance" residuals; default is "pearson"
    correct_counts: bool
                    Whether to correct counts by reversing the GLM with median values
    exclude_poisson: bool
                     To exclude poisson genes from regularization and set final parameters based on offset model; default is False
    fix_slope: bool
               Whether to fix the slope; default is False
    verbosity: bool
               Print verbose messages
    """
    count_data = count_data.copy()

    if n_cells is None:
        n_cells = count_data.shape[0]
    if n_genes is None:
        n_genes = count_data.shape[1]
    n_cells = min(n_cells, count_data.shape[0])
    n_genes = min(n_genes, count_data.shape[1])

    if gene_names is None:
        if not isinstance(count_data, pd.DataFrame):
            raise RuntimeError(
                "`gene_names` and `cell_names` are required when count_data is not a dataframe"
            )
        else:
            gene_names = count_data.index.tolist()
            cell_names = count_data.columns.tolist()
            count_data = csr_matrix(count_data.values)

    if cell_names is None:
        cell_names = [x for x in range(count_data.shape[0])]
    gene_names = np.asarray(gene_names, dtype="U")
    cell_names = np.asarray(cell_names, dtype="U")
    genes_cell_count = np.asarray((count_data >= 0.01).sum(0))
    min_cells_genes_index = np.squeeze(genes_cell_count >= min_cells)
    genes = gene_names[min_cells_genes_index]
    print(f'Using {len(genes)} genes')

    cell_attr = make_cell_attr(count_data, cell_names)

    if isinstance(count_data, pd.DataFrame):
        count_data = count_data.loc[genes]
    else:
        count_data = count_data[:, min_cells_genes_index]
    
    if scipy.sparse.issparse(count_data):
        genes_log10_gmean = np.log10(gmean_sparse(count_data)) 
    else:
        genes_log10_gmean = np.log10(gmean_array(count_data)) 
    
    genes_log10_amean = np.log10(np.ravel(count_data.mean(0)))

    if n_cells is not None and n_cells < count_data.shape[0]:
        # downsample cells to speed up the first step
        cells_step1_index = np.random.choice(a=np.argsort(cell_names), size=n_cells, replace=False)
        cells_step1 = cell_names[cells_step1_index]
        genes_cell_count_step1 = np.squeeze(np.array((count_data[cells_step1_index, :] > 0).sum(0)))
        genes_bool = genes_cell_count_step1 >= min_cells
        genes_step1 = genes[genes_bool]
        
        if scipy.sparse.issparse(count_data):
            genes_log10_gmean_step1 = np.log10(gmean_sparse(count_data[:, genes_bool]))
        else:
            genes_log10_gmean_step1 = np.log10(gmean_array(count_data[:, genes_bool]))
        
        genes_log10_amean_step1 = np.log10(np.ravel(count_data[:, np.argsort(genes_step1)].mean(0)))
        umi_step1 = count_data[cells_step1_index, :]
        umi_step1 = umi_step1[:, genes_bool]

    else:
        cells_step1_index = np.arange(len(cell_names), dtype=int)
        cells_step1 = cell_names
        genes_step1 = genes
        genes_log10_gmean_step1 = genes_log10_gmean
        genes_log10_amean_step1 = genes_log10_amean
        umi_step1 = count_data

    data_step1 = cell_attr.loc[cells_step1]

    if (n_genes is not None) and (n_genes < len(genes_step1)):
        # density-sample genes to speed up the first step
        sampling_prob = dds(genes_log10_gmean_step1)
        genes_step1_index = np.random.choice(a=np.arange(len(genes_step1)), size=n_genes, replace=False, p=sampling_prob)
        genes_step1 = gene_names[genes_step1_index]
        umi_step1 = umi_step1[:, genes_step1_index]
        
        if scipy.sparse.issparse(count_data):
            genes_log10_gmean_step1 = np.log10(gmean_sparse(umi_step1, gmean_eps=gmean_eps))
        else:
            genes_log10_gmean_step1 = np.log10(gmean_array(umi_step1, gmean_eps=gmean_eps))
        
        genes_log10_amean_step1 = np.log10(umi_step1.mean(0))
    
        # Step 1: Estimate theta

    if verbosity:
        print("Running Step1")
    start = time.time()

    model_parameters = get_model_params_allgene_glmgp(umi_step1, data_step1, use_offset=True)
    model_parameters.index = genes_step1

    # useR = False

    gene_attr = pd.DataFrame(index=genes)
    gene_attr["gene_amean"] = np.power(10, genes_log10_amean)
    gene_attr["gene_gmean"] = np.power(10, genes_log10_gmean)
    gene_attr["gene_detectation_rate"] = (np.squeeze(np.asarray((count_data > 0).sum(0))) / count_data.shape[1])
    gene_attr["theta"] = model_parameters["theta"]
    if scipy.sparse.issparse(count_data):
        gene_attr["gene_variance"] = sparse_var(count_data, 0)
    else:
        gene_attr["gene_variance"] = dense_var(count_data, 0)
    poisson_genes = None

    if exclude_poisson:
        poisson_genes1 = gene_attr.loc[gene_attr["gene_amean"] >= gene_attr["gene_variance"]].index.tolist()
        poisson_genes2 = gene_attr.loc[gene_attr["gene_amean"] <= 1e-3].index.tolist()
        poisson_genes = set(poisson_genes1).union(poisson_genes2)

        poisson_genes_step1 = set(poisson_genes).intersection(genes_step1)

        if verbosity:
            print("Found ", len(poisson_genes1), " genes with var <= mean")
            print("Found ", len(poisson_genes2), " genes with mean < 1e-3")
            print("Found ", len(poisson_genes), " poisson genes")
            print("Setting there estimates to Inf")
        
        if poisson_genes_step1:
            model_parameters.loc[poisson_genes_step1, "theta"] = np.inf

    end = time.time()

    step1_time = np.ceil(end - start)
    if verbosity:
        print("Step1 done. Took {} seconds.".format(np.ceil(end - start)))

    # Step 2: Do regularization

    if verbosity:
        print("Running Step2")
    start = time.time()

    genes_log10_gmean_step1_to_return = genes_log10_gmean_step1.copy()
    genes_log10_amean_step1_to_return = genes_log10_amean_step1.copy()
    outliers_df = pd.DataFrame(index=genes_step1)

    for col in model_parameters.columns:
        col_outliers = is_outlier(model_parameters[col].values, genes_log10_gmean_step1)
        outliers_df[col] = col_outliers

    if exclude_poisson:
        outliers_df.loc[poisson_genes_step1, "theta"] = True

    model_parameters["od_factor"] = np.log10(1 + np.power(10, genes_log10_gmean_step1) / model_parameters["theta"])

    model_parameters_to_return = model_parameters.copy()

    non_outliers = outliers_df.sum(1) == 0
    outliers = outliers_df.sum(1) > 0

    if verbosity:
        print("Total outliers: {}".format(np.sum(outliers)))
    
    genes_non_outliers = genes_step1[non_outliers]
    genes_step1 = genes_step1[non_outliers]
    genes_log10_gmean_step1 = genes_log10_gmean_step1[non_outliers]

    model_parameters = model_parameters.loc[genes_non_outliers]

    if exclude_poisson:
        non_poisson_genes = set(model_parameters.index.tolist()).difference(poisson_genes)

        model_parameters = model_parameters.loc[non_poisson_genes]
    
    model_parameters_fit = get_regularized_params(
        model_parameters,
        genes,
        genes_step1,
        genes_log10_gmean_step1,
        genes_log10_gmean,
        cell_attr,
        count_data,
        exclude_poisson=exclude_poisson,
        poisson_genes=poisson_genes,
        useR=useR)

    end = time.time()
    step2_time = np.ceil(end - start)
    if verbosity:
        print("Step2 done. Took {} seconds.".format(np.ceil(end - start)))
    
    # Step 3: Calculate residuals
    if verbosity:
        print("Running Step3")

    start = time.time()
    model_matrix = dmatrix(" + ".join(latent_var), cell_attr)
    residuals = pd.DataFrame(get_residuals(count_data, model_matrix, model_parameters_fit)).T

    residuals.index = genes
    residuals.columns = cell_names
    end = time.time()
    step3_time = np.ceil(end - start)
    if verbosity:
        print("Step3 done. Took {} seconds.".format(np.ceil(end - start)))

    gene_attr["theta_regularized"] = model_parameters_fit["theta"]
    gene_attr["residual_mean"] = residuals.mean(1)
    gene_attr["residual_variance"] = residuals.var(1)

    corrected_counts = None
    if correct_counts:
        corrected_counts = correct(
            residuals, cell_attr, latent_var, model_parameters_fit, count_data)
    
    return {
        "residuals": residuals,
        "model_parameters": model_parameters_to_return,
        "model_parameters_fit": model_parameters_fit,
        "corrected_counts": corrected_counts,
        "genes_log10_gmean_step1": genes_log10_gmean_step1_to_return,
        "genes_log10_gmean": genes_log10_gmean,
        "genes_log10_amean_step1": genes_log10_amean_step1_to_return,
        "genes_log10_amean": genes_log10_amean,
        "cell_attr": cell_attr,
        "model_matrix": model_matrix,
        "gene_attr": gene_attr,
        "step1_time": step1_time,
        "step2_time": step2_time,
        "step3_time": step3_time,
        "total_cells": len(cell_names),
    }
