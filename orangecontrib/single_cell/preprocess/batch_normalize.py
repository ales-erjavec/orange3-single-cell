import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition.pca import PCA
from scipy.stats import pearsonr, spearmanr

from Orange.data import Domain, Table, ContinuousVariable, DiscreteVariable
from Orange.data.util import SharedComputeValue
from Orange.statistics.util import nansum, nanmedian
from Orange.preprocess.preprocess import Preprocess


hlp = """ Regress out cell-cycle effects on a gene-by-gene basis.
 
 Links:
 http://www.statsmodels.org/dev/generated/statsmodels.genmod.generalized_linear_model.GLM.html#statsmodels.genmod.generalized_linear_model.GLM
 http://satijalab.org/seurat/cell_cycle_vignette.html
 """

# TODO: how to treat zeroes?
# TODO: speed-up using pure least squares
# TODO: one-hot encoding


# Link / inverse link functions
LINK_IDENTITY = "identity"
LINK_LOG = "log"

LINKS = {
    LINK_IDENTITY: lambda x: x,
    LINK_LOG: lambda x: np.log(x)
}

INV_LINKS = {
    LINK_IDENTITY: lambda x: x,
    LINK_LOG: lambda x: np.exp(x)
}

def batch_fit_ls(data, link="identity", nonzero_only=True):
    """ Fit linear regression coefficients for each gene. """
    assert link in LINKS and link in INV_LINKS
    assert link != "log" or nonzero_only
    batch_vars = data.domain.metas
    atts = data.domain.attributes
    n = len(data)
    Z = np.hstack((np.ones((n, 1)), data[:, batch_vars].metas))
    W = np.zeros((len(atts), Z.shape[1]))
    for i, a in enumerate(atts):
        y = data[:, a].X.ravel()
        Zt = Z
        if nonzero_only:
            y = y[y != 0]
            Zt = Z[y != 0, :]
        W[i, :] = np.linalg.lstsq(Zt, LINKS[link](y))[0]
    return W, batch_vars


def batch_transform_ls(data, link=LINK_IDENTITY, nonzero_only=True):
    """ Transform given data according to model. """
    global W, batch_vars
    Z = np.hstack((np.ones((len(data), 1)), data[:, batch_vars].metas))
    Xc = data.X.copy()
    Xn = INV_LINKS[link](
        LINKS[link](Xc) -
        Z.dot(W.T)
    )
    if nonzero_only:
        nz = np.where(Xc)
        Xc[nz] = Xn[nz]
    else:
        Xc = Xn
    return Table.from_numpy(domain=data.domain,
                            metas=data.metas,
                            X=Xc,
                            Y=data.Y,
                            W=data.W)






def test_fit_multivariate(plot=False):
    """ Test using log data. """
    np.random.seed(42)
    n = 100
    bias = 1
    alpha = 0.05
    Z = np.hstack([np.random.randn(n, 1),
                   np.random.choice([0, 1], size=n).reshape(n, 1)])
    noise = 0.1 * np.random.randn(n, 1)
    w = np.ones((2, 1))
    for link in LINKS.keys():
        X = INV_LINKS[link](bias + Z.dot(w) + noise)
        data = Table.from_numpy(metas=Z,
                                X=X,
                                domain=Domain(metas=[ContinuousVariable("Z0"),
                                                     DiscreteVariable("Z1", values=["a", "b"])],
                                              attributes=[ContinuousVariable("X")]))
        W, batch_vars = batch_fit_ls(data, link=link)
        newdata = batch_transform_ls(data, link=link)
        for z in Z.T:
            assert pearsonr(newdata.X.ravel(), z.ravel())[1] > alpha
            assert pearsonr(LINKS[link](newdata.X).ravel(), z.ravel())[1] > alpha
        assert pearsonr(LINKS[link](newdata.X), noise)[1] < alpha

    if plot:
        plt.figure()
        plt.plot(Z[:, 0].ravel(),  newdata.X.ravel(), ".", label="Processed")
        plt.plot(Z[:, 0].ravel(),  data.X.ravel(), ".", label="Original")
        plt.xlabel("Batch variable 0")
        plt.ylabel("Data")
        plt.legend()

        plt.figure()
        plt.plot(Z[:, 1].ravel()+0.05, newdata.X.ravel(), ".", label="Processed")
        plt.plot(Z[:, 1].ravel(), data.X.ravel(), ".", label="Original")
        plt.xlabel("Batch variable 0")
        plt.ylabel("Data")
        plt.legend()

        plt.figure()
        plt.plot(noise.ravel(), newdata.X.ravel() / newdata.X.max(), ".", label="Processed")
        plt.plot(noise.ravel(), data.X.ravel() / data.X.max(), ".", label="Original")
        plt.xlabel("Noise")
        plt.ylabel("data / max(data)")
        plt.show()



def test_fit_continuous(plot=False):
    """ Test all link functions for univariate continuous data. """
    np.random.seed(42)
    n = 100
    bias = 1
    alpha = 0.05
    Z = np.linspace(1, 3, n).reshape(n, 1)
    noise = 0.1 * np.random.randn(n, 1)
    for link in LINKS.keys():
        X = INV_LINKS[link](bias + Z + noise)
        data = Table.from_numpy(metas=Z,
                                X=X,
                                domain=Domain(metas=[ContinuousVariable("Z")],
                                              attributes=[ContinuousVariable("X")]))

        W, batch_vars = batch_fit_ls(data, link=link)
        newdata = batch_transform_ls(data, link=link)

        assert pearsonr(newdata.X, Z)[1] > alpha
        assert pearsonr(LINKS[link](newdata.X), Z)[1] > alpha
        assert pearsonr(LINKS[link](newdata.X), noise)[1] < alpha

    if plot:
        plt.figure()
        plt.plot(newdata.metas.ravel(), newdata.X.ravel(), label="Processed")
        plt.plot(Z.ravel(), X.ravel(), label="Original")
        plt.xlabel("Batch variable")
        plt.xlabel("Data")
        plt.legend()

        plt.figure()
        plt.plot(newdata.metas.ravel(), newdata.X.ravel(), label="Processed")
        plt.plot(newdata.metas.ravel(), noise, label="Noise")
        plt.legend()
        plt.show()


def test_batch():
    """ Test with Seurat data. """
    global models, batch_vars
    data = Table("/Users/martins/Dev/orange3-single-cell/orangecontrib/single_cell/preprocess/matrix.tab")
    W, batch_vars = batch_fit_ls(data)
    newdata = batch_transform_ls(data)

    # Plot PCA
    vals = set(data.Y)
    colors = ["blue", "red", "orange"]
    pc1 = PCA(n_components=2).fit_transform(data.X)
    pc2 = PCA(n_components=2).fit_transform(newdata.X)
    for pcs, tit in zip((pc1, pc2), ("Original", "Processed")):
        plt.figure()
        plt.title(tit)
        for v in vals:
            inxs = np.where(data.Y == v)[0].astype(int)
            plt.plot(pcs[inxs, 0], pcs[inxs, 1], ".",
                     label=data.domain.class_var.values[int(v)], color=colors[int(v)])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
    plt.show()



