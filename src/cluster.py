# This calss is for the main clustering of a dataset
# The class should contain the Utils object that splits thing for it

from chameleon import cluster as chameleonAlgo
from chameleon import plot2d_data as plotChameleon
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
from typing import Any
from utils import *

class ClusterPipeline:
    """
    This class is for clustering muti dimenional data
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    Arguments:
        data: pd.DataFrame | np.ndarray, A structure with the data
    """
    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
    ) -> None:
        if isinstance(data, pd.DataFrame):
            self.data: np.ndarray = data.to_numpy()
            self.chameleonData = data
        else:
            self.chameleonData = None
            self.data: np.ndarray = data

    def chameleon(
        self,
        data: np.ndarray | None = None,
        k: int = 20,
    ):
        """
        This class is for clustering muti dimenional data
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            data: pd.DataFrame | np.ndarray, A structure with the data
        """
        if data is None:
            data = self.chameleonData
        if self.chameleonData is not None:
            clustering: pd.DataFrame = chameleonAlgo(self.chameleonData, knn = k, k=4)
            # plotChameleon(*clustering)
        else:
            print('pending implementation')
            raise ValueError
        return clustering

    def knn(
        self,
        data: np.ndarray | None = None,
        k: int = 3
    ) -> tuple[np.ndarray]:
        """
        Defines the single link hierarchical agllomerative clustering algorithms for our class. It uses the pre built function from the sklearn library and wraps it for us.
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            data: np.ndarray | None = None, Imput data to cluster
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            labels: np.ndarray, The index of which cluster the points belong to
            self.sl, The sincle link object
        """
        if data is None:
            data = self.data
        knn_graph = kneighbors_graph(data, n_neighbors=k, mode='connectivity')
        n_components, labels = connected_components(knn_graph, directed=False)
        return labels

    def singleLink(
        self,
        data: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, AgglomerativeClustering]:
        """
        Defines the single link hierarchical agllomerative clustering algorithms for our class. It uses the pre built function from the sklearn library and wraps it for us.
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            data: np.ndarray | None = None, Imput data to cluster
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            labels: np.ndarray, The index of which cluster the points belong to
            self.sl, The sincle link object
        """
        if data is None:
            data = self.data
        self.sl: AgglomerativeClustering = AgglomerativeClustering(linkage='single')
        return self.sl.fit_predict(data)

    def completeLink(
        self,
        data: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, AgglomerativeClustering]:
        """
        Defines the complete link hierarchical agllomerative clustering algorithms for our class. It uses the pre built function from the sklearn library and wraps it for us.
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            data: np.ndarray | None = None, Imput data to cluster
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            labels: np.ndarray, The index of which cluster the points belong to
            self.cl, The complete link object
        """
        if data is None:
            data = self.data
        self.cl: AgglomerativeClustering = AgglomerativeClustering(linkage='complete')
        return self.cl.fit_predict(data)

    def ward(
        self,
        data: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Defines the complete link hierarchical agllomerative clustering algorithms for our class. It uses the pre built function from the sklearn library and wraps it for us.
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            data: np.ndarray | None = None, Imput data to cluster
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            labels: np.ndarray, The index of which cluster the points belong to
            self.wrd, The Ward algorithm object
        """
        if data is None:
            data = self.data
        self.wrd: AgglomerativeClustering = AgglomerativeClustering(linkage='ward')
        result = self.wrd.fit_predict(data)
        return result

    def spectral(
        self,
        data: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, SpectralClustering]:
        """
        Defines the complete link hierarchical agllomerative clustering algorithms for our class. It uses the pre built function from the sklearn library and wraps it for us.
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            data: np.ndarray | None = None, Imput data to cluster
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            labels: np.ndarray, the index of which cluster the points belong to
            self.spec, SpectralClustering object
        """
        if data is None:
            data = self.data
        self.spec: SpectralClustering = SpectralClustering()
        result = self.spec.fit_predict(data)
        return result

    def pipeline(
        self,
        data: np.ndarray | None = None,
        algorithms: list[str] = [
            #'means',
            'knn',
            'single',
            'complete',
            'ward',
            'spectral',
            'chameleon',
        ],
        arguments: list[dict[str, Any]] = [
            {},
            {},
            {},
            {},
            {},
            {},
        ]
    ) -> list[np.ndarray]:
        """
        Runs a small pipeline of the class methods
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            data: np.ndarray | None = None, Imput data to cluster
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            self.predictions: list[np.ndarray], A list of the outputs of all the algorithms
        """
        if data is None:
            data = self.data
        alias: dict[str, Callable] = {
            #'means': self.means,
            'knn': self.knn,
            'single': self.singleLink,
            'complete': self.completeLink,
            'ward': self.ward,
            'spectral': self.spectral,
            'chameleon': self.chameleon,
        }
        self.predictions: list = []
        for index, (query, args) in enumerate(zip(algorithms, arguments)):
            algo: Callable = alias[query]
            result: float = algo(data, **args)
            self.predictions.append(result)
        return self.predictions

    def _save_(
        self,
        path: str = './Clustering/',
        name: str = 'obj',
    ) -> None:
        """
        Saves the instance object of the class to path
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            path: str = './Clustering/', The folder for the instance
            name: str = 'obj', The filename to save (note adds the .pkl by itself)
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        os.makedirs(f'{path}', exist_ok = True)
        joblib.dump(self, f'{path}{name}.pkl')

