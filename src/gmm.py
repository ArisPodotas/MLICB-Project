 # this is a pupative file for if we use a gaussian mixture to model and generate data for cells
 # We used a gmm actually

from collections.abc import Callable
import joblib
from math import ceil, sqrt, log
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from random import choice, uniform
from scipy import stats
from sklearn.mixture import GaussianMixture
from typing import Any
from utils import Utils, stdev, rnHexPicker

class CellGenerator:
    """
    Encapsulates the generative model based on gaussian mixture models
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    Arguments:
        dataFolder: str, A folder to read data from 
        outputPath: str, The path to save optional outputs to
        distributions: dict[str, list[float]], Some data structure from the Utils calss for input on the gene domains
    """
    def __init__(
        self,
        dataFolder: str,
        outputPath: str,
        distributions: dict[str, list[float]],
    ) -> None:
        self.outputPath: str = outputPath + '/Generator/'
        self.dataFolder: str = dataFolder
        self.dataStructure: dict[str, list[float]] = distributions
        self.fitted: bool = False # This parameter is to avoid redundant fitting of the models for time efficiency
        self.isFit: dict[str, bool] = {}
        self.GMM: dict[str, GaussianMixture] = {}

    def showDomain(
        self,
        gene: str,
        data: dict[str, list[float]] | None = None,
        path: str | None = None,
        show: bool = False,
        hold: int = 2,
    ) -> None:
        """
        Makes a plot for the distribution of a gene from the given data
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Note: will make a plot that optionally gets saved
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            gene: str, The gene for who's distribution to look for
            data: dict[str, list[float]] | None = None, Data to find the gene in
            path: str | None = None, The path to save to (note that this toggles saving)
            show: bool = False, Toggle showing the outputs of the distribution
            hold: int = 2, How long to show the plot for
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        data = data or self.dataStructure
        assert isinstance(data, dict), f'Problem with parameter data in method showDomain in class CellGenerator'
        fig, ax = plt.subplots(figsize = (12, 16))
        query: list[float] = data[gene]
        # Scotts rule with a larger resolution
        n: int = len(query)
        sigma: float = stdev(query)
        h: float = 3.5 * sigma / (n ** (1/3))
        k: int = ceil((np.max(query) - np.min(query)) / h + sqrt(n) + log(n, 2))
        ax.hist(query, bins = k)
        ax.set_title(f"Gene: {gene}'s histogram")
        ax.legend()
        ax.grid()
        if show:
            plt.pause(hold)
        plt.close()
        if path:
            os.makedirs(f'{path}/Histograms/', exist_ok = True)
            fig.savefig(f'{path}/Histograms/gene_{gene}_hist.png')
            assert os.path.exists(f'{path}/Histograms/gene_{gene}_hist.png'), f'Path to save histograms could not be found'

    def showAllDomains(
        self,
        data: dict[str, list[float]] | None = None,
        path: str | None = None,
        show: bool = False,
        hold: int = 2,
    ) -> None:
        """
        Iterates over all genes to show the domain of the expression values of each
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Note: will make a plot that optionally gets saved
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            data: dict[str, list[float]] | None = None, Data to find the gene in
            path: str | None = None, The path to save to (note that this toggles saving)
            show: bool = False, Toggle showing the outputs of the distribution
            hold: int = 2, How long to show the plot for
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        data = data or self.dataStructure
        for gene in data.keys():
            self.showDomain(
                gene = gene,
                data = data,
                path = path,
                show = show,
                hold = hold,
            )

    def sampleRandom(
        self,
        distribution,
        **samplerArguments,
    ) -> float:
        """
        Handles the gene expression generations for the random sampler
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Arguments:
            distribution: list[float],
            **samplerArguments,
        None default arguments will default to some class attribute
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Returns:
            cell: dict[str, float], The cell datastructure
        """
        start: float = min(distribution)
        stop: float = max(distribution)
        return uniform(start, stop, **samplerArguments)

    def sampleGaussian(
        self,
        distribution,
        **samplerArguments,
    ) -> float:
        """
        Handles the gene expression generations for the Gaussian sampler
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Arguments:
        None default arguments will default to some class attribute
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Returns:
            cell: dict[str, float], The cell datastructure
        """
        mean: float = np.mean(distribution)
        std: float = np.std(distribution)
        addedArgs: dict[str, Any] = {'loc': mean, 'scale': std, 'size': 1}
        addedArgs.update(samplerArguments)  # Override with user-provided args
        return np.random.normal(distribution, **samplerArguments)[0]

    def sampleKDE(
        self,
        distribution,
        **samplerArguments,
    ) -> float:
        """
        Handles the gene expression generations for the Gaussian KDE sampler
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Arguments:
        None default arguments will default to some class attribute
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Returns:
            cell: dict[str, float], The cell datastructure
        """
        pdf: stats.gaussian_kde = stats.gaussian_kde(distribution, **samplerArguments)
        return pdf.resample(1)[0]

    def sampleGMM(
        self,
        distribution: list[float],
        gene: str,
        **samplerArguments,
    ) -> float:
        """
        Handles the gene expression generations for the GMM sampler
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Arguments:
        None default arguments will default to some class attribute
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Returns:
            cell: dict[str, float], The cell datastructure
        """
        if self.isFit[gene] == True:
            gmm: GaussianMixture = self.GMM[gene]
        else:
            gmm: GaussianMixture = GaussianMixture(**samplerArguments)
            gmm.fit(np.array(distribution, dtype = float).reshape(-1, 1))
        return gmm.sample(1)[0][0][0]

    def fitSelf(self, distributions) -> bool:
        """
        Ensures that the fitting of the GMM generator is done only once per gene so that the same distribution can be re used without redundant fitting of the model
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Arguments:
            distributions: dict[str, list[float]], The data structure that hold the gene expression distributions
        None default arguments will default to some class attribute
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Returns:
            cell: dict[str, float], The cell datastructure
        """
        distributions = distributions or self.dataStructure
        for gene in distributions.keys():
            if gene in self.isFit.keys():
                continue
            gmm: GaussianMixture = GaussianMixture()
            gmm.fit(np.array(distributions[gene], dtype = float).reshape(-1, 1))
            self.GMM[gene] = gmm
            self.isFit[gene] = True

    def generateCell(
        self,
        sampler: str,
        samplerArguments: dict[str, Any] = {},
        distributions: dict[str, list[float]] | None = None,
        genes: list[str] | None = None,
        nullGenes: bool = False,
    ) -> dict[str, float]:
        """
        Iterates over all genes to show the domain of the expression values of each
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Arguments:
            distributions: dict[str, list[float]], The data structure that hold the gene expression distributions
            sampler: str, A string to use as a key for a dictionary with callable functions that sample in different ways
                'Discreet': rn.choice,
                'Gaussian': uniform,
                'Random': np.random.normal,
                'Distribution': stats.gaussian_kde,
                'GMM': GaussianMixture,
            samplerArguments: dict[str, Any], The arguments to pass to the Callble
            genes: list[str] | None = None, A list of gene names to sample from
            nullGenes: bool = False, Toggle setting genes you don't want expressed to 0 values
        None default arguments will default to some class attribute
        ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Returns:
            cell: dict[str, float], The cell datastructure
        """
        distributions = distributions or self.dataStructure
        genes = genes or list(distributions.keys()) # So when none are given all the genes are done
        assert isinstance(genes, list), f'Problem with parameter genes in method generateCell in class CellGenerator'
        self.fitSelf(distributions)
        cell: dict[str, float] = {}
        alias: dict[str, Callable] = {
            'Random': self.sampleRandom,
            'Discreet': choice,
            'Gaussian': self.sampleGaussian,
            'Distribution': self.sampleKDE,
            'GMM': self.sampleGMM,
        }
        for gene in distributions.keys():
            query: Callable = alias[sampler]
            if gene in genes:
                distribution: list[float] = distributions[gene]
                # Something like a switch case
                if sampler == 'Random':
                    cell[gene] = query(distribution, **samplerArguments)
                elif sampler == 'Discreet':
                    cell[gene] = query(distribution, **samplerArguments)
                elif sampler == 'Gaussian':
                    cell[gene] = query(distribution, **samplerArguments)
                elif sampler == 'Distribution':
                    cell[gene] = query(distribution, **samplerArguments)
                elif sampler == 'GMM':
                    cell[gene] = query(distribution, gene, **samplerArguments)
                else:
                    raise ValueError(f'Problem with sampler parameter value in method generateCell in class CellGenerator')
            elif nullGenes:
                cell[gene] = 0.0
            else:
                continue
        return cell

    def generateCells(
        self,
        count: int,
        sampler: str,
        samplerArguments: dict[str, Any] = {},
        distributions: dict[str, list[float]] | None = None,
        genes: list[str] | None = None,
        nullGenes: bool = False,
    ) -> list[dict[str, float]]:
        """
        Generates as many cells as specified
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            count: int, The number of cells to make
            distributions: dict[str, list[float]], The data structure that hold the gene expression distributions
            sampler: str, A string to use as a key for a dictionary with callable functions that sample in different ways
                'Discreet': rn.choice,
                'Gaussian': uniform,
                'Random': np.random.noraml,
                'Distribution': stats.gaussian_kde,
                'GMM': GaussianMixture,
            genes: list[str], A list of gene names to sample from
            nullGenes: bool = False, Toggle setting genes you don't want expressed to 0 values
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            cells: list[dict[str, float]], The list of cell datastructures
        """
        output: list = [0] * count
        for index in range(count):
            query: dict[str, float] = self.generateCell(
                distributions = distributions,
                sampler = sampler,
                genes = genes,
                nullGenes = nullGenes,
            )
            output[index] = query
        return output

    def plotCellsCommon(
        self,
        cells: list[dict[str, float]] | None = None,
        save: bool = True,
        path: str | None = None,
        show: bool = False,
        hold: int = 2,
    ) -> None:
        """
        Plotst he scatter plots of all the genes and their expression together on one plot
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Note: Plots outputs
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            cells: list[dict[str, float]] | None = None, The data structure that holds the cells
            save: bool = True, Toggle saving to a file
            path: str | None = None, Specify the save path, names are auto generated
            show: bool = False, Toggle showing the outputs of the distribution
            hold: int = 2, How long to show the plot for
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        cells = cells or self.cells
        path = path or self.outputPath
        fig, ax = plt.subplots()
        for index, cell in enumerate(cells):
            col: str = rnHexPicker()
            ax.scatter(cell.keys(), cell.values(), label = f'Cell {index}', color = col)
        ax.set_title(f'Cells')
        ax.set_xlabel(f'Gene')
        ax.set_ylabel(f'Expression')
        if len(cells) < 5:
            ax.legend()
        if show:
            plt.pause(hold)
        if save:
            os.makedirs(f'{path}/Cells/', exist_ok = True)
            fig.savefig(f'{path}/Cells/Cells.png')
            assert os.path.exists(f'{path}/Cells/Cells.png')
        plt.close()

    def plotCellsDistinct(
        self,
        cells: list[dict[str, float]] | None = None,
        save: bool = True,
        path: str | None = None,
        show: bool = False,
        hold: int = 2,
    ) -> None:
        """
        Plotst he scatter plots of all the genes and their expression together on one plot
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Note: Plots outputs
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            cells: list[dict[str, float]] | None = None, The data structure that holds the cells
            save: bool = True, Toggle saving to a file
            path: str | None = None, Specify the save path, names are auto generated
            show: bool = False, Toggle showing the outputs of the distribution
            hold: int = 2, How long to show the plot for
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        cells = cells or self.cells
        path = path or self.outputPath
        for index, cell in enumerate(cells):
            fig, ax = plt.subplots()
            col: str = rnHexPicker()
            ax.scatter(cell.keys(), cell.values(), label = 'Cell {index}', color = col)
            ax.set_title(f'Cell index: {index}')
            ax.set_xlabel(f'Gene')
            ax.set_ylabel(f'Expression')
            if show:
                plt.pause(hold)
            if save:
                os.makedirs(f'{path}/Cells/', exist_ok = True)
                fig.savefig(f'{path}/Cells/Cell {index}.png')
                assert os.path.exists(f'{path}/Cells/Cell {index}.png')
            plt.close()

    def cellsToDF(
        self,
        cells: list[dict[str, float]] | None = None,
        path: str | None = None,
        name:str = 'GeneratedCells',
    ) -> pd.DataFrame:
        """
        Writes a dataframe of the cells
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            path: str | None = None, The folder for the instance
            name: str = 'CellGenerator', The filename to save (note adds the .pkl by itself)
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            df: pd.DataFrame, A dataframe of the cells
        """
        cells = cells or self.cells
        path = path or self.outputPath
        df: pd.DataFrame = pd.DataFrame(cells)
        os.makedirs(f'{path}/Cells/', exist_ok = True)
        df.to_csv(f'{path}/Cells/{name}.csv')
        return df

    def _save_(
        self,
        path: str | None = None,
        name: str = 'CellGenerator',
    ) -> None:
        """
        Saves the instance object of the class to path
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            path: str | None = None, The folder for the instance
            name: str = 'CellGenerator', The filename to save (note adds the .pkl by itself)
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        path = path or self.outputPath
        os.makedirs(f'{path}/Objects/', exist_ok = True)
        joblib.dump(self, f'{path}/Objects/{name}.pkl')

    def main(
        self,
        count: int = 5,
        sampler: str = 'GMM',
        samplerArguments: dict[str, Any] = {},
        distributions: dict[str, list[float]] | None = None,
        genes: list[str] | None = None,
        nullGenes: bool = False,
        save: bool = True,
        path: str | None = None,
        show: bool = False,
        hold: int = 2,
    ) -> list[dict[str, float]]:
        """
        Calls a small pipeline of the class methods
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            count: int, The number of cells to make
            distributions: dict[str, list[float]], The data structure that hold the gene expression distributions
            sampler: str, A string to use as a key for a dictionary with callable functions that sample in different ways
                'Discreet': rn.choice
                'Gaussian': uniform
                'Random': np.random.noraml
            genes: list[str], A list of gene names to sample from
            nullGenes: bool = False, Toggle setting genes you don't want expressed to 0 values
            save: bool = True, Toggle saving to a file
            path: str | None = None, Specify the save path, names are auto generated
            show: bool = False, Toggle showing the outputs of the distribution
            hold: int = 2, How long to show the plot for
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        self.cells: list[dict[str, float]] = self.generateCells(
            count = count,
            sampler = sampler,
            samplerArguments = samplerArguments,
            distributions = distributions,
            genes = genes,
            nullGenes = nullGenes,
        )
        self.cellsToDF()
        self.plotCellsDistinct(self.cells, save = save, hold = hold, path = path, show = show)
        return self.cells

