import argparse as arg
from collections.abc import Callable
import joblib
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
from random import randint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as Scale
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, silhouette_score, davies_bouldin_score
from typing import Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

class Utils:
    """
    Encapsulates of of the utility functions into one object
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    Arguments:
        dataFolder: str, A folder to read data from 
        outputPath: str, The path to save optional outputs to
    """
    def __init__(
        self,
        dataFolder: str,
        outputPath: str,
    ) -> None:
        assert isinstance(dataFolder, str), 'The folder for the data could not be read due to mismatch in argument type in class Utils in method __init__'
        assert isinstance(outputPath, str), 'The path should be interpretable for saving images in class Utils in method __init__'
        # Class inputs
        self.outputPath: str = outputPath + '/Utils/'
        self.dataFolder: str = dataFolder
        os.makedirs(self.outputPath, exist_ok = True) # Note: this does not overwrite the folder

    def excelStruct(
        self,
        folder: str | None = None,
    ) -> dict[str, pd.ExcelFile]:
        """
        Reads Excel files in a directory and stores their data in a dictionary.
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            folder: str, The path to the directory containing Excel files.
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            dataDict: dict[str, pd.ExcelFile] A dictionary listing each Excel file as a pandas ExcelFile object.
        """
        folder = folder or self.dataFolder
        assert isinstance(folder, str), f'Problem with parameter folder in method ExcelStruct in class Utils'
        dataDict: dict[str, pd.ExcelFile] = {}
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.xlsx'): 
                    full_path: os.PathLike = os.path.join(root, file)
                    xlsx: pd.ExcelFile = pd.ExcelFile(full_path)  
                    dataDict[file] = xlsx
                   # xlsx.close() # You do not want to close the files here, it will error when you want to use them again
        return dataDict

    def convertToDataFrames(
        self,
        dataDict: dict[str, pd.ExcelFile] | None = None,
        path: str | None = None
    ) -> dict[str, list[pd.DataFrame]]:
        """
        Takes a ExcelFile filled iterable and unwraps the structures to a bunch of dataframes
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataDict: dict[str, pd.ExcelFile] | None = None,
            path: str | None = None, Where to save if enabled (Note will be diabled if none enabled otherwise)
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            output: dict[str, list[pd.DataFrame]], A datastructure for the output dataframes similar to the input
        """
        dataDict = dataDict or self.dataDict
        assert isinstance(dataDict, dict), f'Problem with parameter dataDict in class Utils in method convertToDataFrames'
        path = path or self.outputPath
        output: dict[str, list[pd.DataFrame]] = {}
        if path:
            os.makedirs(f'{path}/CSV/', exist_ok = True)
        # Iterate the dataDict
        for name, file in dataDict.items():
            allNames: list[str] = file.sheet_names
            frameHolder: list = [0] * len(allNames)
            for index, sheet in enumerate(allNames):
                frameHolder[index] = file.parse(sheet)
                if path:
                    frameHolder[index].to_csv(f'{path}/CSV/{name[:-6]}_sheet_{sheet}.csv')
            output[name] = frameHolder
            file.close()
        return output

    def compileDataStructure(
        self,
        dataFrames: dict[str, list[pd.DataFrame]] | None = None,
        path: str | None = None,
        name: str = 'UtilsDataStructure',
    ) -> tuple[pd.DataFrame, dict[str, list[float]]]:
        """
        Takes the dataFrames attribute and compiles all the sheets to one data structure
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataFrames: dict[str, list[pd.DataFrame]] | None = None, Data to compile to the new datastructure
            path: str | None = None, Where to save if enabled (Toggle enabling saving too)
            name: str = 'UtilsDataStructure', The filename for saving, the .csv is added inside the function
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            output: pd.DataFrame, A dataframe that hold all the gene values
            holder: dict[str, list[float]], A dictionary with the full domain of each gene
        """
        dataFrames = dataFrames or self.dataFrames
        assert isinstance(dataFrames, dict), f'Problem with parameter dataFrames in class Utils in method compileDataStructure'
        holder: dict[str, list[float]] = {}
        for fileName, file in dataFrames.items(): # outputs a list
            for sheetIndex, sheet in enumerate(file): # sheet is a pd.DataFrame
                filtered: pd.DataFrame = isolateGenes(sheet) # Getting all gene value fields
                for gene in filtered.itertuples():
                    # To undo redundant key name like atp-1 and atp1
                    geneName: str = gene[1].replace('-', '')
                    if geneName in holder.keys():
                        # Handle seen genes
                        for value in gene[2:]:
                            try:
                                holder[geneName].append(float(value))
                            except:
                                pass
                    else:
                        # Handle new genes
                        holder[geneName] = []
                        for value in gene[2:]:
                            try:
                                holder[geneName].append(float(value))
                            except:
                                pass
        output: pd.DataFrame = pd.DataFrame([holder])
        if path: 
            os.makedirs(f'{path}/CSV/', exist_ok = True)
            output.to_csv(f'{path}/CSV/{name}.csv')
        return output, holder

    # Todo
    def wholeFileGeneDistributions(
        self,
        save: bool = True,
        path: str | None = None,
        show: bool = False,
        hold: int = 2,
    ) -> None:
        """
        Generates the domain histogram of a gene from one dataframe
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            gene: str, The name of the gene
            domain: list[float], The genes domain
            save: bool = True, Toggle saving to a file
            path: str | None = None, Specify the save path, names are auto generated
            show: bool = False, Toggle showing the outputs of the distribution
            hold: int = 2, How long to show the plot for
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        pass

    def wholeSheelGeneDistributions(
        self,
        save: bool = True,
        path: str | None = None,
        show: bool = False,
        hold: int = 2,
    ) -> None:
        """
        Generates the domain histogram of a gene from one dataframe
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            gene: str, The name of the gene
            domain: list[float], The genes domain
            save: bool = True, Toggle saving to a file
            path: str | None = None, Specify the save path, names are auto generated
            show: bool = False, Toggle showing the outputs of the distribution
            hold: int = 2, How long to show the plot for
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        pass

    def geneDistribution(
        self,
        gene: str,
        domain: list[float],
        save: bool = True,
        path: str | None = None,
        show: bool = False,
        hold: int = 2,
    ) -> None:
        """
        Generates the domain histogram of a gene from one dataframe
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            gene: str, The name of the gene
            domain: list[float], The genes domain
            save: bool = True, Toggle saving to a file
            path: str | None = None, Specify the save path, names are auto generated
            show: bool = False, Toggle showing the outputs of the distribution
            hold: int = 2, How long to show the plot for
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        path = path or self.outputPath
        if save:
            os.makedirs(f'{path}/Histograms/GeneDistributions/', exist_ok = True)
        # Scotts rule
        n: int = len(domain)
        sigma: float = stdev(domain)
        h: float = 3.5 * sigma / (n ** (1/3))
        k: int = ceil((np.max(domain) - np.min(domain)) / h + sqrt(n) + log(n, 2))
        plt.hist(domain, bins = k)
        ax.set_title(f"Gene: {gene}'s histogram")
        ax.legend()
        ax.grid()
        if show:
            plt.pause(hold)
        plt.close()
        if path:
            os.makedirs(f'{path}/Histograms/GeneDistributions/', exist_ok = True)
            fig.savefig(f'{path}/Histograms/GeneDistributions/gene_{gene}_hist.png')
            assert os.path.exists(f'{path}/Histograms/GeneDistributions/gene_{gene}_hist.png'), f'Path to save histograms could not be found'

    def dataFrameHistograms(
        self,
        structure: dict[str, list[pd.DataFrame]] | None = None,
        save: bool = True,
        path: str | None = None,
    ) -> None:
        """
        Generates the histogram representation of the dataframe for each file for each sheet in the class self.dataFrames attribute or a given structure
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            structure: dict[str, list[pd.DataFrame]] | None = None, The data structure to find the data in
            save: bool = True, Toggle saving to a file
            path: str | None = None, Specify the save path, names are auto generated
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        structure = structure or self.dataFrames
        path = path or self.outputPath
        if save:
            os.makedirs(f'{path}/Histograms/DataFrames/', exist_ok = True)
        for name, excel in tqdm(
                structure.items(),
                desc = f'Making DataFrame Histograms',
                nrows = 40,
            ):
            for index, frame in enumerate(excel):
                ax = frame.hist(figsize = (25, 28))
                if save:
                    plt.savefig(f'{path}/Histograms/DataFrames/{name[:-6]}_{index}.png')
                plt.close()

    def dataFramesDescribe(
        self,
        structure: dict[str, list[pd.DataFrame]] | None = None,
    ) -> None:
        """
        Generates the descriptive statistics of the dataframe for each file for each sheet in the class self.dataFrames attribute or a given structure
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            structure: dict[str, list[pd.DataFrame]] | None = None, The data structure to find the data in
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        structure = structure or self.dataFrames
        for excel in structure.values():
            for frame in excel:
                frame.describe()

    def dataFramesHead(
        self,
        structure: dict[str, list[pd.DataFrame]] | None = None,
        size: int = 10,
    ) -> None:
        """
        Generates the histogram head (so the first few entries) of the dataframe for each file for each sheet in the class self.dataFrames attribute or a given structure
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            structure: dict[str, list[pd.DataFrame]] | None = None, The data structure to find the data in
            size: int = 10, How many rows to show
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        structure = structure or self.dataFrames
        for excel in structure.values():
            for frame in excel:
                frame.head(size)
    
    def convertFieldToBinaryLabels(
        self,
         col: int,
        categories: list, 
        dataframe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        This function takes a dataframe and converts a field to a binary valued 0, 1, 2 ... field
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataframe: pd.DataFram | None = None, The input data
            col: int, The index of the column to use
            categories: list, The list of the labels to convert values to
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            output: pd.DataFrame, A copy of the data with the interpolated labels
        """
        dataframe = dataframe or self.data
        assert isinstance(dataframe, pd.DataFrame), f'Problem using data in class method findMissing'
        output: pd.DataFrame = dataframe.copy()
        # Isolating our column
        target = output.iloc[:, col].astype("category")
        # Casting to categorical as in the documentation https://pandas.pydata.org/docs/user_guide/10min.html
        target = target.cat.rename_categories(categories) 
        # Re-assigning to dataframe
        output.iloc[:, col] = target
        return output


    def upperTriangle(
        self,
        matrix: np.ndarray,
    ) -> tuple[list, list]:
        """
        This function is to return the two lists of indecies to use to plot a matrix
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            matrix: np.ndarray, The matrix to use
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            redundant: list, A list of the redundant index values to use (say rows of the triangle)
            serisl: list, A list of non redundant indexes to use (say columns of the triangle)
        """
        # I need two arrays, one for redundant values and one for non redundant ones
        # Since i need something like
        # 1 - 1
        # 2 - 1
        # 2 - 2
        # 3 - 1
        # 3 - 2
        # ...
        width: int = len(matrix) - 1
        height: int = width
        volume: float = int((width) * (height) / 2) # of a triangle (technically an area)
        redundant: list = [0] * volume
        serial: list = [0] * volume
        holder: int = 0
        for i in range(1, len(matrix) - 1): # This one starts at 1 since the 0 index is on the diagonal I guess we could leave it and it would skip it
            for j in range(0, len(matrix) - 1):
                if i > j:
                    holder += 1
                    redundant[holder] = i
                    serial[holder] = j
                else:
                    continue
        return redundant, serial

    def correlations(
        self,
        dataframe: pd.DataFrame | None = None,
        show: bool = False,
        hold: int = 2,
        save: bool = True,
        name: str = '/Correlations of the dataframe',
    ) -> None:
        """
        This function will take the correlation matrix of the dataframe and make a scatter plot for it (col 1 index, col 2 index, corr). The correlation matrix is the Pearson correlation matrix
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Note: Will make files in the output directory
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataframe: pd.DataFram | None = None, The input data
            show: bool = False, Toggle showing the image in the matplotlib gui
            hold: int = 2, How long to present the image in the gui
            save: bool = True, Toggle saving to a file
            name: str = 'Correlations of the dataframe', Name of the output image (the folder is still the output dir and can't change from inputs here)
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            output: None
        """
        dataframe = dataframe or self.data
        table: pd.DataFrame = dataframe.corr()
        # We only need everything above the diagonal
        rows, cols = self.upperTriangle(table.values)  # Originally solved inside this function now moved to the one above
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        # I'm going to multiply the corr() output by alot since I want to have the delta z visible on the plot outside of just the color heatmap
        ax.scatter(rows, cols, 1000 * table.values[rows, cols], label = 'Pearson corr', c = table.values[rows, cols], cmap = 'YlOrBr')
        plt.title(f"Correlations of the dataframe")
        ax.set_xlabel(f"Col 1")
        ax.set_ylabel(f"Col 2")
        ax.set_zlabel("Pearson correlation of dataframe")
        plt.grid()
        if show:
            plt.pause(hold)
        if save:
            fig.savefig(self.outputPath + '/' + name + '.png')

    def twoColCorrelation(
        self,
        anchor: int,  
        dataframe: pd.DataFrame | None = None,
        show: bool = False,
        hold: int = 2,
        save: bool = True,
    ) -> None:
        """
        this function does the 2d scatter plots of the anchor column with all the rest
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataframe: pd.DataFram | None = None, The input data
            anchor: int, The index of the column of the data to use as the anchor (to be the one common axis)
            show: bool = False, Toggle showing the image in the matplotlib gui
            hold: int = 2, How long to present the image in the gui
            save: bool = True, Toggle saving to a file
        None defaults will be set to some class attribute
        Names the images in a default format
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            output: None,
        """
        dataframe = dataframe or self.data
        for index in range(len(dataframe.columns)):
            plt.scatter(dataframe.iloc[:, anchor], dataframe.iloc[:, index], label = f'col {anchor}, {index}')
            plt.xlabel(f'anchor: {dataframe.columns[anchor]}')
            plt.ylabel(f'col {index}: {dataframe.columns[index]}')
            plt.grid()
            plt.title(f'Scatter plot of columns {anchor} {dataframe.columns[anchor]}, {index} {dataframe.columns[index]}')
            # You don't really want to turn this option on trust me
            if show:
                plt.pause(hold)
            if save:
                fig.figsave(self.outputPath + f'/Regression of {anchor}, {index}' '.png')

    def zScaler(
        self,
        start: int,
        stop: int, 
        dataframe: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, Scale]:
        """
        Scales the columns from start to end of the dataframe using z-score scaling
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataframe: pd.DataFram | None = None, The input data
            start: int, Column to start from
            stop: int, Column to end on
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            output: pd.DataFrame, The scaled dataframe
            transform: Scale, The scaler object used
        """
        dataframe = dataframe or self.data
        target: pd.DataFrame | pd.Series = dataframe.iloc[:, start:stop]
        transform: Scale = Scale()
        transform.fit(target)
        holder = transform.transform(target)
        output: pd.DataFrame = dataframe.copy()
        output.iloc[:, start:stop] = holder
        return output, transform

    def applyScaler(
        self,
         start: int,
        stop: int,
        scaler: Scale, 
        dataframe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Applies a scaler to columns from start to end of the dataframe
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataframe: pd.DataFram | None = None, The input data
            start: int, The starting column to apply the scaling to
            stop: int, The end column to apply the scaling to
            scaler: Scale, The scaler object
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            output: pd.DataFrame, The scaled dataframe
        """
        dataframe = dataframe or self.data
        target: pd.DataFrame | pd.Series = dataframe.iloc[:, start:stop]
        holder = scaler.transform(target)
        output: pd.DataFrame = dataframe.copy()
        output.iloc[:, start:stop] = holder
        return output

    def searchPca(
        self,
        data: pd.DataFrame | None = None,
        show: bool = False,
        hold: int = 2,
        save: bool = True,
        name: str = 'Explain of all principaled components', 
    ) -> None:
        """
        Calculates and plots the explain values for each feature
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Note: Will make images as output
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataframe: pd.DataFram | None = None, The input data
            show: bool = False, Toggle showing the image in the matplotlib gui
            hold: int = 2, How long to present the image in the gui
            save: bool = True, Toggle saving to a file
            name: str = 'Explain of all principaled components', Name of the output image (the folder is still the output dir and can't change from inputs here)
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            output: None
        """
        data = data or self.data
        # I notmalize the blue curve to the red one so that both are visible in detail
        pca = PCA()
        pca.fit(data)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(data.columns) + 1), pca.explained_variance_ratio_, color = 'red', label = 'Raw variance')
        ax.plot(range(1, len(data.columns) + 1), [sum(pca.explained_variance_ratio_[0:i]) for i in range(len(data.columns))], color = 'blue', label = 'Cummulative variance')
        ax.grid()
        ax.legend()
        ax.set_title('Explain (%) of principal components')
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance')
        if show:
            plt.pause(hold)
        if save:
            fig.savefig(f'{self.outputPath}/{name}.png')

    def transformPca(
        self,
        components: int, 
        dataframe: pd.DataFrame | None = None,
    ) -> tuple:
        """
        Applies a PCA to the dataframe
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataframe: pd.DataFram | None = None, The input data
            components: int, # of components to keep
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            pcaDf: pd.DataFrame, The transformed space dataframe
            obj.explained_variance_ratio_, The explained variance ratio of each principal component
        """
        dataframe = dataframe or self.data
        data = dataframe.iloc[:, 2:].values
        obj = PCA(n_components = components)
        fit: PCA = obj.fit_transform(data)
        pcaDf: pd.DataFrame = pd.DataFrame(fit, columns = [f'P.C. {i+1}' for i in range(components)])
        return pcaDf, obj.explained_variance_ratio_

    def componentSearch(
        self,
          cutoff: float | int, 
        dataframe: pd.DataFrame | None = None,
    ) -> int:
        """
        This function will do a search on the results of the pca for different components until it find the minimum number of omponents with a cutoff explain. Will return the components to keep
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataframe: pd.DataFram | None = None, The input data
            cutoff: float | int, The cutoff for the explain to search for (inclusive)
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            output: int, the # of components to use, -1 if None
        """
        dataframe = dataframe or self.data
        for n in range(len(dataframe.columns) - 2): # The -2 is because we skip the first 2 columns
            dfPca, explain = self.transformPca(dataframe, n)
            if sum(explain) >= cutoff:
                output: int = n
                break
        if output:
            return output
        else:
            return -1

    def implementPca(
        self,
         cutoff: float | int, 
        dataframe: pd.DataFrame | None = None,
        verbose: bool = False
    ) -> tuple:
        """
        This function will return the pca resutls for the first Pca run that passes cutoff
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            dataframe: pd.DataFram | None = None, The input data
            cutoff: float | int, The cutoff for the explain to search for (inclusive)
            verbose: bool = False, Prints result
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            pcaDf: pd.DataFrame, The transformed space dataframe
            obj.explained_variance_ratio_, The explained variance ratio of each principal component
        """
        dataframe = dataframe or self.data
        components = self.componentSearch(dataframe, cutoff)
        pca = self.transformPca(dataframe, components)
        if verbose:
            print(f'Total explain: {sum(pca[1])}\nComponents {components}')
        return pca

    def _save_(
        self,
        name: str = 'Utils',
        path: str | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Generates a .pkl file for the class instance
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            name: str = 'Utils', Name of the file to save to (.pkl is added in the function)
            path: str | None = None, The folder to put it in
            verbose: bool = False, Toggle showing the file path in the terminal
        None defaults will be set to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        path = path or self.outputPath
        assert isinstance(path, str), f'Problem with parameter path in class Utils method _save_'
        self.closeAllOpenFiles()
        os.makedirs(f'{path}/Objects/', exist_ok = True)
        joblib.dump(self, f'{path}/Objects/{name}.pkl')
        if verbose:
            print(f'Saved Utils instance to {path}/Objects/{name}.pkl')

    def closeAllOpenFiles(
        self,
    ) -> None:
        """
        Somewhere in the class there are open files despite the .close() method calls so we aim to close them
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        if self.dataDict:
            for file in self.dataDict.values():
                file.close()

    def main(
        self,
        name: str = 'Utils',
        path: str | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Calls a minimal pipeline of the class methods
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Arguments:
            name: str = 'Utils', Name of the file to save to (.pkl is added in the function)
            path: str | None = None, The folder to put outputs in
            verbose: bool = False, Toggle showing the file path in the terminal
        None default parameters will default to some class attribute
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
        Returns:
            None
        """
        self.dataDict: dict[str, pd.ExcelFile] = self.excelStruct()
        self.dataFrames: dict[str, list[pd.DataFrame]] = self.convertToDataFrames(path = path)
        self.compiledGenes, self.dataStructure = self.compileDataStructure(path = path)
        self.closeAllOpenFiles()
        if path:
            self._save_(path = path, name = name, verbose = verbose)

def liner() -> arg.Namespace:
    """
    Parsers command line input
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    Note some of the arguments are executed in the function, things like making directories. You will not need to do prerequisites for some of the returned parameters
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    Returns:
        cmd: argparse.Namespace, The parsed arguments
    """
    prs: arg.ArgumentParser = arg.ArgumentParser(
        prog = f'Assignment for MLICB',
        description = 'Aris Podotas, Rafail Adam, George Leventis. 2024-2025\nThe paths are relative to the working directory not the python file, the script is made to run from the directory the python files are in (so ../src/). Keep that in mind because it uses shutil.rmtree(cmd.out).',
        epilog = 'ID\'s: 7115152400040, 7115152400009, 711572100024',
    )
    general = prs.add_argument_group("General input settings")
    general.add_argument(
        "--version",
        dest="ver",
        action="version",
        version="%(prog)s 0.1.3",
        help = 'Prints the version of the program',
    )
    files = prs.add_argument_group(f'Paths')
    files.add_argument(
        '-inp',
        '--input',
        dest = 'input',
        default = '/home/bio/MLICB/data/',
        help = 'Defines the input folder to use'
    )
    files.add_argument(
        '-out',
        '--output',
        dest = 'out',
        default = '/home/bio/MLICB/outputs/',
        help = 'Defines the output folder for all the images. Please use relative paths'
    )
    funcs = prs.add_argument_group('Functionality')
    funcs.add_argument(
        '-clean',
        '--clean-output',
        dest = 'clean',
        action = 'store_true',
        default = False,
        help = 'If set will delete the output dir and re make it. Please use relative paths'
    )
    funcs.add_argument(
        '-cells',
        '--num-cells',
        dest = 'cells',
        type = int,
        default = 100,
        help = 'The number of cells to generate',
    )
    funcs.add_argument(
        '-runs',
        '--num-runs',
        dest = 'runs',
        type = int,
        default = 100,
        help = 'The number of fit, generate , evaluate loops to do',
    )
    cmd: arg.Namespace = prs.parse_args()
    cmd.output = cmd.out # Literally becasue I would forget that I named it out
    if cmd.clean:
        if os.path.exists(cmd.out):
            inp = input(f'Remove dir {cmd.out}? (Y/n):')
            if inp.lower() == 'y':
                    shutil.rmtree(cmd.out) # Surely nothing bad can come of this
    os.makedirs(cmd.out, exist_ok = True)
    return cmd

def isolateGenes(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    Arguments:
        df: pd.DataFrame, The input df
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    Returns:
        filtdf: pd.DataFrame | pd.Series, only the subsections of df that has gene data
    """
    geneCols: list = [col for col in df.columns if "Group" in col or "gene" in col or "Gene" in col]
    filtdf: pd.DataFrame | pd.Series = df[geneCols]
    return filtdf

def stdev(data):
    """
    Calculates the standard deviation of a dtaset since the imports wont work for some reason
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    Arguments:
        data: Sequence[Literal], The data
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    Returns:
        std: float, The standard deviation of the data
    """
    n: int = len(data)
    mean: float = sum(data) / n
    sumSquaredDiff = sum((x - mean) ** 2 for x in data)
    variance = sumSquaredDiff / (n - 1)
    return sqrt(variance)

def rnHexPicker():
    """
    Generates a random string for a hex color
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    Returns:
        hex: str, The hex color 
    """
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return f"#{r:02x}{g:02x}{b:02x}"

def applyMetrics(
    labels: list[np.ndarray],
    X: np.ndarray,
    algorithms: list[str],
    metrics: list[Callable] = [
        calinski_harabasz_score,
        silhouette_score,
        davies_bouldin_score,
    ],
    arguments: list[dict[str, Any]] = [
        {},
        {},
        {},
    ],
) -> np.ndarray:
    """
    Applies all the given metrics to the predictions assuming as ground truth the truth given.
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Arguments:
        show: bool = False: Toggle drawing the matplot canvas
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    """
    holder: int = len(algorithms)
    output: np.ndarray = np.array(
        [
            [
                0
            ] * holder
        ] * len(metrics)
    , dtype = np.float64)
    # So we will end up with the following
    # [[[option1], [option2], [option3], ...], # metric1
    # [[option1], [option2], [option3], ...], # metric2
    # [...], # ...3
    # [...], # ...4
    # ...]
    i: int = 0
    for metric, args in zip(metrics, arguments):
        for index in range(holder):
            try:
                output[i, index] = metric(X, labels[index], **args)
            except:
                output[i, index] = 0.0
        i += 1
    return output

def visualizeMetrics(
    input: np.ndarray,
    methods: list[str],
    metrics: list[str],
    xBuffer: float = -0.025,
    yBuffer: float = 0.05,
    name: str | None = None,
    show: bool = False,
    save: bool = True,
    hold: int = 2,
    path: str = '../outputs/',
) -> None:
    """
    Takes the output of the applyMetrics function and displayes it with matplotlib
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Arguments:
        show: bool = False: Toggle drawing the matplot canvas
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Returns:
        None
    """
    holder: int = len(methods)
    temp: int = len(metrics)
    fig, ax = plt.subplots(nrows = temp, ncols = 1, figsize = (6, min(3 * temp, 12)))
    plt.subplots_adjust(hspace = 0.5)
    for i in range(temp): # Should iterate input col
        title_: str = metrics[i]
        for index in range(holder): # Should iterate input row
            lab: str = methods[index]
            ax[i].scatter(index, input[i, index], label = f'{lab}')
            ax[i].text(index + xBuffer, input[i, index] + yBuffer, str(np.around(input[i, index], 2)))
        ax[i].set_title(f"Metric: {title_}") # Funciton are first class objects in python so __name__ just returns the function name string
        ax[i].set_ylabel(f'{title_} Value')
        ax[i].set_xlabel('Method')
        # ax[i].set_ylim(-0.05, 1.05)
        ax[i].legend()
        ax[i].grid()
    if show:
        plt.pause(hold)
    if save:
        os.makedirs(path + 'Evaluations/', exist_ok=True)
        fig.savefig(path + f'Evaluations/{name}_eval.png')
        assert os.path.exists(path + f'Evaluations/{name}_eval.png')
    plt.close()

def makeBoxPlots(
    scores: np.ndarray,
    methods: list[str],
    metrics: list[str],
    xBuffer: float = -0.025,
    yBuffer: float = 0.05,
    save: bool = True,
    show: bool = False,
    hold: int = 2,
    name: str = './full_run_metrics.png',
) -> None:
    """
    Documentation
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Arguments:
        arg: type
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Returns:
        output: None
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    """
    assert isinstance(methods, list), f'Type mismatch in function {insp.currentframe().f_code_co_name}\'s methods parameter'
    assert isinstance(metrics , list), f'Type mismatch in method {insp.currentframe().f_code_co_name}\'s metrics parameter'
    holder = len(methods)
    temp = len(metrics)
    fig, ax = plt.subplots(
        nrows=holder,
        ncols=temp,
        figsize=(
            24,
            28,
        ),
    )
    for i in range(temp): # Should iterate input col
        for index in range(holder): # Should iterate input row
            ax[index, i].boxplot(scores[:, i, index], showmeans=True, meanline=True, sym = '.') # index, i is row, col in matplotlib
            ax[index, i].text(
                1 + xBuffer,  # Always 1 for single box
                np.mean(scores[:, i, index]) + yBuffer,
                str(np.around(np.mean(scores[:, i, index]), 2)),
                ha='center',
                va='bottom',
                clip_on=True
            )
            ax[index, i].set_title(f"Metric: {metrics[i]}") # Funciton are first class objects in python so __name__ just returns the function name string
            ax[index, i].grid()
            ax[index, i].set_ylabel(f'{metrics[i]} Value')
            ax[index, i].set_xlabel(f'Method: {methods[index]}')
    if show:
        plt.pause(hold)
    if save:
        fig.savefig(name)
        assert os.path.exists(name)
    plt.close()

