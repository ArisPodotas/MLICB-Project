import numpy as np
import pandas as pd
from argparse import Namespace
from utils import liner, Utils, applyMetrics, visualizeMetrics, makeBoxPlots
from gmm import CellGenerator
from cluster import ClusterPipeline
from tqdm import tqdm

# Todo add pca
def main(cmd: Namespace) -> tuple[float]:
    """
    This function calls the mainloop of fitting generating and evaluating the pipeline for cells
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    """
    util: Utils = Utils(cmd.input, cmd.out)
    file: str = '../data/CDgroup/expression/CD34_CD164_GroupLevelExpression.xlsx'
    df: pd.DataFrame = pd.read_excel(file)
    _, dist = util.compileDataStructure(dataFrames = {'only': [df]})
    generator: CellGenerator = CellGenerator(cmd.input, cmd.out, dist)
    cells: list[dict[str, float]] = generator.generateCells(cmd.cells, sampler = 'GMM')
    generator.cellsToDF(cells)
    inputs: pd.DataFrame = pd.read_csv(f'{cmd.out}Generator/Cells/GeneratedCells.csv')
    cluster: ClusterPipeline() = ClusterPipeline(inputs)
    outputs = cluster.pipeline()
    evaluations: tuple = applyMetrics(
        X = inputs.to_numpy(),
        labels = outputs,
        algorithms = [
            'knn',
            'single',
            'complete',
            'ward',
            'spectral',
            'chameleon'
        ],
    )
    visualizeMetrics(
        evaluations,
        [
            'knn',
            'single',
            'complete',
            'ward',
            'spectral',
            'chameleon'
        ],
        [
            'calinski_harabasz',
            'silhouette',
            'davies_bouldin',
        ],
        path = cmd.out
    )
    return evaluations

def mainWrapper() -> None:
    """
    Wraps the main loop to generate distributions of metrics
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    """
    cmd: Namespace = liner()
    methods = [
        'knn',
        'single',
        'complete',
        'ward',
        'spectral',
        'chameleon'
    ]
    metrics = [
        'calinski_harabasz',
        'silhouette',
        'davies_bouldin',
    ]
    holder: np.ndarray = np.zeros((cmd.runs, len(metrics), len(methods)), dtype = float)
    for index in tqdm(
        range(cmd.runs),
        desc = 'Runs',
        nrows = 40,
    ):
        holder[index] = main(cmd)
    makeBoxPlots(holder,
        methods = methods,
        metrics = metrics,
        name = f'{cmd.out}full_run_metrics.png'
    )

if __name__ == "__main__":
    mainWrapper()


