import numpy as np
import pandas as pd
from argparse import Namespace
from utils import liner, Utils, applyMetrics, visualizeMetrics, makeBoxPlots
from gmm import CellGenerator
from cluster import ClusterPipeline
from tqdm import tqdm

def searchK(cmd: Namespace) -> tuple[float]:
    """
    Searches for the optimal K for the KNN
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
    outputs = cluster.pipeline(
        algorithms = [
            'knn',
            'knn',
            'knn',
            'knn',
            'knn',
        ],
        arguments = [
            {'k': 3},
            {'k': 4},
            {'k': 5},
            {'k': 6},
            {'k': 7},
        ]
    )
    evaluations: tuple = applyMetrics(
        X = inputs.to_numpy(),
        labels = outputs,
        algorithms = [
            'knn-3',
            'knn-4',
            'knn-5',
            'knn-6',
            'knn-7',
        ],
    )
    visualizeMetrics(
        evaluations,
        [
            'knn-3',
            'knn-4',
            'knn-5',
            'knn-6',
            'knn-7',
        ],
        [
            'calinski_harabasz',
            'silhouette',
            'davies_bouldin',
        ],
        path = cmd.out
    )
    return evaluations

def kSearchWrapper() -> None:
    """
    Wraps the main loop to generate distributions of metrics
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
    """
    cmd: Namespace = liner()
    methods = [
        'knn-3',
        'knn-4',
        'knn-5',
        'knn-6',
        'knn-7',
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
        holder[index] = searchK(cmd)
    makeBoxPlots(holder,
        methods = methods,
        metrics = metrics,
        name = f'{cmd.out}full_run_metrics.png'
    )

if __name__ == "__main__":
    kSearchWrapper()

