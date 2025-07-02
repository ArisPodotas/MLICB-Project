# Note that if you call all the functions with liner() called within them and
# The clean parameter is on then each test will clean each time liner() is called

from argparse import Namespace
from utils import *
# from hmm import * # Causing some import error I think because we dont have keras
from gmm import CellGenerator
from cluster import *

def testLiner():
    """
    Not really needed but in case someone wants it
    """
    cmd: Namespace = liner()

def testUtils():
    """
    A little test for the whole Utils class
    """
    cmd: Namespace = liner()
    obj: Utils = Utils(dataFolder = cmd.input, outputPath = cmd.out)
    obj.main()

def testAttributes():
    pass

def testConvertToCsv():
    """
    Tests the convertToDataFames method
    """
    cmd: Namespace = liner()
    obj: Utils = Utils(dataFolder = cmd.input, outputPath = cmd.out)
    obj.convertToDataFrames(save = True)

def testDescriptiveStatistics():
    """
    """
    cmd: Namespace = liner()
    obj: Utils = Utils(dataFolder = cmd.input, outputPath = cmd.out)
    obj.main()
    obj.dataFrameHistograms()

def testDataStructure():
    """
    Depricated
    """
    cmd: Namespace = liner()
    query: Utils = Utils(dataFolder = cmd.input, outputPath = cmd.out)
    query.main()

def testGeneDistribution():
    """
    Tests the utils gene Distribution method
    """
    pass

def testGMMHists():
    """
    Tests the ploting functions of the gmm CellGenerator class
    """
    cmd: Namespace = liner()
    generator: CellGenerator = CellGenerator(dataFolder = cmd.input, outputPath = cmd.out)
    generator.showAllDomains(path = cmd.out)

def testGeneratorOfCells():
    """
    Tests the cell making funcitons of the gnerator gmm
    """
    cmd: Namespace = liner()
    gen: CellGenerator = CellGenerator(cmd.input, cmd.out)
    cells: list[dict[str, float]] = gen.generateCells(count = 2, sampler = 'Random')
    gen.cellsToDF(cells, name = 'TestCells')
    gen.plotCellsCommon(cells)
    gen.plotCellsDistinct(cells)

def testMains():
    """
    Tests that all the main methods work smoothly
    """
    cmd: Namespace = liner()
    ut: Utils = Utils(cmd.input, cmd.out)
    gen: CellGenerator = CellGenerator(cmd.input, cmd.out)
    ut.main()
    gen.main()

def testSamplers():
    cmd: Namespace = liner()
    gen: CellGenerator = CellGenerator(cmd.input, cmd.out)
    cellsRN: list[dict[str, float]] = gen.generateCells(count = 2, sampler = 'Random')
    cellsDS: list[dict[str, float]] = gen.generateCells(count = 2, sampler = 'Discreet')
    cellsGN: list[dict[str, float]] = gen.generateCells(count = 2, sampler = 'Gaussian')
    cellsGMM: list[dict[str, float]] = gen.generateCells(count = 2, sampler = 'GMM')
    cellsKDE: list[dict[str, float]] = gen.generateCells(count = 2, sampler = 'Distribution')

if __name__ == "__main__":
    # testLiner()
    # testUtils()
    # testAttributes()
    # testDescriptiveStatistics()
    # testDataStructure()
    # testGMMHists()
    # testGeneratorOfCells()
    testSamplers()
    # testMains()

