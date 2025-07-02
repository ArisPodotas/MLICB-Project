Note:
> I made this folder for our project in machine learning for professor Manolakos

The ./src/ dir is for the python (or other) files.

> Note
Make a big notice that the command line input parameter clean will use ```shutil.rmtree``` which will essentially call an rm -rf for a folder, it's set to the output dir of the script (i.e. ./outputs/) but it uses relative paths from the CWD in the terminal executing the script so be careful.

I have made help menus for the different files so use -h if you need to.

The intent is to run the python files from the ./src/ directory directly (terminal current working directory must be ./.../src/)

Metis needs building the binaries first before usage for now

>Note
Some of the test are out dated and some major changes were facilitated from the start to the end of th scripts so some funciotns may be unused nas ome things might not work
